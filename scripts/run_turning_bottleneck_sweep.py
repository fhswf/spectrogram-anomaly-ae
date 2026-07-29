from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Reshape, UpSampling2D
from tensorflow.keras.models import Sequential
from tqdm.auto import tqdm

from spectrogram_anomaly_ae.turning_cv import (
    fold_balance,
    make_grouped_cv_assignments,
    run_turning_ae_grouped_cv,
)


SCORE_COLUMNS = ["global_mse", "global_mae", "ver_max", "ver_topk"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the turning CNN autoencoder across bottleneck dimensions."
    )
    parser.add_argument("--dims", nargs="+", type=int, default=[8, 16, 32, 64, 128])
    parser.add_argument("--epochs", type=int, default=4000)
    parser.add_argument("--patience", type=int, default=250)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42, help="Legacy single-seed option.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Train each bottleneck dimension once per seed. Defaults to --seed.",
    )
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--n-ver-segments", type=int, default=10)
    parser.add_argument("--ver-top-k", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/bottleneck_sweep"))
    parser.add_argument("--model-dir", type=Path, default=Path("models/bottleneck_sweep"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reference-dim", type=int, default=16)
    parser.add_argument(
        "--grouped-cv",
        action="store_true",
        help="Run repeated stratified grouped k-fold CV by source_run instead of the fixed validation/test sweep.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--save-cv-models",
        action="store_true",
        help="Persist every grouped-CV fold model. Disabled by default to avoid large sweep output.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def load_image(path: Path, image_size: tuple[int, int]) -> np.ndarray:
    return np.asarray(Image.open(path).convert("RGB").resize(image_size), dtype="float32") / 255.0


def load_training_images(dataset_root: Path, image_size: tuple[int, int]) -> np.ndarray:
    folder = dataset_root / "train" / "no_chatter"
    paths = sorted(
        path for path in folder.iterdir() if path.suffix.lower() in {".png", ".jpg", ".jpeg"}
    )
    if not paths:
        raise FileNotFoundError(f"No training images found in {folder}")
    return np.stack([load_image(path, image_size) for path in paths], axis=0)


def load_manifest_images(
    manifest: pd.DataFrame, repo_root: Path, image_size: tuple[int, int]
) -> tuple[np.ndarray, pd.DataFrame]:
    images: list[np.ndarray] = []
    rows: list[pd.Series] = []
    for _, row in manifest.iterrows():
        image_path = repo_root / row["image_path"]
        if image_path.exists():
            images.append(load_image(image_path, image_size))
            rows.append(row)
    if not images:
        raise FileNotFoundError("No manifest images could be loaded.")
    return np.stack(images, axis=0), pd.DataFrame(rows).reset_index(drop=True)


def build_autoencoder(bottleneck_dim: int, learning_rate: float) -> tf.keras.Model:
    model = Sequential(
        [
            Input(shape=(100, 150, 3)),
            Conv2D(4, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 2), padding="same"),
            Conv2D(8, (3, 3), activation="relu", padding="same"),
            MaxPooling2D((2, 3), padding="same"),
            Conv2D(12, (3, 3), activation="relu", padding="same"),
            Flatten(),
            Dense(bottleneck_dim, activation="relu", name=f"bottleneck_{bottleneck_dim}"),
            Dense(25 * 25 * 12, activation="relu"),
            Reshape((25, 25, 12)),
            Conv2D(12, (3, 3), activation="relu", padding="same"),
            UpSampling2D((2, 3)),
            Conv2D(8, (3, 3), activation="relu", padding="same"),
            UpSampling2D((2, 2)),
            Conv2D(4, (3, 3), activation="relu", padding="same"),
            Conv2D(3, (3, 3), activation="sigmoid", padding="same"),
        ]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    return model


class TqdmEpochCallback(tf.keras.callbacks.Callback):
    def __init__(self, total_epochs: int, desc: str, disable: bool = False) -> None:
        super().__init__()
        self.total_epochs = total_epochs
        self.desc = desc
        self.disable = disable
        self.progress: tqdm | None = None
        self.seen_epochs = 0

    def on_train_begin(self, logs: dict | None = None) -> None:
        self.progress = tqdm(
            total=self.total_epochs,
            desc=self.desc,
            unit="epoch",
            leave=True,
            disable=self.disable,
        )

    def on_epoch_end(self, epoch: int, logs: dict | None = None) -> None:
        if self.progress is None:
            return
        logs = logs or {}
        target_seen = epoch + 1
        self.progress.update(max(0, target_seen - self.seen_epochs))
        self.seen_epochs = target_seen
        postfix = {}
        for key in ("loss", "val_loss"):
            if key in logs:
                postfix[key] = f"{logs[key]:.6f}"
        self.progress.set_postfix(postfix)

    def on_train_end(self, logs: dict | None = None) -> None:
        if self.progress is not None:
            self.progress.close()


def vertical_segment_scores(
    error_map: np.ndarray, n_segments: int, top_k: int
) -> tuple[np.ndarray, np.ndarray]:
    width = error_map.shape[2]
    boundaries = np.linspace(0, width, n_segments + 1, dtype=int)
    segment_scores = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right > left:
            segment_scores.append(error_map[:, :, left:right, :].mean(axis=(1, 2, 3)))
    segment_scores_array = np.stack(segment_scores, axis=1)
    ver_max = segment_scores_array.max(axis=1)
    effective_top_k = min(top_k, segment_scores_array.shape[1])
    ver_topk = np.sort(segment_scores_array, axis=1)[:, -effective_top_k:].mean(axis=1)
    return ver_max, ver_topk


def score_reconstructions(
    model: tf.keras.Model,
    images: np.ndarray,
    n_ver_segments: int,
    ver_top_k: int,
) -> pd.DataFrame:
    recon = model.predict(images, verbose=0)
    squared_error = (images - recon) ** 2
    absolute_error = np.abs(images - recon)
    ver_max, ver_topk = vertical_segment_scores(squared_error, n_ver_segments, ver_top_k)
    return pd.DataFrame(
        {
            "global_mse": squared_error.mean(axis=(1, 2, 3)),
            "global_mae": absolute_error.mean(axis=(1, 2, 3)),
            "ver_max": ver_max,
            "ver_topk": ver_topk,
        }
    )


def select_best_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return {
            "threshold": float(np.max(scores)),
            "validation_f1": 0.0,
            "validation_precision": 0.0,
            "validation_recall": 0.0,
        }
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_idx = int(np.nanargmax(f1))
    return {
        "threshold": float(thresholds[best_idx]),
        "validation_f1": float(f1[best_idx]),
        "validation_precision": float(precision[best_idx]),
        "validation_recall": float(recall[best_idx]),
    }


def evaluate_at_threshold(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "pr_auc": float(average_precision_score(y_true, scores)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def score_summary(scores: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for (split, label), group in scores.groupby(["split", "label"], dropna=False):
        for column in columns:
            values = group[column].to_numpy()
            rows.append(
                {
                    "split": split,
                    "label": label,
                    "score": column,
                    "n": int(len(values)),
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "median": float(np.median(values)),
                    "p90": float(np.quantile(values, 0.9)),
                }
            )
    return pd.DataFrame(rows)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def summarize_numeric(
    frame: pd.DataFrame,
    group_columns: list[str],
    numeric_columns: list[str],
) -> pd.DataFrame:
    available_numeric_columns = [column for column in numeric_columns if column in frame.columns]
    if frame.empty or not available_numeric_columns:
        return pd.DataFrame()

    summary = (
        frame.groupby(group_columns, dropna=False)[available_numeric_columns]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]

    for column in available_numeric_columns:
        count_column = f"{column}_count"
        std_column = f"{column}_std"
        if count_column in summary.columns and std_column in summary.columns:
            sem = summary[std_column] / np.sqrt(summary[count_column].clip(lower=1))
            summary[f"{column}_ci95"] = 1.96 * sem.fillna(0.0)
    return summary


def metric_deltas_vs_reference(metrics: pd.DataFrame, reference_dim: int) -> pd.DataFrame:
    if reference_dim not in set(metrics["bottleneck_dim"]):
        return pd.DataFrame()

    merge_columns = ["dataset", "method", "score"]
    for candidate in ("seed", "cv_seed", "cv_fold"):
        if candidate in metrics.columns:
            merge_columns.append(candidate)
    numeric_columns = [
        "validation_f1",
        "validation_precision",
        "validation_recall",
        "pr_auc",
        "f1",
        "precision",
        "recall",
        "tn",
        "fp",
        "fn",
        "tp",
    ]
    reference = metrics[metrics["bottleneck_dim"] == reference_dim][
        merge_columns + numeric_columns
    ].copy()
    reference = reference.rename(columns={column: f"{column}_reference" for column in numeric_columns})
    deltas = metrics.merge(reference, on=merge_columns, how="inner")
    deltas["reference_dim"] = reference_dim
    for column in numeric_columns:
        deltas[f"{column}_delta"] = deltas[column] - deltas[f"{column}_reference"]
    return deltas


def run_grouped_cv_sweep(
    *,
    args: argparse.Namespace,
    seeds: list[int],
    repo_root: Path,
    manifest: pd.DataFrame,
    image_size: tuple[int, int],
) -> None:
    all_metric_frames: list[pd.DataFrame] = []
    all_score_frames: list[pd.DataFrame] = []
    all_summary_frames: list[pd.DataFrame] = []
    all_balance_frames: list[pd.DataFrame] = []
    threshold_payload: dict[str, object] = {
        "threshold_protocol": "fold_internal_best_f1",
        "cv_folds": args.cv_folds,
        "seeds": seeds,
        "bottleneck_dims": args.dims,
        "bottleneck_thresholds": {},
    }

    for seed in seeds:
        assignments = make_grouped_cv_assignments(manifest, n_splits=args.cv_folds, seed=seed)
        balance = fold_balance(assignments)
        balance.insert(0, "cv_seed", seed)
        all_balance_frames.append(balance)

    for bottleneck_dim in tqdm(
        args.dims,
        desc="Grouped-CV bottleneck sweep",
        unit="dim",
        disable=not args.progress,
    ):
        run_name = f"bn{bottleneck_dim}_grouped_cv"
        metrics_path = args.output_dir / f"metrics_{run_name}.csv"
        scores_path = args.output_dir / f"scores_{run_name}.csv"
        summary_path = args.output_dir / f"metrics_summary_{run_name}.csv"
        thresholds_path = args.output_dir / f"thresholds_{run_name}.json"
        cv_model_dir = args.model_dir / run_name if args.save_cv_models else None

        if (
            metrics_path.exists()
            and scores_path.exists()
            and summary_path.exists()
            and thresholds_path.exists()
            and not args.overwrite
        ):
            print(f"Skipping {run_name}: existing outputs found.", flush=True)
            metrics_df = pd.read_csv(metrics_path)
            scores_df = pd.read_csv(scores_path)
            summary_df = pd.read_csv(summary_path)
            with thresholds_path.open("r", encoding="utf-8") as file:
                thresholds = json.load(file)
        else:
            print(f"Training grouped CV {run_name}...", flush=True)
            metrics_df, scores_df, summary_df, thresholds = run_turning_ae_grouped_cv(
                manifest,
                repo_root,
                n_splits=args.cv_folds,
                seeds=seeds,
                bottleneck_dim=bottleneck_dim,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                image_size=image_size,
                n_ver_segments=args.n_ver_segments,
                ver_top_k=args.ver_top_k,
                model_dir=cv_model_dir,
            )
            metrics_df.to_csv(metrics_path, index=False)
            scores_df.to_csv(scores_path, index=False)
            summary_df.to_csv(summary_path, index=False)
            write_json(thresholds_path, thresholds)

        all_metric_frames.append(metrics_df)
        all_score_frames.append(scores_df)
        all_summary_frames.append(summary_df)
        threshold_payload["bottleneck_thresholds"][str(bottleneck_dim)] = thresholds

    all_metrics = pd.concat(all_metric_frames, ignore_index=True)
    all_scores = pd.concat(all_score_frames, ignore_index=True)
    all_summaries = pd.concat(all_summary_frames, ignore_index=True)
    fold_balance_df = pd.concat(all_balance_frames, ignore_index=True)

    all_metrics.to_csv(args.output_dir / "metrics_grouped_cv_all.csv", index=False)
    all_scores.to_csv(args.output_dir / "scores_grouped_cv_all.csv", index=False)
    all_summaries.to_csv(args.output_dir / "metrics_summary_grouped_cv_by_dim.csv", index=False)
    fold_balance_df.to_csv(args.output_dir / "fold_balance_grouped_cv.csv", index=False)
    write_json(args.output_dir / "thresholds_grouped_cv_all.json", threshold_payload)

    metrics_summary = summarize_numeric(
        all_metrics,
        group_columns=["bottleneck_dim", "dataset", "method", "score"],
        numeric_columns=[
            "validation_f1",
            "validation_precision",
            "validation_recall",
            "pr_auc",
            "f1",
            "precision",
            "recall",
            "tn",
            "fp",
            "fn",
            "tp",
            "epochs_trained",
            "best_val_loss",
        ],
    )
    metrics_summary.to_csv(args.output_dir / "metrics_summary_grouped_cv.csv", index=False)

    metric_deltas = metric_deltas_vs_reference(all_metrics, args.reference_dim)
    if not metric_deltas.empty:
        metric_deltas.to_csv(
            args.output_dir / f"metric_deltas_grouped_cv_vs_bn{args.reference_dim}.csv",
            index=False,
        )
        delta_columns = [column for column in metric_deltas.columns if column.endswith("_delta")]
        metric_delta_summary = summarize_numeric(
            metric_deltas,
            group_columns=["reference_dim", "bottleneck_dim", "dataset", "method", "score"],
            numeric_columns=delta_columns,
        )
        metric_delta_summary.to_csv(
            args.output_dir / f"metric_delta_summary_grouped_cv_vs_bn{args.reference_dim}.csv",
            index=False,
        )

    print(f"Wrote grouped-CV sweep outputs to {args.output_dir}", flush=True)


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]
    repo_root = Path.cwd()
    dataset_root = repo_root / "data" / "02_spectrograms_150x100px_dataset"
    manifest_path = repo_root / "reports" / "manifests" / "turning_split_seed42.csv"
    image_size = (150, 100)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    full_manifest = pd.read_csv(manifest_path)
    if args.grouped_cv:
        run_grouped_cv_sweep(
            args=args,
            seeds=seeds,
            repo_root=repo_root,
            manifest=full_manifest,
            image_size=image_size,
        )
        return

    manifest = full_manifest[
        (full_manifest["source_dataset"] == "turning")
        & full_manifest["split"].isin(["validation", "test"])
        & full_manifest["image_path"].fillna("").ne("")
    ].copy()
    manifest["target"] = (manifest["label"] == "chatter").astype(int)

    validation_images, validation_rows = load_manifest_images(
        manifest[manifest["split"] == "validation"], repo_root, image_size
    )
    test_images, test_rows = load_manifest_images(
        manifest[manifest["split"] == "test"], repo_root, image_size
    )

    training_images = load_training_images(dataset_root, image_size)

    all_metric_rows: list[dict[str, object]] = []
    all_fit_rows: list[dict[str, object]] = []
    all_score_stat_rows: list[pd.DataFrame] = []
    run_grid = [(bottleneck_dim, seed) for bottleneck_dim in args.dims for seed in seeds]

    for bottleneck_dim, seed in tqdm(
        run_grid, desc="Bottleneck/seed sweep", unit="run", disable=not args.progress
    ):
        run_name = f"bn{bottleneck_dim}_seed{seed}"
        metrics_path = args.output_dir / f"metrics_{run_name}.csv"
        history_path = args.output_dir / f"history_{run_name}.csv"
        scores_path = args.output_dir / f"scores_{run_name}.csv"
        stats_path = args.output_dir / f"score_stats_{run_name}.csv"
        thresholds_path = args.output_dir / f"thresholds_{run_name}.json"
        model_path = args.model_dir / f"ae_bn{bottleneck_dim}_turning_seed{seed}.keras"

        if metrics_path.exists() and history_path.exists() and scores_path.exists() and not args.overwrite:
            print(f"Skipping {run_name}: existing outputs found.", flush=True)
            existing_metrics = pd.read_csv(metrics_path)
            existing_history = pd.read_csv(history_path)
            existing_stats = pd.read_csv(stats_path)
            if "seed" not in existing_metrics.columns:
                existing_metrics.insert(1, "seed", seed)
            all_metric_rows.extend(existing_metrics.to_dict("records"))
            all_fit_rows.append(
                {
                    "bottleneck_dim": bottleneck_dim,
                    "seed": seed,
                    "epochs_trained": int(existing_history["epoch"].max() + 1),
                    "best_epoch": int(existing_history["val_loss"].idxmin() + 1),
                    "final_loss": float(existing_history["loss"].iloc[-1]),
                    "final_val_loss": float(existing_history["val_loss"].iloc[-1]),
                    "best_val_loss": float(existing_history["val_loss"].min()),
                    "model_path": str(model_path),
                }
            )
            existing_stats.insert(0, "bottleneck_dim", bottleneck_dim)
            existing_stats.insert(1, "seed", seed)
            all_score_stat_rows.append(existing_stats)
            continue

        print(f"Training {run_name}...", flush=True)
        tf.keras.backend.clear_session()
        set_seed(seed)
        x_train, x_val = train_test_split(
            training_images, test_size=0.2, random_state=seed, shuffle=True
        )
        model = build_autoencoder(bottleneck_dim, args.learning_rate)
        callbacks = [
            TqdmEpochCallback(
                total_epochs=args.epochs,
                desc=run_name,
                disable=not args.progress,
            ),
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.patience,
                min_delta=1e-6,
                restore_best_weights=True,
                verbose=1,
            )
        ]
        history = model.fit(
            x_train,
            x_train,
            epochs=args.epochs,
            batch_size=args.batch_size,
            validation_data=(x_val, x_val),
            shuffle=True,
            verbose=0,
            callbacks=callbacks,
        )

        history_df = pd.DataFrame(history.history)
        history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
        history_df.to_csv(history_path, index=False)
        model.save(model_path)

        internal_train_scores = score_reconstructions(
            model, x_train, args.n_ver_segments, args.ver_top_k
        )
        internal_val_scores = score_reconstructions(model, x_val, args.n_ver_segments, args.ver_top_k)

        validation_scores = pd.concat(
            [
                validation_rows.reset_index(drop=True),
                score_reconstructions(model, validation_images, args.n_ver_segments, args.ver_top_k),
            ],
            axis=1,
        )
        test_scores = pd.concat(
            [
                test_rows.reset_index(drop=True),
                score_reconstructions(model, test_images, args.n_ver_segments, args.ver_top_k),
            ],
            axis=1,
        )
        all_scores = pd.concat([validation_scores, test_scores], ignore_index=True)
        all_scores.to_csv(scores_path, index=False)

        y_val = validation_scores["target"].to_numpy()
        y_test = test_scores["target"].to_numpy()
        thresholds: dict[str, dict[str, float]] = {}
        metric_rows: list[dict[str, object]] = []

        for score_name in SCORE_COLUMNS:
            selected = select_best_f1_threshold(y_val, validation_scores[score_name].to_numpy())
            thresholds[score_name] = selected
            test_metrics = evaluate_at_threshold(
                y_test, test_scores[score_name].to_numpy(), selected["threshold"]
            )
            metric_rows.append(
                {
                    "bottleneck_dim": bottleneck_dim,
                    "seed": seed,
                    "dataset": "turning",
                    "method": "cnn_ae",
                    "score": score_name,
                    "threshold": selected["threshold"],
                    "validation_f1": selected["validation_f1"],
                    "validation_precision": selected["validation_precision"],
                    "validation_recall": selected["validation_recall"],
                    **test_metrics,
                }
            )

        write_json(thresholds_path, thresholds)
        metrics_df = pd.DataFrame(metric_rows)
        metrics_df.to_csv(metrics_path, index=False)
        all_metric_rows.extend(metric_rows)

        stats_df = score_summary(all_scores, SCORE_COLUMNS)
        stats_df.to_csv(stats_path, index=False)
        stats_with_dim = stats_df.copy()
        stats_with_dim.insert(0, "bottleneck_dim", bottleneck_dim)
        stats_with_dim.insert(1, "seed", seed)
        all_score_stat_rows.append(stats_with_dim)

        fit_row = {
            "bottleneck_dim": bottleneck_dim,
            "seed": seed,
            "epochs_trained": int(len(history_df)),
            "best_epoch": int(history_df["val_loss"].idxmin() + 1),
            "final_loss": float(history_df["loss"].iloc[-1]),
            "final_val_loss": float(history_df["val_loss"].iloc[-1]),
            "best_val_loss": float(history_df["val_loss"].min()),
            "internal_train_global_mse_mean": float(internal_train_scores["global_mse"].mean()),
            "internal_val_global_mse_mean": float(internal_val_scores["global_mse"].mean()),
            "model_path": str(model_path),
        }
        all_fit_rows.append(fit_row)

        print(
            f"{run_name}: best val_loss={fit_row['best_val_loss']:.6f}, "
            f"global_mse PR-AUC={metrics_df.loc[metrics_df['score'] == 'global_mse', 'pr_auc'].iloc[0]:.6f}, "
            f"ver_topk PR-AUC={metrics_df.loc[metrics_df['score'] == 'ver_topk', 'pr_auc'].iloc[0]:.6f}",
            flush=True,
        )

    all_metrics = pd.DataFrame(all_metric_rows)
    all_metrics.to_csv(args.output_dir / "metrics_all.csv", index=False)
    metrics_summary = summarize_numeric(
        all_metrics,
        group_columns=["bottleneck_dim", "dataset", "method", "score"],
        numeric_columns=[
            "validation_f1",
            "validation_precision",
            "validation_recall",
            "pr_auc",
            "f1",
            "precision",
            "recall",
            "tn",
            "fp",
            "fn",
            "tp",
        ],
    )
    metrics_summary.to_csv(args.output_dir / "metrics_summary.csv", index=False)

    metric_deltas = metric_deltas_vs_reference(all_metrics, args.reference_dim)
    if not metric_deltas.empty:
        metric_deltas.to_csv(
            args.output_dir / f"metric_deltas_vs_bn{args.reference_dim}.csv", index=False
        )
        delta_columns = [column for column in metric_deltas.columns if column.endswith("_delta")]
        metric_delta_summary = summarize_numeric(
            metric_deltas,
            group_columns=["reference_dim", "bottleneck_dim", "dataset", "method", "score"],
            numeric_columns=delta_columns,
        )
        metric_delta_summary.to_csv(
            args.output_dir / f"metric_delta_summary_vs_bn{args.reference_dim}.csv",
            index=False,
        )

    fit_summary = pd.DataFrame(all_fit_rows)
    fit_summary.to_csv(args.output_dir / "fit_summary.csv", index=False)
    summarize_numeric(
        fit_summary,
        group_columns=["bottleneck_dim"],
        numeric_columns=[
            "epochs_trained",
            "best_epoch",
            "final_loss",
            "final_val_loss",
            "best_val_loss",
            "internal_train_global_mse_mean",
            "internal_val_global_mse_mean",
        ],
    ).to_csv(args.output_dir / "fit_summary_by_dim.csv", index=False)

    if all_score_stat_rows:
        score_stats_all = pd.concat(all_score_stat_rows, ignore_index=True)
        score_stats_all.to_csv(args.output_dir / "score_stats_all.csv", index=False)
        summarize_numeric(
            score_stats_all,
            group_columns=["bottleneck_dim", "split", "label", "score"],
            numeric_columns=["mean", "std", "median", "p90"],
        ).to_csv(
            args.output_dir / "score_stats_summary.csv",
            index=False,
        )
    print(f"Wrote sweep outputs to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
