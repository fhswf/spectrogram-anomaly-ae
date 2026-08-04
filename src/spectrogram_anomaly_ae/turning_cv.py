"""Grouped cross-validation helpers for the turning anomaly experiments."""

from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable, Iterable

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM


AE_SCORE_COLUMNS = ["global_mse", "global_mae", "ver_max", "ver_topk"]
BASELINE_SCORE_NAME = "anomaly_score"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def set_seed(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf

        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def prepare_turning_manifest(
    manifest: pd.DataFrame,
    *,
    source_dataset: str = "turning",
) -> pd.DataFrame:
    """Return a CV-ready turning manifest with target and dataset columns."""

    required = {"source_dataset", "source_run", "label", "image_path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    prepared = manifest[
        (manifest["source_dataset"] == source_dataset)
        & manifest["image_path"].fillna("").ne("")
    ].copy()
    prepared["dataset"] = source_dataset
    prepared["target"] = (prepared["label"] == "chatter").astype(int)
    prepared = prepared.reset_index(drop=True)

    if prepared.empty:
        raise ValueError("No turning rows with image paths found.")
    if prepared["target"].nunique() < 2:
        raise ValueError("Grouped CV requires both no_chatter and chatter rows.")
    return prepared


def make_grouped_cv_assignments(
    manifest: pd.DataFrame,
    *,
    n_splits: int = 5,
    seed: int = 42,
) -> pd.DataFrame:
    """Assign stratified grouped CV folds, keeping source_run together."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")

    prepared = prepare_turning_manifest(manifest)
    positive_groups = prepared.loc[prepared["target"] == 1, "source_run"].nunique()
    if positive_groups < n_splits:
        raise ValueError(
            f"n_splits={n_splits} exceeds the {positive_groups} chatter-bearing groups."
        )

    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    assigned = prepared.copy()
    assigned["cv_fold"] = -1
    for fold, (_, eval_idx) in enumerate(
        splitter.split(assigned, assigned["target"], groups=assigned["source_run"])
    ):
        assigned.loc[eval_idx, "cv_fold"] = fold

    if (assigned["cv_fold"] < 0).any():
        raise RuntimeError("Some samples were not assigned to a CV fold.")
    return assigned


def fold_balance(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fold, fold_rows in assignments.groupby("cv_fold"):
        rows.append(
            {
                "cv_fold": int(fold),
                "n_samples": int(len(fold_rows)),
                "n_runs": int(fold_rows["source_run"].nunique()),
                "n_chatter": int((fold_rows["target"] == 1).sum()),
                "n_no_chatter": int((fold_rows["target"] == 0).sum()),
                "n_chatter_runs": int(
                    fold_rows.loc[fold_rows["target"] == 1, "source_run"].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def load_rgb_images(
    rows: pd.DataFrame,
    repo_root: Path,
    *,
    image_size: tuple[int, int] = (150, 100),
) -> tuple[np.ndarray, pd.DataFrame]:
    images: list[np.ndarray] = []
    loaded_rows = []
    for _, row in rows.iterrows():
        image_path = repo_root / row["image_path"]
        if not image_path.exists():
            continue
        image = Image.open(image_path).convert("RGB").resize(image_size)
        images.append(np.asarray(image, dtype="float32") / 255.0)
        loaded_rows.append(row)

    if not images:
        raise ValueError("No images loaded.")
    return np.stack(images, axis=0), pd.DataFrame(loaded_rows).reset_index(drop=True)


def open_resized_image(image_path: Path, image_size: tuple[int, int], mode: str) -> np.ndarray:
    image = Image.open(image_path).convert(mode).resize(image_size)
    return np.asarray(image, dtype="float32") / 255.0


def image_vector(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    return open_resized_image(image_path, image_size, "L").reshape(-1)


def image_descriptor(image_path: Path, image_size: tuple[int, int]) -> np.ndarray:
    arr = open_resized_image(image_path, image_size, "L")
    grad_y, grad_x = np.gradient(arr)
    hist, _ = np.histogram(arr, bins=32, range=(0.0, 1.0), density=True)
    percentiles = np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])
    summary = np.array(
        [
            arr.mean(),
            arr.std(),
            arr.min(),
            arr.max(),
            np.abs(grad_x).mean(),
            np.abs(grad_y).mean(),
            np.sqrt(grad_x**2 + grad_y**2).mean(),
        ],
        dtype="float32",
    )
    return np.concatenate(
        [
            summary,
            percentiles.astype("float32"),
            hist.astype("float32"),
            arr.mean(axis=0),
            arr.mean(axis=1),
            arr.std(axis=0),
            arr.std(axis=1),
        ]
    ).astype("float32")


def load_matrix(
    rows: pd.DataFrame,
    repo_root: Path,
    extractor: Callable[[Path, tuple[int, int]], np.ndarray],
    *,
    image_size: tuple[int, int] = (150, 100),
) -> np.ndarray:
    values = [extractor(repo_root / image_path, image_size) for image_path in rows["image_path"]]
    if not values:
        raise ValueError("No images loaded.")
    return np.stack(values, axis=0)


def build_cnn_autoencoder(
    *,
    bottleneck_dim: int = 16,
    learning_rate: float = 1e-4,
    input_shape: tuple[int, int, int] = (100, 150, 3),
):
    import tensorflow as tf
    from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input, MaxPooling2D, Reshape, UpSampling2D
    from tensorflow.keras.models import Sequential

    model = Sequential(
        [
            Input(shape=input_shape),
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


def vertical_segment_scores(
    error_map: np.ndarray,
    n_segments: int,
    top_k: int,
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
    model,
    images: np.ndarray,
    *,
    n_ver_segments: int = 10,
    ver_top_k: int = 3,
    batch_size: int = 32,
) -> pd.DataFrame:
    recon_batches = []
    for start in range(0, len(images), batch_size):
        batch = images[start : start + batch_size]
        recon_batches.append(np.asarray(model(batch, training=False)))
    recon = np.concatenate(recon_batches, axis=0)
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


def evaluate_at_threshold(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
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


def summarize_cv_metrics(
    metrics: pd.DataFrame,
    *,
    group_columns: list[str],
    numeric_columns: Iterable[str] = (
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
    ),
) -> pd.DataFrame:
    available = [column for column in numeric_columns if column in metrics.columns]
    if metrics.empty or not available:
        return pd.DataFrame()

    summary = (
        metrics.groupby(group_columns, dropna=False)[available]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
    )
    summary.columns = [
        "_".join(str(part) for part in column if part)
        if isinstance(column, tuple)
        else str(column)
        for column in summary.columns
    ]
    for column in available:
        count_column = f"{column}_count"
        std_column = f"{column}_std"
        if count_column in summary.columns and std_column in summary.columns:
            sem = summary[std_column] / np.sqrt(summary[count_column].clip(lower=1))
            summary[f"{column}_ci95"] = 1.96 * sem.fillna(0.0)
    return summary


def _score_metadata_frame(
    rows: pd.DataFrame,
    *,
    method: str,
    score_name: str,
    score_values: np.ndarray,
    cv_seed: int,
    cv_fold: int,
    extra_columns: dict[str, object] | None = None,
) -> pd.DataFrame:
    source_split = rows["split"].to_numpy() if "split" in rows else np.repeat("", len(rows))
    frame = pd.DataFrame(
        {
            "dataset": rows["dataset"].to_numpy(),
            "method": method,
            "score": score_name,
            "sample_id": rows["sample_id"].to_numpy(),
            "source_run": rows["source_run"].to_numpy(),
            "source_split": source_split,
            "cv_seed": cv_seed,
            "cv_fold": cv_fold,
            "split": "cv_evaluation",
            "label": rows["label"].to_numpy(),
            "target": rows["target"].to_numpy(),
            "score_value": score_values,
        }
    )
    for column, value in (extra_columns or {}).items():
        frame[column] = value
    return frame


def _metric_row(
    *,
    dataset_name: str,
    method: str,
    score_name: str,
    selected: dict[str, float],
    metrics: dict[str, float | int],
    cv_seed: int,
    cv_fold: int,
    train_nominal: pd.DataFrame,
    evaluation: pd.DataFrame,
) -> dict[str, object]:
    return {
        "dataset": dataset_name,
        "method": method,
        "score": score_name,
        "cv_seed": cv_seed,
        "cv_fold": cv_fold,
        "threshold_protocol": "fold_internal_best_f1",
        "threshold": selected["threshold"],
        "validation_f1": selected["validation_f1"],
        "validation_precision": selected["validation_precision"],
        "validation_recall": selected["validation_recall"],
        "n_train_nominal": int(len(train_nominal)),
        "n_eval": int(len(evaluation)),
        "n_eval_chatter": int((evaluation["target"] == 1).sum()),
        "n_eval_no_chatter": int((evaluation["target"] == 0).sum()),
        "n_eval_runs": int(evaluation["source_run"].nunique()),
        **metrics,
    }


def run_turning_baseline_grouped_cv(
    manifest: pd.DataFrame,
    repo_root: Path,
    *,
    n_splits: int = 5,
    seeds: Iterable[int] = (42,),
    image_size: tuple[int, int] = (150, 100),
    pca_components: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run grouped CV for turning descriptor/PCA baselines."""

    metric_rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    threshold_payload: dict[str, object] = {
        "threshold_protocol": "fold_internal_best_f1",
        "folds": [],
    }

    for cv_seed in seeds:
        assignments = make_grouped_cv_assignments(manifest, n_splits=n_splits, seed=cv_seed)
        for cv_fold in range(n_splits):
            train_nominal = assignments[
                (assignments["cv_fold"] != cv_fold) & (assignments["target"] == 0)
            ].reset_index(drop=True)
            evaluation = assignments[assignments["cv_fold"] == cv_fold].reset_index(drop=True)
            y_eval = evaluation["target"].to_numpy()

            X_train_desc = load_matrix(
                train_nominal, repo_root, image_descriptor, image_size=image_size
            )
            X_eval_desc = load_matrix(evaluation, repo_root, image_descriptor, image_size=image_size)
            scaler = StandardScaler().fit(X_train_desc)
            X_train_desc = scaler.transform(X_train_desc)
            X_eval_desc = scaler.transform(X_eval_desc)

            ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_train_desc)
            iforest = IsolationForest(random_state=cv_seed, contamination="auto").fit(X_train_desc)
            descriptor_models = {
                "one_class_svm_image_features": -ocsvm.decision_function(X_eval_desc),
                "isolation_forest_image_features": -iforest.decision_function(X_eval_desc),
            }

            fold_thresholds = {}
            for method, eval_scores in descriptor_models.items():
                selected = select_best_f1_threshold(y_eval, eval_scores)
                metrics = evaluate_at_threshold(y_eval, eval_scores, selected["threshold"])
                metric_rows.append(
                    _metric_row(
                        dataset_name="turning",
                        method=method,
                        score_name=BASELINE_SCORE_NAME,
                        selected=selected,
                        metrics=metrics,
                        cv_seed=cv_seed,
                        cv_fold=cv_fold,
                        train_nominal=train_nominal,
                        evaluation=evaluation,
                    )
                )
                score_frames.append(
                    _score_metadata_frame(
                        evaluation,
                        method=method,
                        score_name=BASELINE_SCORE_NAME,
                        score_values=eval_scores,
                        cv_seed=cv_seed,
                        cv_fold=cv_fold,
                    )
                )
                fold_thresholds[method] = selected

            X_train_vec = load_matrix(train_nominal, repo_root, image_vector, image_size=image_size)
            X_eval_vec = load_matrix(evaluation, repo_root, image_vector, image_size=image_size)
            n_components = min(pca_components, X_train_vec.shape[0], X_train_vec.shape[1])
            pca = PCA(n_components=n_components, random_state=cv_seed, svd_solver="randomized").fit(
                X_train_vec
            )
            eval_recon = pca.inverse_transform(pca.transform(X_eval_vec))
            eval_scores = np.mean((X_eval_vec - eval_recon) ** 2, axis=1)
            method = "pca_image_reconstruction"
            selected = select_best_f1_threshold(y_eval, eval_scores)
            metrics = evaluate_at_threshold(y_eval, eval_scores, selected["threshold"])
            metric_rows.append(
                _metric_row(
                    dataset_name="turning",
                    method=method,
                    score_name="reconstruction_mse",
                    selected=selected,
                    metrics=metrics,
                    cv_seed=cv_seed,
                    cv_fold=cv_fold,
                    train_nominal=train_nominal,
                    evaluation=evaluation,
                )
            )
            score_frames.append(
                _score_metadata_frame(
                    evaluation,
                    method=method,
                    score_name="reconstruction_mse",
                    score_values=eval_scores,
                    cv_seed=cv_seed,
                    cv_fold=cv_fold,
                )
            )
            fold_thresholds[method] = selected
            threshold_payload["folds"].append(
                {
                    "cv_seed": cv_seed,
                    "cv_fold": cv_fold,
                    "thresholds": fold_thresholds,
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    scores_df = pd.concat(score_frames, ignore_index=True)
    summary_df = summarize_cv_metrics(
        metrics_df, group_columns=["dataset", "method", "score"]
    )
    return metrics_df, scores_df, summary_df, threshold_payload


def run_turning_ae_grouped_cv(
    manifest: pd.DataFrame,
    repo_root: Path,
    *,
    n_splits: int = 5,
    seeds: Iterable[int] = (42,),
    bottleneck_dim: int = 16,
    epochs: int = 4000,
    patience: int = 250,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    image_size: tuple[int, int] = (150, 100),
    n_ver_segments: int = 10,
    ver_top_k: int = 3,
    model_dir: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Train and evaluate the CNN autoencoder with repeated grouped CV."""

    import tensorflow as tf

    metric_rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    threshold_payload: dict[str, object] = {
        "threshold_protocol": "fold_internal_best_f1",
        "folds": [],
    }

    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    for cv_seed in seeds:
        assignments = make_grouped_cv_assignments(manifest, n_splits=n_splits, seed=cv_seed)
        for cv_fold in range(n_splits):
            tf.keras.backend.clear_session()
            set_seed(cv_seed + cv_fold)

            train_nominal = assignments[
                (assignments["cv_fold"] != cv_fold) & (assignments["target"] == 0)
            ].reset_index(drop=True)
            evaluation = assignments[assignments["cv_fold"] == cv_fold].reset_index(drop=True)

            train_images, train_rows = load_rgb_images(
                train_nominal, repo_root, image_size=image_size
            )
            eval_images, eval_rows = load_rgb_images(evaluation, repo_root, image_size=image_size)
            y_eval = eval_rows["target"].to_numpy()

            x_train, x_stop = train_test_split(
                train_images,
                test_size=0.2,
                random_state=cv_seed + cv_fold,
                shuffle=True,
            )
            model = build_cnn_autoencoder(
                bottleneck_dim=bottleneck_dim,
                learning_rate=learning_rate,
            )
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=patience,
                    min_delta=1e-6,
                    restore_best_weights=True,
                    verbose=0,
                )
            ]
            history = model.fit(
                x_train,
                x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_stop, x_stop),
                shuffle=True,
                verbose=0,
                callbacks=callbacks,
            )

            if model_dir is not None:
                model.save(model_dir / f"ae_bn{bottleneck_dim}_turning_cv_seed{cv_seed}_fold{cv_fold}.keras")

            eval_scores = pd.concat(
                [
                    eval_rows.reset_index(drop=True),
                    score_reconstructions(
                        model,
                        eval_images,
                        n_ver_segments=n_ver_segments,
                        ver_top_k=ver_top_k,
                        batch_size=batch_size,
                    ),
                ],
                axis=1,
            )

            fold_thresholds = {}
            for score_name in AE_SCORE_COLUMNS:
                values = eval_scores[score_name].to_numpy()
                selected = select_best_f1_threshold(y_eval, values)
                metrics = evaluate_at_threshold(y_eval, values, selected["threshold"])
                row = _metric_row(
                    dataset_name="turning",
                    method="cnn_ae",
                    score_name=score_name,
                    selected=selected,
                    metrics=metrics,
                    cv_seed=cv_seed,
                    cv_fold=cv_fold,
                    train_nominal=train_rows,
                    evaluation=eval_rows,
                )
                row.update(
                    {
                        "bottleneck_dim": bottleneck_dim,
                        "epochs_trained": int(len(history.history["loss"])),
                        "best_val_loss": float(np.min(history.history["val_loss"])),
                    }
                )
                metric_rows.append(row)
                score_frames.append(
                    _score_metadata_frame(
                        eval_rows,
                        method="cnn_ae",
                        score_name=score_name,
                        score_values=values,
                        cv_seed=cv_seed,
                        cv_fold=cv_fold,
                        extra_columns={"bottleneck_dim": bottleneck_dim},
                    )
                )
                fold_thresholds[score_name] = selected

            threshold_payload["folds"].append(
                {
                    "cv_seed": cv_seed,
                    "cv_fold": cv_fold,
                    "thresholds": fold_thresholds,
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    scores_df = pd.concat(score_frames, ignore_index=True)
    summary_df = summarize_cv_metrics(
        metrics_df, group_columns=["dataset", "method", "score", "bottleneck_dim"]
    )
    return metrics_df, scores_df, summary_df, threshold_payload
