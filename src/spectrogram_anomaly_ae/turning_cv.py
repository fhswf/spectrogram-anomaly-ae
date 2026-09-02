"""Grouped cross-validation helpers for the turning anomaly experiments."""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
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
NESTED_THRESHOLD_PROTOCOL = "nested_grouped_reused_pairwise_inner_best_f1"


class TqdmEpochCallback:
    def __init__(self, total_epochs: int, desc: str, disable: bool = False) -> None:
        self.total_epochs = total_epochs
        self.desc = desc
        self.disable = disable
        self.progress = None
        self.seen_epochs = 0

    def set_model(self, model: object) -> None:
        self.model = model

    def set_params(self, params: dict | None) -> None:
        self.params = params or {}

    def on_batch_begin(self, batch: int, logs: dict | None = None) -> None:
        pass

    def on_batch_end(self, batch: int, logs: dict | None = None) -> None:
        pass

    def on_epoch_begin(self, epoch: int, logs: dict | None = None) -> None:
        pass

    def on_train_begin(self, logs: dict | None = None) -> None:
        if self.disable:
            return
        from tqdm.auto import tqdm

        self.progress = tqdm(
            total=self.total_epochs,
            desc=self.desc,
            unit="epoch",
            leave=False,
            dynamic_ncols=True,
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

    def on_train_batch_begin(self, batch: int, logs: dict | None = None) -> None:
        pass

    def on_train_batch_end(self, batch: int, logs: dict | None = None) -> None:
        pass

    def on_test_begin(self, logs: dict | None = None) -> None:
        pass

    def on_test_end(self, logs: dict | None = None) -> None:
        pass

    def on_test_batch_begin(self, batch: int, logs: dict | None = None) -> None:
        pass

    def on_test_batch_end(self, batch: int, logs: dict | None = None) -> None:
        pass

    def on_predict_begin(self, logs: dict | None = None) -> None:
        pass

    def on_predict_end(self, logs: dict | None = None) -> None:
        pass

    def on_predict_batch_begin(self, batch: int, logs: dict | None = None) -> None:
        pass

    def on_predict_batch_end(self, batch: int, logs: dict | None = None) -> None:
        pass


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


def _resolve_reusable_inner_splits(n_splits: int, inner_splits: int | None) -> int:
    """Require inner CV to reuse the fixed outer fold partition."""

    expected = n_splits - 1
    if expected < 2:
        raise ValueError("Reusable nested grouped CV requires at least 3 outer folds.")
    resolved = expected if inner_splits is None else inner_splits
    if resolved != expected:
        raise ValueError(
            "Reusable nested grouped CV requires inner_cv_folds="
            f"n_splits - 1 ({expected}); got {resolved}."
        )
    return resolved


def _fold_pair(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first < second else (second, first)


def _fold_pairs(n_splits: int) -> list[tuple[int, int]]:
    return list(combinations(range(n_splits), 2))


def _configure_worker_gpu(tf: object, gpu_index: int | None) -> None:
    """Restrict a spawned worker to one logical GPU before model creation."""

    if gpu_index is None:
        return
    gpus = tf.config.list_physical_devices("GPU")
    if gpu_index >= len(gpus):
        raise RuntimeError(
            f"Worker requested GPU {gpu_index}, but only {len(gpus)} GPUs are visible."
        )
    tf.config.set_visible_devices(gpus[gpu_index], "GPU")


def _run_parallel_jobs(
    worker: Callable[[dict[str, object]], dict[str, object]],
    jobs: list[dict[str, object]],
    *,
    gpu_count: int,
    progress: bool,
    description: str,
) -> list[dict[str, object]]:
    """Run independent TensorFlow jobs concurrently, one per GPU."""

    if gpu_count < 2:
        raise ValueError("Parallel GPU execution requires at least two GPUs.")

    context = mp.get_context("spawn")
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=gpu_count,
        mp_context=context,
    ) as executor:
        futures = {}
        for job_index, job in enumerate(jobs):
            assigned_job = dict(job)
            assigned_job["gpu_index"] = job_index % gpu_count
            if progress:
                print(
                    f"Starting {assigned_job['description']} on GPU "
                    f"{assigned_job['gpu_index']}",
                    flush=True,
                )
            future = executor.submit(worker, assigned_job)
            futures[future] = assigned_job

        completed = as_completed(futures)
        if progress:
            from tqdm.auto import tqdm

            completed = tqdm(
                completed,
                total=len(futures),
                desc=description,
                unit="model",
                leave=False,
                dynamic_ncols=True,
            )
        for future in completed:
            result = future.result()
            results.append(result)
            if progress:
                print(f"Completed {result['description']}", flush=True)
    return results


def _spawn_workers_available() -> bool:
    """Return whether the current entry point can be imported by spawn workers."""

    import __main__

    main_file = getattr(__main__, "__file__", None)
    return bool(main_file and Path(main_file).exists())


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
    threshold_protocol: str = "fold_internal_best_f1",
) -> dict[str, object]:
    return {
        "dataset": dataset_name,
        "method": method,
        "score": score_name,
        "cv_seed": cv_seed,
        "cv_fold": cv_fold,
        "threshold_protocol": threshold_protocol,
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


def _score_turning_baselines(
    train_nominal: pd.DataFrame,
    evaluation: pd.DataFrame,
    repo_root: Path,
    *,
    image_size: tuple[int, int],
    pca_components: int,
    random_state: int,
) -> dict[str, np.ndarray]:
    """Fit turning baselines on nominal rows and score an evaluation partition."""

    X_train_desc = load_matrix(
        train_nominal, repo_root, image_descriptor, image_size=image_size
    )
    X_eval_desc = load_matrix(evaluation, repo_root, image_descriptor, image_size=image_size)
    scaler = StandardScaler().fit(X_train_desc)
    X_train_desc = scaler.transform(X_train_desc)
    X_eval_desc = scaler.transform(X_eval_desc)

    ocsvm = OneClassSVM(kernel="rbf", gamma="scale", nu=0.05).fit(X_train_desc)
    iforest = IsolationForest(random_state=random_state, contamination="auto").fit(X_train_desc)
    scores = {
        "one_class_svm_image_features": -ocsvm.decision_function(X_eval_desc),
        "isolation_forest_image_features": -iforest.decision_function(X_eval_desc),
    }

    X_train_vec = load_matrix(train_nominal, repo_root, image_vector, image_size=image_size)
    X_eval_vec = load_matrix(evaluation, repo_root, image_vector, image_size=image_size)
    n_components = min(pca_components, X_train_vec.shape[0], X_train_vec.shape[1])
    pca = PCA(n_components=n_components, random_state=random_state, svd_solver="randomized").fit(
        X_train_vec
    )
    eval_recon = pca.inverse_transform(pca.transform(X_eval_vec))
    scores["pca_image_reconstruction"] = np.mean((X_eval_vec - eval_recon) ** 2, axis=1)
    return scores


def _select_thresholds_from_pairwise_scores(
    pairwise_scores: dict[tuple[int, int], pd.DataFrame],
    *,
    outer_fold: int,
    score_columns: Iterable[str],
) -> tuple[dict[str, dict[str, float]], dict[str, int]]:
    """Select thresholds from pairwise inner scores excluding one outer fold.

    A pairwise score frame contains predictions for both folds omitted when its
    model was trained. For a given outer fold, only the other omitted fold is
    used as inner validation data. Thus every score used for threshold
    selection comes from a model trained without the outer evaluation fold.
    """

    inner_validation: list[pd.DataFrame] = []
    inner_scores: dict[str, list[np.ndarray]] = {
        score_name: [] for score_name in score_columns
    }
    n_splits = len({fold for pair in pairwise_scores for fold in pair})

    for inner_fold in range(n_splits):
        if inner_fold == outer_fold:
            continue
        pair = _fold_pair(outer_fold, inner_fold)
        pair_scores = pairwise_scores[pair]
        validation = pair_scores[pair_scores["cv_fold"] == inner_fold].reset_index(drop=True)
        if validation.empty:
            raise RuntimeError(
                f"No cached inner scores for outer fold {outer_fold}, "
                f"inner fold {inner_fold}."
            )
        inner_validation.append(validation)
        for score_name in score_columns:
            inner_scores[score_name].append(validation[score_name].to_numpy())

    validation_rows = pd.concat(inner_validation, ignore_index=True)
    y_inner = validation_rows["target"].to_numpy()
    thresholds = {
        score_name: select_best_f1_threshold(y_inner, np.concatenate(values))
        for score_name, values in inner_scores.items()
    }
    metadata = {
        "n_splits": n_splits - 1,
        "n_samples": int(len(validation_rows)),
        "n_chatter": int((validation_rows["target"] == 1).sum()),
        "n_no_chatter": int((validation_rows["target"] == 0).sum()),
        "n_runs": int(validation_rows["source_run"].nunique()),
    }
    return thresholds, metadata


def _build_pairwise_baseline_scores(
    assignments: pd.DataFrame,
    repo_root: Path,
    *,
    cv_seed: int,
    n_splits: int,
    image_size: tuple[int, int],
    pca_components: int,
) -> dict[tuple[int, int], pd.DataFrame]:
    """Fit each reusable inner baseline model once and score both held-out folds."""

    pairwise_scores: dict[tuple[int, int], pd.DataFrame] = {}
    for first_fold, second_fold in _fold_pairs(n_splits):
        held_out = assignments[
            assignments["cv_fold"].isin([first_fold, second_fold])
        ].reset_index(drop=True)
        train_nominal = assignments[
            (~assignments["cv_fold"].isin([first_fold, second_fold]))
            & (assignments["target"] == 0)
        ].reset_index(drop=True)
        scores = _score_turning_baselines(
            train_nominal,
            held_out,
            repo_root,
            image_size=image_size,
            pca_components=pca_components,
            random_state=cv_seed + first_fold * n_splits + second_fold,
        )
        pair_scores = held_out.copy()
        for method, values in scores.items():
            pair_scores[method] = values
        pairwise_scores[(first_fold, second_fold)] = pair_scores
    return pairwise_scores


def run_turning_baseline_grouped_cv(
    manifest: pd.DataFrame,
    repo_root: Path,
    *,
    n_splits: int = 5,
    seeds: Iterable[int] = (42,),
    inner_splits: int | None = None,
    image_size: tuple[int, int] = (150, 100),
    pca_components: int = 32,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Run nested grouped CV for turning descriptor/PCA baselines.

    The fixed outer fold assignment is also used for inner CV. Each model
    trained on all folds except a pair of folds is fitted once and reused for
    both possible outer-fold threshold selections.
    """

    inner_splits = _resolve_reusable_inner_splits(n_splits, inner_splits)

    metric_rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    threshold_payload: dict[str, object] = {
        "threshold_protocol": NESTED_THRESHOLD_PROTOCOL,
        "inner_cv_folds": inner_splits,
        "folds": [],
    }

    for cv_seed in seeds:
        assignments = make_grouped_cv_assignments(manifest, n_splits=n_splits, seed=cv_seed)
        pairwise_scores = _build_pairwise_baseline_scores(
            assignments,
            repo_root,
            cv_seed=cv_seed,
            n_splits=n_splits,
            image_size=image_size,
            pca_components=pca_components,
        )
        for cv_fold in range(n_splits):
            train_nominal = assignments[
                (assignments["cv_fold"] != cv_fold) & (assignments["target"] == 0)
            ].reset_index(drop=True)
            evaluation = assignments[assignments["cv_fold"] == cv_fold].reset_index(drop=True)
            y_eval = evaluation["target"].to_numpy()

            nested_thresholds, inner_metadata = _select_thresholds_from_pairwise_scores(
                pairwise_scores,
                outer_fold=cv_fold,
                score_columns=[
                    "one_class_svm_image_features",
                    "isolation_forest_image_features",
                    "pca_image_reconstruction",
                ],
            )
            outer_scores = _score_turning_baselines(
                train_nominal,
                evaluation,
                repo_root,
                image_size=image_size,
                pca_components=pca_components,
                random_state=cv_seed + cv_fold,
            )

            fold_thresholds = {}
            for method in (
                "one_class_svm_image_features",
                "isolation_forest_image_features",
            ):
                eval_scores = outer_scores[method]
                selected = nested_thresholds[method]
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
                        threshold_protocol=NESTED_THRESHOLD_PROTOCOL,
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

            method = "pca_image_reconstruction"
            eval_scores = outer_scores[method]
            selected = nested_thresholds[method]
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
                    threshold_protocol=NESTED_THRESHOLD_PROTOCOL,
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
                    "inner_cv": inner_metadata,
                    "thresholds": fold_thresholds,
                }
            )

    metrics_df = pd.DataFrame(metric_rows)
    scores_df = pd.concat(score_frames, ignore_index=True)
    summary_df = summarize_cv_metrics(
        metrics_df, group_columns=["dataset", "method", "score"]
    )
    return metrics_df, scores_df, summary_df, threshold_payload


def _fit_turning_ae_model(
    train_nominal: pd.DataFrame,
    repo_root: Path,
    *,
    bottleneck_dim: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    image_size: tuple[int, int],
    seed: int,
    description: str,
    progress: bool,
    tf: object,
) -> tuple[object, pd.DataFrame, object]:
    """Fit one autoencoder using only nominal rows from the supplied partition."""

    tf.keras.backend.clear_session()
    set_seed(seed)
    train_images, train_rows = load_rgb_images(
        train_nominal, repo_root, image_size=image_size
    )
    if progress:
        print(
            f"Starting {description}: {len(train_images)} nominal images, "
            f"up to {epochs} epochs",
            flush=True,
        )
    x_train, x_stop = train_test_split(
        train_images,
        test_size=0.2,
        random_state=seed,
        shuffle=True,
    )
    model = build_cnn_autoencoder(
        bottleneck_dim=bottleneck_dim,
        learning_rate=learning_rate,
    )
    callbacks = [
        TqdmEpochCallback(
            total_epochs=epochs,
            desc=description,
            disable=not progress,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            min_delta=1e-6,
            restore_best_weights=True,
            verbose=0,
        ),
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
    if progress:
        print(
            f"Completed {description}: {len(history.history['loss'])} epochs, "
            f"best val_loss={np.min(history.history['val_loss']):.6f}",
            flush=True,
        )
    return model, train_rows, history


def _build_pairwise_ae_scores(
    assignments: pd.DataFrame,
    repo_root: Path,
    *,
    cv_seed: int,
    n_splits: int,
    bottleneck_dim: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    image_size: tuple[int, int],
    n_ver_segments: int,
    ver_top_k: int,
    tf: object,
    progress: bool,
) -> dict[tuple[int, int], pd.DataFrame]:
    """Fit reusable pairwise inner AE models and score both held-out folds."""

    pairwise_scores: dict[tuple[int, int], pd.DataFrame] = {}
    pairs = _fold_pairs(n_splits)
    for pair_index, (first_fold, second_fold) in enumerate(pairs, start=1):
        train_nominal = assignments[
            (~assignments["cv_fold"].isin([first_fold, second_fold]))
            & (assignments["target"] == 0)
        ].reset_index(drop=True)
        held_out = assignments[
            assignments["cv_fold"].isin([first_fold, second_fold])
        ].reset_index(drop=True)
        model, _, _ = _fit_turning_ae_model(
            train_nominal,
            repo_root,
            bottleneck_dim=bottleneck_dim,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            image_size=image_size,
            seed=cv_seed + first_fold * n_splits + second_fold,
            description=(
                f"bn{bottleneck_dim} inner {pair_index}/{len(pairs)} "
                f"seed{cv_seed} folds{first_fold}+{second_fold}"
            ),
            progress=progress,
            tf=tf,
        )
        held_out_images, held_out_rows = load_rgb_images(
            held_out, repo_root, image_size=image_size
        )
        scores = score_reconstructions(
            model,
            held_out_images,
            n_ver_segments=n_ver_segments,
            ver_top_k=ver_top_k,
            batch_size=batch_size,
        )
        pair_scores = pd.concat(
            [
                held_out_rows.reset_index(drop=True),
                scores.reset_index(drop=True),
            ],
            axis=1,
        )
        pairwise_scores[(first_fold, second_fold)] = pair_scores
        del model
    return pairwise_scores


def _pairwise_ae_worker(job: dict[str, object]) -> dict[str, object]:
    """Train one pairwise inner AE model in a GPU-isolated worker."""

    import tensorflow as tf

    _configure_worker_gpu(tf, job["gpu_index"])
    assignments = job["assignments"]
    first_fold = job["first_fold"]
    second_fold = job["second_fold"]
    train_nominal = assignments[
        (~assignments["cv_fold"].isin([first_fold, second_fold]))
        & (assignments["target"] == 0)
    ].reset_index(drop=True)
    held_out = assignments[
        assignments["cv_fold"].isin([first_fold, second_fold])
    ].reset_index(drop=True)
    model, _, _ = _fit_turning_ae_model(
        train_nominal,
        job["repo_root"],
        bottleneck_dim=job["bottleneck_dim"],
        epochs=job["epochs"],
        patience=job["patience"],
        batch_size=job["batch_size"],
        learning_rate=job["learning_rate"],
        image_size=job["image_size"],
        seed=job["seed"],
        description=job["description"],
        progress=False,
        tf=tf,
    )
    held_out_images, held_out_rows = load_rgb_images(
        held_out, job["repo_root"], image_size=job["image_size"]
    )
    scores = score_reconstructions(
        model,
        held_out_images,
        n_ver_segments=job["n_ver_segments"],
        ver_top_k=job["ver_top_k"],
        batch_size=job["batch_size"],
    )
    return {
        "pair": (first_fold, second_fold),
        "description": job["description"],
        "scores": pd.concat(
            [held_out_rows.reset_index(drop=True), scores.reset_index(drop=True)],
            axis=1,
        ),
    }


def _build_pairwise_ae_scores_parallel(
    assignments: pd.DataFrame,
    repo_root: Path,
    *,
    cv_seed: int,
    n_splits: int,
    bottleneck_dim: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    image_size: tuple[int, int],
    n_ver_segments: int,
    ver_top_k: int,
    gpu_count: int,
    progress: bool,
) -> dict[tuple[int, int], pd.DataFrame]:
    """Run reusable pairwise inner AE fits concurrently across GPUs."""

    pairs = _fold_pairs(n_splits)
    jobs = [
        {
            "assignments": assignments,
            "repo_root": repo_root,
            "first_fold": first_fold,
            "second_fold": second_fold,
            "bottleneck_dim": bottleneck_dim,
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "image_size": image_size,
            "n_ver_segments": n_ver_segments,
            "ver_top_k": ver_top_k,
            "seed": cv_seed + first_fold * n_splits + second_fold,
            "description": (
                f"bn{bottleneck_dim} inner seed{cv_seed} "
                f"folds{first_fold}+{second_fold}"
            ),
        }
        for first_fold, second_fold in pairs
    ]
    results = _run_parallel_jobs(
        _pairwise_ae_worker,
        jobs,
        gpu_count=gpu_count,
        progress=progress,
        description=f"bn{bottleneck_dim} pairwise inner models",
    )
    return {result["pair"]: result["scores"] for result in results}


def _outer_ae_worker(job: dict[str, object]) -> dict[str, object]:
    """Train and score one final outer-fold AE in a GPU-isolated worker."""

    import tensorflow as tf

    _configure_worker_gpu(tf, job["gpu_index"])
    assignments = job["assignments"]
    cv_fold = job["cv_fold"]
    train_nominal = assignments[
        (assignments["cv_fold"] != cv_fold) & (assignments["target"] == 0)
    ].reset_index(drop=True)
    evaluation = assignments[assignments["cv_fold"] == cv_fold].reset_index(drop=True)
    model, train_rows, history = _fit_turning_ae_model(
        train_nominal,
        job["repo_root"],
        bottleneck_dim=job["bottleneck_dim"],
        epochs=job["epochs"],
        patience=job["patience"],
        batch_size=job["batch_size"],
        learning_rate=job["learning_rate"],
        image_size=job["image_size"],
        seed=job["seed"],
        description=job["description"],
        progress=False,
        tf=tf,
    )
    eval_images, eval_rows = load_rgb_images(
        evaluation, job["repo_root"], image_size=job["image_size"]
    )
    eval_scores = pd.concat(
        [
            eval_rows.reset_index(drop=True),
            score_reconstructions(
                model,
                eval_images,
                n_ver_segments=job["n_ver_segments"],
                ver_top_k=job["ver_top_k"],
                batch_size=job["batch_size"],
            ),
        ],
        axis=1,
    )
    model_path = job.get("model_path")
    if model_path is not None:
        model.save(model_path)
    return {
        "cv_seed": job["cv_seed"],
        "cv_fold": cv_fold,
        "description": job["description"],
        "train_rows": train_rows,
        "eval_rows": eval_rows,
        "eval_scores": eval_scores,
        "history": history.history,
    }


def _run_outer_ae_jobs_parallel(
    assignments_by_seed: dict[int, pd.DataFrame],
    repo_root: Path,
    *,
    seed_list: list[int],
    n_splits: int,
    bottleneck_dim: int,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    image_size: tuple[int, int],
    n_ver_segments: int,
    ver_top_k: int,
    model_dir: Path | None,
    gpu_count: int,
    progress: bool,
) -> list[dict[str, object]]:
    """Run final outer-fold AE fits concurrently across GPUs."""

    jobs = []
    for cv_seed in seed_list:
        for cv_fold in range(n_splits):
            jobs.append(
                {
                    "assignments": assignments_by_seed[cv_seed],
                    "repo_root": repo_root,
                    "cv_seed": cv_seed,
                    "cv_fold": cv_fold,
                    "bottleneck_dim": bottleneck_dim,
                    "epochs": epochs,
                    "patience": patience,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "image_size": image_size,
                    "n_ver_segments": n_ver_segments,
                    "ver_top_k": ver_top_k,
                    "seed": cv_seed + cv_fold,
                    "description": f"bn{bottleneck_dim} outer seed{cv_seed} fold{cv_fold}",
                    "model_path": (
                        model_dir / f"ae_bn{bottleneck_dim}_turning_cv_seed{cv_seed}_fold{cv_fold}.keras"
                        if model_dir is not None
                        else None
                    ),
                }
            )
    return _run_parallel_jobs(
        _outer_ae_worker,
        jobs,
        gpu_count=gpu_count,
        progress=progress,
        description=f"bn{bottleneck_dim} outer models",
    )


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
    progress: bool = False,
    inner_splits: int | None = None,
    parallel_gpus: bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Train and evaluate the CNN autoencoder with reusable nested grouped CV."""

    import tensorflow as tf

    inner_splits = _resolve_reusable_inner_splits(n_splits, inner_splits)

    metric_rows: list[dict[str, object]] = []
    score_frames: list[pd.DataFrame] = []
    threshold_payload: dict[str, object] = {
        "threshold_protocol": NESTED_THRESHOLD_PROTOCOL,
        "inner_cv_folds": inner_splits,
        "folds": [],
    }

    if model_dir is not None:
        model_dir.mkdir(parents=True, exist_ok=True)

    seed_list = list(seeds)
    assignments_by_seed = {
        cv_seed: make_grouped_cv_assignments(manifest, n_splits=n_splits, seed=cv_seed)
        for cv_seed in seed_list
    }
    gpu_count = len(tf.config.list_physical_devices("GPU"))
    use_parallel_gpus = (
        gpu_count > 1 if parallel_gpus is None else parallel_gpus and gpu_count > 1
    )
    if use_parallel_gpus and not _spawn_workers_available():
        use_parallel_gpus = False
        if progress:
            print(
                "Parallel GPU training is unavailable from this interactive entry "
                "point; using sequential training.",
                flush=True,
            )
    if parallel_gpus and gpu_count < 2 and progress:
        print(
            "Parallel GPU training requested, but fewer than two GPUs are visible; "
            "using sequential training.",
            flush=True,
        )
    pairwise_scores_by_seed = {}
    for cv_seed in seed_list:
        if use_parallel_gpus:
            pairwise_scores_by_seed[cv_seed] = _build_pairwise_ae_scores_parallel(
                assignments_by_seed[cv_seed],
                repo_root,
                cv_seed=cv_seed,
                n_splits=n_splits,
                bottleneck_dim=bottleneck_dim,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                learning_rate=learning_rate,
                image_size=image_size,
                n_ver_segments=n_ver_segments,
                ver_top_k=ver_top_k,
                gpu_count=gpu_count,
                progress=progress,
            )
        else:
            pairwise_scores_by_seed[cv_seed] = _build_pairwise_ae_scores(
                assignments_by_seed[cv_seed],
                repo_root,
                cv_seed=cv_seed,
                n_splits=n_splits,
                bottleneck_dim=bottleneck_dim,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                learning_rate=learning_rate,
                image_size=image_size,
                n_ver_segments=n_ver_segments,
                ver_top_k=ver_top_k,
                tf=tf,
                progress=progress,
            )

    outer_results_by_fold: dict[tuple[int, int], dict[str, object]] = {}
    if use_parallel_gpus:
        outer_results = _run_outer_ae_jobs_parallel(
            assignments_by_seed,
            repo_root,
            seed_list=seed_list,
            n_splits=n_splits,
            bottleneck_dim=bottleneck_dim,
            epochs=epochs,
            patience=patience,
            batch_size=batch_size,
            learning_rate=learning_rate,
            image_size=image_size,
            n_ver_segments=n_ver_segments,
            ver_top_k=ver_top_k,
            model_dir=model_dir,
            gpu_count=gpu_count,
            progress=progress,
        )
        outer_results_by_fold = {
            (result["cv_seed"], result["cv_fold"]): result
            for result in outer_results
        }

    fold_jobs = [
        (cv_seed, cv_fold)
        for cv_seed in seed_list
        for cv_fold in range(n_splits)
    ]
    if progress:
        from tqdm.auto import tqdm

        fold_jobs_iter = tqdm(
            fold_jobs,
            desc=f"bn{bottleneck_dim} grouped-CV folds",
            unit="fold",
            leave=False,
            dynamic_ncols=True,
        )
    else:
        fold_jobs_iter = fold_jobs

    for cv_seed, cv_fold in fold_jobs_iter:
        assignments = assignments_by_seed[cv_seed]
        if progress:
            fold_jobs_iter.set_postfix({"seed": cv_seed, "fold": cv_fold})

        train_nominal = assignments[
            (assignments["cv_fold"] != cv_fold) & (assignments["target"] == 0)
        ].reset_index(drop=True)
        evaluation = assignments[assignments["cv_fold"] == cv_fold].reset_index(drop=True)

        nested_thresholds, inner_metadata = _select_thresholds_from_pairwise_scores(
            pairwise_scores_by_seed[cv_seed],
            outer_fold=cv_fold,
            score_columns=AE_SCORE_COLUMNS,
        )
        if use_parallel_gpus:
            outer_result = outer_results_by_fold[(cv_seed, cv_fold)]
            train_rows = outer_result["train_rows"]
            eval_rows = outer_result["eval_rows"]
            eval_scores = outer_result["eval_scores"]
            history_values = outer_result["history"]
        else:
            model, train_rows, history = _fit_turning_ae_model(
                train_nominal,
                repo_root,
                bottleneck_dim=bottleneck_dim,
                epochs=epochs,
                patience=patience,
                batch_size=batch_size,
                learning_rate=learning_rate,
                image_size=image_size,
                seed=cv_seed + cv_fold,
                description=f"bn{bottleneck_dim} seed{cv_seed} fold{cv_fold}",
                progress=progress,
                tf=tf,
            )

            eval_images, eval_rows = load_rgb_images(evaluation, repo_root, image_size=image_size)
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
            if model_dir is not None:
                model.save(model_dir / f"ae_bn{bottleneck_dim}_turning_cv_seed{cv_seed}_fold{cv_fold}.keras")
            history_values = history.history

        y_eval = eval_rows["target"].to_numpy()

        fold_thresholds = {}
        for score_name in AE_SCORE_COLUMNS:
            values = eval_scores[score_name].to_numpy()
            selected = nested_thresholds[score_name]
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
                threshold_protocol=NESTED_THRESHOLD_PROTOCOL,
            )
            row.update(
                {
                    "bottleneck_dim": bottleneck_dim,
                    "epochs_trained": int(len(history_values["loss"])),
                    "best_val_loss": float(np.min(history_values["val_loss"])),
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
                "inner_cv": inner_metadata,
                "thresholds": fold_thresholds,
            }
        )

    if progress and hasattr(fold_jobs_iter, "close"):
        fold_jobs_iter.close()

    metrics_df = pd.DataFrame(metric_rows)
    scores_df = pd.concat(score_frames, ignore_index=True)
    summary_df = summarize_cv_metrics(
        metrics_df, group_columns=["dataset", "method", "score", "bottleneck_dim"]
    )
    return metrics_df, scores_df, summary_df, threshold_payload
