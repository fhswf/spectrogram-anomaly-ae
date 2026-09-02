from __future__ import annotations

import numpy as np
import pandas as pd
from PIL import Image

import spectrogram_anomaly_ae.turning_cv as turning_cv
from spectrogram_anomaly_ae.turning_cv import (
    fold_balance,
    make_grouped_cv_assignments,
    run_turning_baseline_grouped_cv,
    score_reconstructions,
    summarize_cv_metrics,
)


def synthetic_manifest() -> pd.DataFrame:
    rows = []
    for run_idx in range(6):
        for window in range(3):
            rows.append(
                {
                    "source_dataset": "turning",
                    "source_run": f"nominal_run_{run_idx}",
                    "sample_id": f"n_{run_idx}_{window}",
                    "split": "train",
                    "label": "no_chatter",
                    "image_path": f"dummy/n_{run_idx}_{window}.png",
                }
            )
    for run_idx in range(6):
        for window in range(2):
            rows.append(
                {
                    "source_dataset": "turning",
                    "source_run": f"mixed_run_{run_idx}",
                    "sample_id": f"m_{run_idx}_{window}",
                    "split": "test",
                    "label": "chatter" if window == 0 else "no_chatter",
                    "image_path": f"dummy/m_{run_idx}_{window}.png",
                }
            )
    return pd.DataFrame(rows)


def test_grouped_cv_keeps_source_runs_together() -> None:
    assignments = make_grouped_cv_assignments(synthetic_manifest(), n_splits=3, seed=7)

    folds_per_run = assignments.groupby("source_run")["cv_fold"].nunique()
    assert folds_per_run.max() == 1
    assert set(assignments["cv_fold"]) == {0, 1, 2}


def test_fold_balance_counts_classes_and_runs() -> None:
    assignments = make_grouped_cv_assignments(synthetic_manifest(), n_splits=3, seed=7)
    balance = fold_balance(assignments)

    assert set(balance.columns) == {
        "cv_fold",
        "n_samples",
        "n_runs",
        "n_chatter",
        "n_no_chatter",
        "n_chatter_runs",
    }
    assert balance["n_chatter"].sum() == 6
    assert balance["n_no_chatter"].sum() == 24


def test_summarize_cv_metrics_adds_ci_columns() -> None:
    metrics = pd.DataFrame(
        [
            {"method": "a", "score": "s", "f1": 0.5, "pr_auc": 0.6},
            {"method": "a", "score": "s", "f1": 0.7, "pr_auc": 0.8},
        ]
    )

    summary = summarize_cv_metrics(metrics, group_columns=["method", "score"])

    assert summary.loc[0, "f1_count"] == 2
    assert summary.loc[0, "f1_mean"] == 0.6
    assert "f1_ci95" in summary.columns


def test_turning_baseline_grouped_cv_uses_inner_thresholds(tmp_path, monkeypatch) -> None:
    manifest = synthetic_manifest()
    for row_idx, row in manifest.reset_index(drop=True).iterrows():
        image_path = tmp_path / row["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        base = np.linspace(0, 255, num=12 * 10, dtype=np.uint8).reshape(10, 12)
        image = (base + row_idx * 7) % 255
        Image.fromarray(image.astype(np.uint8)).save(image_path)

    threshold_selection_lengths = []
    original_selector = turning_cv.select_best_f1_threshold

    def recording_selector(y_true, scores):
        threshold_selection_lengths.append(len(y_true))
        return original_selector(y_true, scores)

    monkeypatch.setattr(turning_cv, "select_best_f1_threshold", recording_selector)

    metrics, scores, summary, thresholds = run_turning_baseline_grouped_cv(
        manifest,
        tmp_path,
        n_splits=3,
        seeds=[7],
        image_size=(12, 10),
        pca_components=2,
    )

    assert set(metrics["method"]) == {
        "one_class_svm_image_features",
        "isolation_forest_image_features",
        "pca_image_reconstruction",
    }
    assert set(metrics["cv_fold"]) == {0, 1, 2}
    assert len(metrics) == 9
    assert set(scores["method"]) == set(metrics["method"])
    assert not summary.empty
    assert thresholds["threshold_protocol"] == "nested_grouped_inner_best_f1"
    assert thresholds["inner_cv_folds"] == 3
    assert len(thresholds["folds"]) == 3
    assignments = make_grouped_cv_assignments(manifest, n_splits=3, seed=7)
    expected_lengths = [
        len(assignments[assignments["cv_fold"] != cv_fold])
        for cv_fold in range(3)
        for _ in range(3)
    ]
    assert threshold_selection_lengths == expected_lengths
    assert all("inner_cv" in fold for fold in thresholds["folds"])


def test_turning_ae_nested_threshold_helper_uses_inner_scores(tmp_path, monkeypatch) -> None:
    manifest = synthetic_manifest()
    for row_idx, row in manifest.reset_index(drop=True).iterrows():
        image_path = tmp_path / row["image_path"]
        image_path.parent.mkdir(parents=True, exist_ok=True)
        image = np.full((10, 12), row_idx, dtype=np.uint8)
        Image.fromarray(image).save(image_path)

    assignments = make_grouped_cv_assignments(manifest, n_splits=3, seed=7)
    outer_training = assignments[assignments["cv_fold"] != 0].reset_index(drop=True)
    threshold_selection_lengths = []

    def fake_fit(*_args, **_kwargs):
        return object(), pd.DataFrame(), object()

    def fake_score(_model, images, **_kwargs):
        values = np.arange(len(images), dtype="float32")
        return pd.DataFrame({score_name: values for score_name in turning_cv.AE_SCORE_COLUMNS})

    original_selector = turning_cv.select_best_f1_threshold

    def recording_selector(y_true, scores):
        threshold_selection_lengths.append(len(y_true))
        return original_selector(y_true, scores)

    monkeypatch.setattr(turning_cv, "_fit_turning_ae_model", fake_fit)
    monkeypatch.setattr(turning_cv, "score_reconstructions", fake_score)
    monkeypatch.setattr(turning_cv, "select_best_f1_threshold", recording_selector)

    _, metadata = turning_cv._select_nested_ae_thresholds(
        outer_training,
        tmp_path,
        outer_seed=7,
        outer_fold=0,
        inner_splits=3,
        bottleneck_dim=8,
        epochs=1,
        patience=1,
        batch_size=2,
        learning_rate=1e-3,
        image_size=(12, 10),
        n_ver_segments=3,
        ver_top_k=2,
        tf=object(),
    )

    assert threshold_selection_lengths == [len(outer_training)] * 4
    assert metadata["n_splits"] == 3
    assert metadata["n_samples"] == len(outer_training)


class FakeInferenceModel:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def __call__(self, batch: np.ndarray, *, training: bool) -> np.ndarray:
        assert training is False
        self.batch_sizes.append(len(batch))
        return np.zeros_like(batch)

    def predict(self, *_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("score_reconstructions should use direct inference, not predict")


def test_score_reconstructions_uses_direct_batched_inference() -> None:
    model = FakeInferenceModel()
    images = np.ones((5, 4, 6, 3), dtype="float32")

    scores = score_reconstructions(
        model,
        images,
        n_ver_segments=3,
        ver_top_k=2,
        batch_size=2,
    )

    assert model.batch_sizes == [2, 2, 1]
    assert list(scores.columns) == ["global_mse", "global_mae", "ver_max", "ver_topk"]
    assert np.allclose(scores["global_mse"], 1.0)
