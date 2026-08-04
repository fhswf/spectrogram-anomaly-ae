from __future__ import annotations

import numpy as np
import pandas as pd

from spectrogram_anomaly_ae.turning_cv import (
    fold_balance,
    make_grouped_cv_assignments,
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
