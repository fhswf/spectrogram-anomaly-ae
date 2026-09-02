from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from spectrogram_anomaly_ae.turning_cv import (
    NESTED_THRESHOLD_PROTOCOL,
    fold_balance,
    make_grouped_cv_assignments,
    run_turning_baseline_grouped_cv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate turning classical baselines with grouped k-fold CV."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("reports/manifests/turning_split_seed42.csv"),
        help="Frozen manifest containing turning source_run, label, and image_path columns.",
    )
    parser.add_argument("--cv-folds", type=int, default=5)
    parser.add_argument(
        "--inner-cv-folds",
        type=int,
        default=None,
        help=(
            "Inner grouped folds used for threshold selection. Must equal "
            "--cv-folds - 1 so pairwise inner models can be reused."
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Legacy single-seed option.")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Repeat grouped CV for each seed. Defaults to --seed.",
    )
    parser.add_argument("--image-width", type=int, default=150)
    parser.add_argument("--image-height", type=int, default=100)
    parser.add_argument("--pca-components", type=int, default=32)
    parser.add_argument("--table-dir", type=Path, default=Path("reports/tables"))
    parser.add_argument("--baseline-dir", type=Path, default=Path("reports/baselines"))
    parser.add_argument("--threshold-dir", type=Path, default=Path("reports/thresholds"))
    parser.add_argument("--cv-dir", type=Path, default=Path("reports/cv"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = args.seeds if args.seeds is not None else [args.seed]
    repo_root = Path.cwd()
    image_size = (args.image_width, args.image_height)

    metrics_path = args.table_dir / "metrics_turning_baselines_grouped_cv.csv"
    summary_path = args.table_dir / "metrics_turning_baselines_grouped_cv_summary.csv"
    scores_path = args.baseline_dir / "baseline_scores_turning_grouped_cv.csv"
    threshold_path = args.threshold_dir / "baseline_thresholds_turning_grouped_cv.json"
    balance_path = args.cv_dir / "turning_baseline_grouped_cv_fold_balance.csv"

    output_paths = [metrics_path, summary_path, scores_path, threshold_path, balance_path]
    existing_thresholds = None
    if threshold_path.exists():
        with threshold_path.open("r", encoding="utf-8") as file:
            existing_thresholds = json.load(file)
    if (
        all(path.exists() for path in output_paths)
        and existing_thresholds is not None
        and existing_thresholds.get("threshold_protocol") == NESTED_THRESHOLD_PROTOCOL
        and not args.overwrite
    ):
        print("Skipping turning baseline grouped CV: existing outputs found.", flush=True)
        for path in output_paths:
            print(f"  {path}", flush=True)
        return

    for directory in [args.table_dir, args.baseline_dir, args.threshold_dir, args.cv_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    balance_frames = []
    for cv_seed in seeds:
        assignments = make_grouped_cv_assignments(
            manifest,
            n_splits=args.cv_folds,
            seed=cv_seed,
        )
        balance = fold_balance(assignments)
        balance.insert(0, "cv_seed", cv_seed)
        balance_frames.append(balance)

    metrics, scores, summary, thresholds = run_turning_baseline_grouped_cv(
        manifest,
        repo_root,
        n_splits=args.cv_folds,
        seeds=seeds,
        inner_splits=args.inner_cv_folds,
        image_size=image_size,
        pca_components=args.pca_components,
    )
    thresholds.update(
        {
            "cv_folds": args.cv_folds,
            "inner_cv_folds": (
                args.inner_cv_folds
                if args.inner_cv_folds is not None
                else args.cv_folds - 1
            ),
            "seeds": seeds,
            "pca_components": args.pca_components,
            "image_size": {"width": args.image_width, "height": args.image_height},
        }
    )

    pd.concat(balance_frames, ignore_index=True).to_csv(balance_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    summary.to_csv(summary_path, index=False)
    scores.to_csv(scores_path, index=False)
    write_json(threshold_path, thresholds)

    print(f"Wrote {metrics_path}", flush=True)
    print(f"Wrote {summary_path}", flush=True)
    print(f"Wrote {scores_path}", flush=True)
    print(f"Wrote {threshold_path}", flush=True)
    print(f"Wrote {balance_path}", flush=True)


if __name__ == "__main__":
    main()
