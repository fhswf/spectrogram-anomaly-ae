#!/usr/bin/env python3
"""Prepare the Bosch Research CNC Machining dataset for this project."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from spectrogram_anomaly_ae.cnc_machining import (
    CNC_MACHINING_REFERENCE_WINDOW_SAMPLES,
    CNC_MACHINING_SAMPLE_RATE_HZ,
    assign_cnc_record_splits,
    export_cnc_record_windows,
    iter_cnc_records,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert boschresearch/CNC_Machining HDF5 files into windowed NPZ "
            "segments and write a deterministic manifest."
        )
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Path to the cloned CNC_Machining repository or its data directory.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/01_windowed_labeled_cnc_machining"),
        help="Directory for generated windowed NPZ files.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=Path("reports/manifests/cnc_machining_split_seed42.csv"),
        help="CSV path for the generated sample manifest.",
    )
    parser.add_argument(
        "--summary-path",
        type=Path,
        default=Path("reports/manifests/cnc_machining_split_summary.csv"),
        help="CSV path for split/label counts.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--window-samples",
        type=int,
        default=CNC_MACHINING_REFERENCE_WINDOW_SAMPLES,
        help=(
            "Target samples per window. Defaults to the turning dataset window size; "
            "shorter source files are exported as one full-file window."
        ),
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=None,
        help="Optional duration-based window size. If provided, overrides --window-samples.",
    )
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument(
        "--label-scheme",
        choices=("anomaly", "health", "turning_compat"),
        default="anomaly",
        help=(
            "Use anomaly labels (nominal/anomaly), source health labels "
            "(good/bad), or labels compatible with the current turning notebooks "
            "(no_chatter/chatter)."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing NPZ files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    records = list(iter_cnc_records(args.source_root))
    split_by_path = assign_cnc_record_splits(records, seed=args.seed)

    rows: list[dict[str, object]] = []
    for record in records:
        rows.extend(
            export_cnc_record_windows(
                record,
                args.output_root,
                split=split_by_path[record.path],
                sample_rate_hz=CNC_MACHINING_SAMPLE_RATE_HZ,
                window_samples=args.window_samples,
                window_seconds=args.window_seconds,
                overlap=args.overlap,
                label_scheme=args.label_scheme,
                overwrite=args.overwrite,
            )
        )

    manifest = pd.DataFrame(rows)
    if not manifest.empty:
        manifest["npz_path"] = manifest["npz_path"].map(
            lambda path: str(Path(path).resolve().relative_to(Path.cwd().resolve()))
            if Path(path).resolve().is_relative_to(Path.cwd().resolve())
            else path
        )
        manifest["source_file"] = manifest["source_file"].astype(str)
    else:
        print(
            "No CNC windows were generated. "
            "Check that the source root contains non-empty CNC vibration files."
        )

    args.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.manifest_path, index=False)

    summary = (
        manifest.groupby(["source_dataset", "split", "label"], dropna=False)
        .size()
        .rename("n")
        .reset_index()
        .sort_values(["source_dataset", "split", "label"])
        if not manifest.empty
        else pd.DataFrame(columns=["source_dataset", "split", "label", "n"])
    )
    args.summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_path, index=False)

    print(f"Wrote {len(manifest)} windows from {len(records)} CNC files")
    print(f"Manifest: {args.manifest_path}")
    print(f"Summary: {args.summary_path}")


if __name__ == "__main__":
    main()
