"""Utilities for the Bosch Research CNC Machining dataset."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path
import re

import h5py
import numpy as np

CNC_MACHINING_SAMPLE_RATE_HZ = 2_000
CNC_MACHINING_DATASET_KEY = "vibration_data"
CNC_MACHINING_REFERENCE_WINDOW_SAMPLES = 400_000

_FILENAME_RE = re.compile(
    r"^(?P<machine>M\d{2})_(?P<month>[A-Za-z]{3})_(?P<year>\d{4})_"
    r"(?P<operation>OP\d{2})_(?P<example_id>\d{3})\.h5$"
)

_LABEL_SCHEMES = {
    "anomaly": {"good": "nominal", "bad": "anomaly"},
    "health": {"good": "good", "bad": "bad"},
    "turning_compat": {"good": "no_chatter", "bad": "chatter"},
}


@dataclass(frozen=True)
class CNCMachiningRecord:
    """Metadata for one CNC Machining HDF5 sample."""

    path: Path
    machine: str
    operation: str
    health_label: str
    timeframe: str
    example_id: str

    @property
    def is_anomaly(self) -> bool:
        return self.health_label == "bad"

    @property
    def sample_id(self) -> str:
        return f"{self.machine}_{self.timeframe}_{self.operation}_{self.example_id}"


@dataclass(frozen=True)
class CNCChannelNormalization:
    """Per-channel z-score parameters for CNC vibration windows."""

    mean: np.ndarray
    std: np.ndarray
    sample_count: int
    method: str = "zscore_train_nominal"

    def __post_init__(self) -> None:
        mean = np.asarray(self.mean, dtype=np.float64)
        std = np.asarray(self.std, dtype=np.float64)
        if mean.shape != (3,) or std.shape != (3,):
            raise ValueError(
                "CNC channel normalization mean and std must have shape (3,), "
                f"got {mean.shape} and {std.shape}"
            )
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(std)):
            raise ValueError("CNC channel normalization values must be finite")
        if np.any(std <= 0):
            raise ValueError("CNC channel normalization std values must be positive")
        if self.sample_count <= 0:
            raise ValueError("CNC channel normalization sample_count must be positive")

        object.__setattr__(self, "mean", mean)
        object.__setattr__(self, "std", std)

    def apply(self, data: np.ndarray) -> np.ndarray:
        """Return data normalized channel-wise with these training statistics."""

        data = np.asarray(data, dtype=np.float32)
        if data.ndim != 2 or data.shape[1] != 3:
            raise ValueError(
                "Expected CNC vibration data with shape (n_samples, 3), got "
                f"{data.shape}"
            )
        return ((data - self.mean) / self.std).astype(np.float32)


def resolve_cnc_data_root(path: str | Path) -> Path:
    """Return the dataset's ``data`` directory from a repo root or data root."""

    root = Path(path).expanduser().resolve()
    if (root / "data").is_dir():
        root = root / "data"
    if not root.is_dir():
        raise FileNotFoundError(f"CNC data root does not exist: {root}")
    return root


def parse_cnc_record(path: str | Path) -> CNCMachiningRecord:
    """Parse a CNC Machining HDF5 path into structured metadata."""

    path = Path(path).expanduser().resolve()
    match = _FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected CNC Machining filename: {path.name}")

    label = path.parent.name
    if label not in {"good", "bad"}:
        raise ValueError(f"Unexpected CNC Machining health label: {label}")

    operation = path.parent.parent.name
    machine = path.parent.parent.parent.name
    groups = match.groupdict()
    if machine != groups["machine"] or operation != groups["operation"]:
        raise ValueError(
            "CNC Machining path metadata does not match filename metadata: "
            f"{path}"
        )

    return CNCMachiningRecord(
        path=path,
        machine=machine,
        operation=operation,
        health_label=label,
        timeframe=f"{groups['month']}_{groups['year']}",
        example_id=groups["example_id"],
    )


def iter_cnc_records(
    data_root: str | Path,
    *,
    labels: Iterable[str] = ("good", "bad"),
) -> Iterator[CNCMachiningRecord]:
    """Yield CNC Machining records under ``data_root`` in stable path order."""

    root = resolve_cnc_data_root(data_root)
    allowed_labels = set(labels)
    unknown_labels = allowed_labels - {"good", "bad"}
    if unknown_labels:
        raise ValueError(f"Unknown CNC Machining labels: {sorted(unknown_labels)}")

    for path in sorted(root.glob("M[0-9][0-9]/OP[0-9][0-9]/*/*.h5")):
        if path.parent.name in allowed_labels:
            yield parse_cnc_record(path)


def load_cnc_vibration(
    record_or_path: CNCMachiningRecord | str | Path,
    *,
    dataset_key: str = CNC_MACHINING_DATASET_KEY,
) -> np.ndarray:
    """Load one tri-axial vibration array with shape ``(n_samples, 3)``."""

    path = record_or_path.path if isinstance(record_or_path, CNCMachiningRecord) else record_or_path
    with h5py.File(path, "r") as handle:
        if dataset_key not in handle:
            raise KeyError(f"{path} does not contain HDF5 dataset {dataset_key!r}")
        vibration_data = np.asarray(handle[dataset_key])

    if vibration_data.ndim != 2 or vibration_data.shape[1] != 3:
        raise ValueError(
            "Expected CNC vibration data with shape (n_samples, 3), got "
            f"{vibration_data.shape}"
        )
    return vibration_data


def map_cnc_label(health_label: str, *, label_scheme: str = "anomaly") -> str:
    """Map ``good``/``bad`` into the label scheme used by a downstream workflow."""

    try:
        return _LABEL_SCHEMES[label_scheme][health_label]
    except KeyError as exc:
        schemes = ", ".join(sorted(_LABEL_SCHEMES))
        raise ValueError(f"Unknown label scheme {label_scheme!r}; choose one of: {schemes}") from exc


def assign_cnc_record_splits(
    records: Iterable[CNCMachiningRecord],
    *,
    seed: int = 42,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
) -> dict[Path, str]:
    """Assign file-level train/validation/test splits.

    Good records are split across train, validation, and test. Bad records are
    split across validation and test so the default unsupervised training split
    remains nominal-only.
    """

    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("train_fraction + validation_fraction must be less than 1")

    by_label: dict[str, list[CNCMachiningRecord]] = {"good": [], "bad": []}
    for record in records:
        by_label[record.health_label].append(record)

    rng = np.random.default_rng(seed)
    splits: dict[Path, str] = {}
    for label, label_records in by_label.items():
        ordered = sorted(label_records, key=lambda record: str(record.path))
        indices = np.arange(len(ordered))
        rng.shuffle(indices)

        if label == "good":
            train_end = round(len(indices) * train_fraction)
            validation_end = round(len(indices) * (train_fraction + validation_fraction))
            boundaries = {
                "train": indices[:train_end],
                "validation": indices[train_end:validation_end],
                "test": indices[validation_end:],
            }
        else:
            validation_end = round(len(indices) * 0.5)
            boundaries = {
                "validation": indices[:validation_end],
                "test": indices[validation_end:],
            }

        for split, split_indices in boundaries.items():
            for index in split_indices:
                splits[ordered[int(index)].path] = split

    return splits


def estimate_cnc_channel_normalization(
    records: Iterable[CNCMachiningRecord],
    split_by_path: dict[Path, str],
    *,
    train_split: str = "train",
    nominal_health_label: str = "good",
) -> CNCChannelNormalization:
    """Estimate per-channel z-score stats from nominal training records only."""

    count = 0
    mean = np.zeros(3, dtype=np.float64)
    m2 = np.zeros(3, dtype=np.float64)

    for record in records:
        if record.health_label != nominal_health_label:
            continue
        if split_by_path.get(record.path) != train_split:
            continue

        data = load_cnc_vibration(record).astype(np.float64, copy=False)
        if len(data) == 0:
            continue

        batch_count = len(data)
        batch_mean = np.mean(data, axis=0)
        centered = data - batch_mean
        batch_m2 = np.sum(centered * centered, axis=0)

        total_count = count + batch_count
        delta = batch_mean - mean
        mean = mean + delta * (batch_count / total_count)
        m2 = m2 + batch_m2 + delta * delta * count * batch_count / total_count
        count = total_count

    if count == 0:
        raise ValueError(
            "No nominal training CNC samples were available for channel normalization"
        )

    std = np.sqrt(m2 / count)
    std = np.where(std > 0, std, 1.0)
    return CNCChannelNormalization(mean=mean, std=std, sample_count=count)


def export_cnc_record_windows(
    record: CNCMachiningRecord,
    output_root: str | Path,
    *,
    split: str,
    sample_rate_hz: int = CNC_MACHINING_SAMPLE_RATE_HZ,
    window_samples: int = CNC_MACHINING_REFERENCE_WINDOW_SAMPLES,
    window_seconds: float | None = None,
    overlap: float = 0.0,
    label_scheme: str = "anomaly",
    channel_normalization: CNCChannelNormalization | None = None,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    """Export one HDF5 record into windowed ``.npz`` files.

    Returns manifest rows matching the generated files.
    """

    if window_seconds is not None and window_seconds <= 0:
        raise ValueError("window_seconds must be positive")
    if window_samples <= 0:
        raise ValueError("window_samples must be positive")
    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in the interval [0, 1)")

    data = load_cnc_vibration(record)
    export_data = (
        channel_normalization.apply(data)
        if channel_normalization is not None
        else data
    )
    window_size = (
        int(round(window_seconds * sample_rate_hz))
        if window_seconds is not None
        else int(window_samples)
    )
    step_size = int(round(window_size * (1 - overlap)))
    if window_size <= 0 or step_size <= 0:
        raise ValueError("window size and overlap produced an invalid window step")

    output_root = Path(output_root)
    output_dir = output_root / record.machine / record.operation / record.health_label
    output_dir.mkdir(parents=True, exist_ok=True)

    label = map_cnc_label(record.health_label, label_scheme=label_scheme)
    window_starts = list(range(0, len(data) - window_size + 1, step_size))
    if not window_starts and len(data) > 0:
        window_starts = [0]

    rows: list[dict[str, object]] = []
    for window, start in enumerate(window_starts):
        stop = min(start + window_size, len(data))
        raw_segment = data[start:stop]
        segment = export_data[start:stop]
        actual_window_size = len(segment)
        out_path = output_dir / f"{record.sample_id}_window_{window:04d}.npz"
        if out_path.exists() and not overwrite:
            raise FileExistsError(f"Refusing to overwrite existing file: {out_path}")

        time_vector = np.arange(actual_window_size, dtype=np.float64) / sample_rate_hz
        actual_window_duration_seconds = actual_window_size / sample_rate_hz
        np.savez_compressed(
            out_path,
            t=time_vector,
            X=segment[:, 0],
            Y=segment[:, 1],
            Z=segment[:, 2],
            label=label,
            source_label=record.health_label,
            is_anomaly=record.is_anomaly,
            A_time=float(np.sqrt(np.mean(segment**2))),
            A_time_raw=float(np.sqrt(np.mean(raw_segment**2))),
            fs=float(sample_rate_hz),
            target_window_samples=window_size,
            window_samples=actual_window_size,
            window_seconds=actual_window_duration_seconds,
            source_dataset="cnc_machining",
            machine=record.machine,
            operation=record.operation,
            timeframe=record.timeframe,
            source_file=str(record.path),
            window=window,
            start_sample=start,
            stop_sample=stop,
            channel_normalization_method=(
                channel_normalization.method
                if channel_normalization is not None
                else "none"
            ),
            channel_normalization_mean=(
                channel_normalization.mean
                if channel_normalization is not None
                else np.full(3, np.nan)
            ),
            channel_normalization_std=(
                channel_normalization.std
                if channel_normalization is not None
                else np.full(3, np.nan)
            ),
            channel_normalization_sample_count=(
                channel_normalization.sample_count
                if channel_normalization is not None
                else 0
            ),
        )

        rows.append(
            {
                "source_dataset": "cnc_machining",
                "machine": record.machine,
                "operation": record.operation,
                "timeframe": record.timeframe,
                "source_run": record.sample_id,
                "window": window,
                "sample_id": f"{record.sample_id}_window_{window:04d}",
                "label": label,
                "source_label": record.health_label,
                "is_anomaly": record.is_anomaly,
                "split": split,
                "sample_rate_hz": sample_rate_hz,
                "target_window_samples": window_size,
                "window_samples": actual_window_size,
                "window_seconds": actual_window_duration_seconds,
                "npz_path": str(out_path),
                "source_file": str(record.path),
                "channel_normalization_method": (
                    channel_normalization.method
                    if channel_normalization is not None
                    else "none"
                ),
            }
        )

    return rows
