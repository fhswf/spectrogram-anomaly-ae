from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from spectrogram_anomaly_ae.cnc_machining import (
    CNCChannelNormalization,
    assign_cnc_record_splits,
    estimate_cnc_channel_normalization,
    export_cnc_record_windows,
    iter_cnc_records,
    load_cnc_vibration,
    parse_cnc_record,
)


class CNCMachiningTests(unittest.TestCase):
    def test_parse_load_and_export_windows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "data" / "M01" / "OP00" / "good" / "M01_Aug_2019_OP00_000.h5"
            h5_path.parent.mkdir(parents=True)
            values = np.arange(18_000, dtype=np.float32).reshape(6_000, 3)
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("vibration_data", data=values)

            records = list(iter_cnc_records(root))
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].sample_id, "M01_Aug_2019_OP00_000")
            np.testing.assert_array_equal(load_cnc_vibration(records[0]), values)

            out_root = root / "npz"
            rows = export_cnc_record_windows(
                records[0],
                out_root,
                split="train",
                window_seconds=2.5,
                label_scheme="anomaly",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["label"], "nominal")
            self.assertEqual(rows[0]["source_label"], "good")

            npz = np.load(rows[0]["npz_path"])
            self.assertEqual(npz["X"].shape, (5_000,))
            self.assertEqual(float(npz["fs"]), 2_000.0)
            self.assertEqual(str(npz["source_dataset"]), "cnc_machining")
            self.assertEqual(int(npz["target_window_samples"]), 5_000)
            self.assertEqual(int(npz["window_samples"]), 5_000)
            self.assertEqual(str(npz["channel_normalization_method"]), "none")

    def test_export_windows_can_apply_channel_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "data" / "M01" / "OP00" / "good" / "M01_Aug_2019_OP00_000.h5"
            h5_path.parent.mkdir(parents=True)
            values = np.array(
                [
                    [2.0, 10.0, -4.0],
                    [4.0, 14.0, 2.0],
                    [6.0, 18.0, 8.0],
                    [8.0, 22.0, 14.0],
                ],
                dtype=np.float32,
            )
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("vibration_data", data=values)

            normalization = CNCChannelNormalization(
                mean=np.array([2.0, 10.0, -4.0]),
                std=np.array([2.0, 4.0, 6.0]),
                sample_count=4,
            )
            record = next(iter_cnc_records(root))
            rows = export_cnc_record_windows(
                record,
                root / "npz",
                split="train",
                window_seconds=0.002,
                label_scheme="anomaly",
                channel_normalization=normalization,
            )

            npz = np.load(rows[0]["npz_path"])
            np.testing.assert_allclose(npz["X"], [0.0, 1.0, 2.0, 3.0])
            np.testing.assert_allclose(npz["Y"], [0.0, 1.0, 2.0, 3.0])
            np.testing.assert_allclose(npz["Z"], [0.0, 1.0, 2.0, 3.0])
            np.testing.assert_allclose(npz["channel_normalization_mean"], normalization.mean)
            np.testing.assert_allclose(npz["channel_normalization_std"], normalization.std)
            self.assertEqual(str(npz["channel_normalization_method"]), "zscore_train_nominal")
            self.assertEqual(rows[0]["channel_normalization_method"], "zscore_train_nominal")

    def test_estimate_channel_normalization_uses_only_nominal_train_records(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths_and_values = [
                (
                    root / "data" / "M01" / "OP00" / "good" / "M01_Aug_2019_OP00_000.h5",
                    np.array([[1.0, 2.0, 3.0], [3.0, 6.0, 9.0]], dtype=np.float32),
                    "train",
                ),
                (
                    root / "data" / "M01" / "OP00" / "good" / "M01_Aug_2019_OP00_001.h5",
                    np.full((2, 3), 1000.0, dtype=np.float32),
                    "validation",
                ),
                (
                    root / "data" / "M01" / "OP00" / "bad" / "M01_Aug_2019_OP00_002.h5",
                    np.full((2, 3), -1000.0, dtype=np.float32),
                    "test",
                ),
            ]
            records = []
            split_by_path = {}
            for path, values, split in paths_and_values:
                path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("vibration_data", data=values)
                record = parse_cnc_record(path)
                records.append(record)
                split_by_path[record.path] = split

            normalization = estimate_cnc_channel_normalization(records, split_by_path)

            np.testing.assert_allclose(normalization.mean, [2.0, 4.0, 6.0])
            np.testing.assert_allclose(normalization.std, [1.0, 2.0, 3.0])
            self.assertEqual(normalization.sample_count, 2)

    def test_short_records_are_exported_as_one_full_file_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            h5_path = root / "data" / "M01" / "OP00" / "good" / "M01_Aug_2019_OP00_000.h5"
            h5_path.parent.mkdir(parents=True)
            values = np.arange(18_000, dtype=np.float32).reshape(6_000, 3)
            with h5py.File(h5_path, "w") as handle:
                handle.create_dataset("vibration_data", data=values)

            record = next(iter_cnc_records(root))
            rows = export_cnc_record_windows(
                record,
                root / "npz",
                split="train",
                window_samples=400_000,
                label_scheme="anomaly",
            )

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["target_window_samples"], 400_000)
            self.assertEqual(rows[0]["window_samples"], 6_000)

            npz = np.load(rows[0]["npz_path"])
            self.assertEqual(npz["X"].shape, (6_000,))
            self.assertEqual(int(npz["target_window_samples"]), 400_000)
            self.assertEqual(int(npz["window_samples"]), 6_000)

    def test_bad_records_are_not_assigned_to_train_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            paths = [
                root / "data" / "M01" / "OP00" / "good" / "M01_Aug_2019_OP00_000.h5",
                root / "data" / "M01" / "OP00" / "bad" / "M01_Aug_2019_OP00_001.h5",
            ]
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(path, "w") as handle:
                    handle.create_dataset("vibration_data", data=np.zeros((5_000, 3), dtype=np.float32))

            records = [parse_cnc_record(path) for path in paths]
            splits = assign_cnc_record_splits(records)
            bad_record = next(record for record in records if record.health_label == "bad")
            self.assertIn(splits[bad_record.path], {"validation", "test"})


if __name__ == "__main__":
    unittest.main()
