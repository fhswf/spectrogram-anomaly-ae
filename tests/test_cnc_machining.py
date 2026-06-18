from pathlib import Path
import tempfile
import unittest

import h5py
import numpy as np

from spectrogram_anomaly_ae.cnc_machining import (
    assign_cnc_record_splits,
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
