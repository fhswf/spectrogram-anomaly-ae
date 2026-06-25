# Changelog

## Unreleased

### Changed

- Updated `notebooks/01_Load_Data_Segmentation_Labeling.ipynb` to lazily download the unprocessed Mendeley turning dataset from the raw-data file URL and extract it to `data/raw_turning_unprocessed`.
- Restricted turning preprocessing to `timeSeries_*/*.mat` files so tagged two-column summary files are ignored.
- Updated turning preprocessing to use the tool-post tri-axial accelerometer channels from paper columns 6-8, represented as zero-based `d` columns 4-6.
- Added order-100 Butterworth low-pass filtering followed by integer subsampling to 10 kHz before windowing.
- Changed turning windows to fixed 2.5 s duration after subsampling, producing 25,000-sample windows at 10 kHz.
- Added `fs`, `fs_raw`, `lowpass_filter_order`, `lowpass_cutoff_hz`, `subsample_factor`, and `source_d_columns` metadata to generated turning `.npz` files.
- Updated the turning notebook process diagram to show lazy download, channel selection, filtering, subsampling, labeling, and NPZ output.
- Updated `notebooks/03_Create_Spectrogram_Datasets.ipynb` to calculate STFT segment length from a configured frequency/time pixel ratio instead of resizing every spectrogram to a fixed target image size.
- Removed final image resizing from spectrogram generation so image geometry follows the STFT configuration.

### Added

- Added `notebooks/01b_Prepare_CNC_Machining_Dataset.ipynb` as a lazy download/preparation workflow for the Bosch Research CNC Machining dataset.
- Documented the CNC preparation notebook and updated data-preparation paths in `README.md`.

### Fixed

- Fixed missing `fs` metadata in newly generated turning window files.
- Fixed the previous accidental use of boring-rod/audio columns as `X/Y/Z` in turning preprocessing.
- Added graceful skipping for empty or incompatible turning `.mat` files during preprocessing.
