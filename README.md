# spectrogram-anomaly-ae
Code accompanying the paper on unsupervised anomaly detection in vibration signals using convolutional autoencoders and spectrogram representations.

## Datasets
This repository applies the proposed methodology to public CNC vibration datasets.

### Turning Chatter Dataset
The primary experiment uses the public turning/chatter diagnosis dataset hosted on Mendeley Data:

> Khasawneh, F., Otto, A., & Yesilli, M. (2019). *Turning Dataset for Chatter Diagnosis Using Machine Learning* (Version 1) [Data set]. Mendeley Data. 
> https://doi.org/10.17632/hvm4wh3jzx.1

The dataset is distributed as `.mat` files containing the time vector (`t`) and multi-sensor vibration signals (`d`). In this repository, we extract the accelerometer channels, convert windowed time-series into spectrogram representations (RGB images), and use these inputs to run unsupervised anomaly detection experiments with convolutional autoencoders.

### Bosch CNC Machining Dataset
An additional supported dataset is the Bosch Research CNC Machining benchmark:

> Tnani, M.-A., Feil, M., & Diepold, K. (2022). *Smart Data Collection System for Brownfield CNC Milling Machines: A New Benchmark Dataset for Data-Driven Machine Monitoring*. Procedia CIRP, 107, 131-136. https://doi.org/10.1016/j.procir.2022.04.022

The Bosch dataset stores tri-axial acceleration samples in HDF5 files under `data/Mxx/OPxx/{good,bad}`. Use `notebooks/01b_Prepare_CNC_Machining_Dataset.ipynb` for lazy download/preparation, or run the preparation script directly:

```bash
git clone https://github.com/boschresearch/CNC_Machining.git data/raw_cnc_machining
.venv/bin/python scripts/prepare_cnc_machining_dataset.py \
  --source-root data/raw_cnc_machining \
  --output-root data/01_windowed_labeled_cnc_machining \
  --manifest-path reports/manifests/cnc_machining_split_seed42.csv \
  --summary-path reports/manifests/cnc_machining_split_summary.csv
```

By default, `good` samples become `nominal`, `bad` samples become `anomaly`, and anomalous records are excluded from the training split. Pass `--label-scheme turning_compat` if you want `good`/`bad` mapped to the existing `no_chatter`/`chatter` labels.

The Bosch dataset does not contain chatter/no-chatter labels. Its labels are process-health annotations (`good` / `bad`), which this repository treats as nominal/anomalous for cross-dataset anomaly-detection comparisons.

## Notebook Workflow

The experiment workflow is organized as a numbered notebook series:

| Notebook | Purpose |
|---|---|
| `notebooks/01_Load_Data_Segmentation_Labeling.ipynb` | Download/extract the unprocessed Mendeley turning dataset, low-pass filter/subsample tool-post accelerometer channels, and create labeled fixed-duration vibration windows. |
| `notebooks/01b_Prepare_CNC_Machining_Dataset.ipynb` | Download/prepare the Bosch CNC Machining dataset and create deterministic CNC split manifests. |
| `notebooks/02_Create_Frozen_Splits_and_Manifests.ipynb` | Create deterministic train/validation/test manifests. |
| `notebooks/03_Create_Spectrogram_Datasets.ipynb` | Generate spectrogram image datasets from the frozen manifest. |
| `notebooks/04_Train_CNN_AE_BN16_150x100px.ipynb` | Train the main CNN autoencoder on nominal training samples. |
| `notebooks/05_Evaluate_AE_Scores_and_Thresholds.ipynb` | Score AE reconstructions and freeze validation-selected thresholds. |
| `notebooks/06_Baseline_Comparisons.ipynb` | Evaluate classical anomaly-detection baselines on the same split. |
| `notebooks/07_VER_Ablation_and_Sensitivity.ipynb` | Run VER ablation and segmentation sensitivity experiments. |
| `notebooks/08_Bootstrap_CIs_and_Report_Tables.ipynb` | Generate metrics, confidence intervals, figures, and report tables. |
| `notebooks/09_Resolution_Contamination_Axis_Studies.ipynb` | Run resolution, contamination, and vibration-axis studies. |
| `notebooks/10_Error_Analysis_and_Deployment_Benchmark.ipynb` | Analyze errors and benchmark inference. |
| `notebooks/11_Method_Documentation_and_Citation_Cleanup.ipynb` | Generate method documentation tables and citation cleanup notes. |
| `notebooks/12_Publication_Quality_Figures_and_Tables.ipynb` | Create paper-ready PDF/SVG/PNG figures and CSV/LaTeX tables. |

The first notebook downloads the unprocessed Mendeley turning dataset, extracts the raw `.mat` files, low-pass filters the tool-post accelerometer channels with an order-100 Butterworth filter, subsamples them to 10 kHz, and creates labeled fixed-duration vibration segments. The Bosch CNC dataset is prepared with `notebooks/01b_Prepare_CNC_Machining_Dataset.ipynb` or `scripts/prepare_cnc_machining_dataset.py`, which uses the same target sample count by default and emits one full-file window when a CNC source file is shorter than the target.

Notebook 03 generates spectrogram datasets for both configured datasets and writes the generated `image_path` values back into each manifest. Notebook 06 writes baseline metrics for each dataset plus combined all-dataset tables. Notebook 05 writes CNN autoencoder metrics for each dataset whose trained model is present, and notebook 08 combines baseline and CNN autoencoder confidence-interval tables for cross-dataset comparison.

The turning notebook writes extracted unprocessed raw data to `data/raw_turning_unprocessed` and saves processed segments to `data/01_windowed_labeled_2,5s`. The CNC preparation notebook clones source data to `data/raw_cnc_machining` and writes processed segments to `data/01_windowed_labeled_cnc_machining`.

File Format (.npz)

Each file contains a single vibration segment:

| Field | Description |
|---|---|
| t | time vector |
| X, Y, Z | accelerometer signals (3-axis vibration) |
| label | class label (`chatter` / `no_chatter` for turning; `nominal` / `anomaly` by default for CNC) |
| A_time | RMS vibration amplitude |
| is_chatter | boolean decision from spectral analysis |
| source_dataset | dataset identifier when generated by the CNC preparation script |
| source_label | original `good` / `bad` CNC label when generated by the CNC preparation script |
| fs | sample rate in Hz when available |
| fs_raw | original sample rate before filtering/subsampling, when generated by the turning notebook |
| lowpass_filter_order | Butterworth filter order, when generated by the turning notebook |
| lowpass_cutoff_hz | Butterworth low-pass cutoff frequency, when generated by the turning notebook |
| subsample_factor | integer subsampling factor, when generated by the turning notebook |
| source_d_columns | source signal-matrix columns used for `X`, `Y`, and `Z`, when generated by the turning notebook |
| target_window_samples | requested sample count for segmentation |
| window_samples | actual sample count in the saved segment |
| window_seconds | actual segment duration in seconds |
