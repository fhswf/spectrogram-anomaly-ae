# spectrogram-anomaly-ae
Code accompanying the paper on unsupervised anomaly detection in vibration signals using convolutional autoencoders and spectrogram representations.

## Dataset
This repository applies the proposed methodology to the public turning/chatter diagnosis dataset hosted on Mendeley Data:

> Khasawneh, F., Otto, A., & Yesilli, M. (2019). *Turning Dataset for Chatter Diagnosis Using Machine Learning* (Version 1) [Data set]. Mendeley Data. 
> https://doi.org/10.17632/hvm4wh3jzx.1

The dataset is distributed as `.mat` files containing the time vector (`t`) and multi-sensor vibration signals (`d`). In this repository, we extract the accelerometer channels, convert windowed time-series into spectrogram representations (RGB images), and use these inputs to run unsupervised anomaly detection experiments with convolutional autoencoders.

## Notebook Workflow

The experiment workflow is organized as a numbered notebook series:

| Notebook | Purpose |
|---|---|
| `notebooks/01_Load_Data_Segmentation_Labeling.ipynb` | Download/extract the original Mendeley dataset and create labeled 2.5-second vibration windows. |
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

The first notebook downloads the original Mendeley dataset, extracts the raw `.mat` files, and creates labeled 2.5-second windowed vibration segments.

The notebook writes extracted raw data to `data/raw_mat` and saves processed segments to `data/01_windowed_labeled_2,5s`.

File Format (.npz)

Each file contains a single 2.5-second vibration segment:

| Field | Description |
|---|---|
| t | time vector |
| X, Y, Z | accelerometer signals (3-axis vibration) |
| label | binary class (chatter / no_chatter) |
| A_time | RMS vibration amplitude |
| is_chatter | boolean decision from spectral analysis |
