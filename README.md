# spectrogram-anomaly-ae
Code accompanying the paper on unsupervised anomaly detection in vibration signals using convolutional autoencoders and spectrogram representations.

## Dataset
This repository applies the proposed methodology to the public turning/chatter diagnosis dataset hosted on Mendeley Data:

> Khasawneh, F., Otto, A., & Yesilli, M. (2019). *Turning Dataset for Chatter Diagnosis Using Machine Learning* (Version 1) [Data set]. Mendeley Data. 
> https://doi.org/10.17632/hvm4wh3jzx.1

The dataset is distributed as `.mat` files containing the time vector (`t`) and multi-sensor vibration signals (`d`). In this repository, we extract the accelerometer channels, convert windowed time-series into spectrogram representations (RGB images), and use these inputs to run unsupervised anomaly detection experiments with convolutional autoencoders.

## Notebook: notebooks/01-1_Load_Data_Segmentation_Labeling.ipynb
This notebook downloads the original Mendeley dataset, extracts the raw `.mat` files, and creates labeled 2.5-second windowed vibration segments.

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


