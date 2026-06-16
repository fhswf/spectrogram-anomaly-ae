# spectrogram-anomaly-ae
Code accompanying the paper on unsupervised anomaly detection in vibration signals using convolutional autoencoders and spectrogram representations.

## Dataset
This repository applies the proposed methodology to the public turning/chatter diagnosis dataset hosted on Mendeley Data:

Khasawneh, Firas; Otto, Andreas; Yesilli, Melih (2019),
*Turning Dataset for Chatter Diagnosis Using Machine Learning*,
Mendeley Data, V1.
DOI: 10.17632/hvm4wh3jzx.1
URL: https://data.mendeley.com/datasets/hvm4wh3jzx/1

The dataset is distributed as `.mat` files containing the time vector (`t`) and multi-sensor vibration signals (`d`). In this repository, we extract the accelerometer channels, convert windowed time-series into spectrogram representations (RGB images), and use these inputs to run unsupervised anomaly detection experiments with convolutional autoencoders.

## Data Preparation: ./data/01_windowed_labeled
This directory contains windowed vibration time series segments extracted from raw machining experiments. Each segment is labeled and serves as the intermediate dataset for spectrogram generation and CNN-based anomaly detection.

File Format (.npz)

Each file contains a single 5-second vibration segment:

t → time vector
X, Y, Z → accelerometer signals (3-axis vibration)
label → binary class (chatter / no_chatter)
A_time → RMS vibration amplitude
is_chatter → boolean decision from spectral analysis


