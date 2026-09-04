# Notebook-First Implementation Plan for TODO.md

This plan turns the reviewer-facing open points in `TODO.md` into a notebook-first experimental workflow. Notebooks are the primary implementation and reporting artifacts; pure Python files should stay minimal and only hold small shared utilities when duplicated notebook code becomes hard to maintain.

## Current Repository State

- Public turning data is present as 651 windowed `.npz` samples under `data/01_windowed_labeled_2,5s`.
- Generated spectrograms are present under `data/02_spectrograms_150x100px_dataset` with:
  - 472 `train/no_chatter`
  - 118 `validation/no_chatter`
  - 61 `validation/chatter`
- The notebook series now uses consistent two-digit numbering from `01_Load_Data_Segmentation_Labeling.ipynb` through `11_Method_Documentation_and_Citation_Cleanup.ipynb`.
- Spectrogram generation is standardized on `data/02_spectrograms_150x100px_dataset`.
- No broaching dataset files are visible in the repository; broaching metrics require adding or referencing that dataset separately.

## Target Deliverables

- A logically numbered notebook series where each notebook consumes artifacts from earlier notebooks.
- Frozen train/validation/test manifests before spectrogram generation and evaluation.
- Baseline anomaly detectors evaluated on the same frozen splits as the CNN autoencoder.
- VER ablation and segmentation sensitivity study.
- Numerical PR-AUC, F1, precision, recall, and confusion-matrix tables.
- Bootstrap confidence intervals for the reported metrics.
- Architecture, STFT/image-generation, filtering, annotation, deployment, error-analysis, and citation-cleanup documentation.
- Dependency metadata managed only through `pyproject.toml` and `uv.lock`; remove `requirements.txt`.

## Notebook Workflow

Use two-digit notebook prefixes. Existing notebooks have been renamed into the sequence where possible.

1. `01_Load_Data_Segmentation_Labeling.ipynb`
   - Create labeled `.npz` windows from raw turning data.
   - Keep the existing data download, segmentation, and labeling logic here.
   - Output `data/01_windowed_labeled_2,5s`.

2. `02_Create_Frozen_Splits_and_Manifests.ipynb`
   - Create deterministic train/validation/test manifests before downstream artifact generation.
   - Include one row per sample with `sample_id`, `source_dataset`, `process`, `stickout`, `rpm`, `doc`, `window`, `label`, `npz_path`, `image_path`, and `split`.
   - Keep all 472 current train nominal samples for nominal AE training unless a later contamination/filtering experiment intentionally changes the training set.
   - Split the current 118 nominal validation images and 61 chatter validation images into threshold-validation and final-test subsets by source run where possible.
   - Store `reports/manifests/turning_split_seed42.csv` and `reports/manifests/turning_split_summary.csv`.

3. `03_Create_Spectrogram_Datasets.ipynb`
   - Generate spectrogram images from the frozen manifests, including the canonical `150x100` dataset.
   - Normalize the generated data path so notebook constants and checked-in data agree.
   - Document STFT and image settings: window function, `nperseg`, overlap, FFT/PSD mode, frequency cutoff, dB scaling, resizing, interpolation, axis stacking, flipping, and normalization.

4. `04_Train_CNN_AE_BN16_150x100px.ipynb`
   - Train the main CNN autoencoder on nominal training samples only.
   - Save trained models under `models/`.
   - Generate an architecture table from the Keras model with layer type, kernel size, stride/pool size, padding, activation, output shape, latent dimension, and parameter count.

5. `05_Evaluate_AE_Scores_and_Thresholds.ipynb`
   - Compute global MSE, global MAE, VER maximum segment score, and VER top-k segment aggregation for the trained AE.
   - Select operating thresholds only on the validation split.
   - Freeze thresholds in `reports/thresholds/*.json`.
   - Evaluate final metrics only on the held-out test split using frozen thresholds.

6. `06_Baseline_Comparisons.ipynb`
   - Evaluate PCA reconstruction error, one-class SVM, isolation forest, and optional 1D time-series AE baselines.
   - Use the same frozen train/validation/test split and threshold discipline as the CNN AE.
   - Use handcrafted features such as per-axis RMS, peak absolute amplitude, crest factor, spectral centroid, spectral bandwidth, band energies, dominant frequency, and top peak amplitudes below 5 kHz.

7. `07_VER_Ablation_and_Sensitivity.ipynb`
   - Compare global MSE, global MAE, VER maximum segment score, and VER top-k segment aggregation.
   - Vary number of vertical segments, for example 5, 10, 15, and 20.
   - Vary segment width or overlap, for example non-overlap and 50 percent overlap.
   - Keep model, data split, and thresholding protocol fixed.

8. `08_Bootstrap_CIs_and_Report_Tables.ipynb`
   - Produce PR-AUC, F1-score, precision, recall, and confusion matrices for turning and broaching when broaching data is available.
   - Bootstrap test rows with replacement, use at least 1,000 samples, and report 2.5 and 97.5 percentile intervals.
   - Save paper-ready CSV/Markdown tables and PR/confusion-matrix figures under `reports/`.

9. `09_Resolution_Contamination_Axis_Studies.ipynb`
   - Compare `150x100` with at least one lower and one higher image resolution, for example `75x50` and `300x200`.
   - Test training-set contamination by injecting small fractions of anomalous samples into nominal training data, for example 1, 2, 5, and 10 percent.
   - Compare X-only, Y-only, Z-only, and combined RGB X/Y/Z spectrograms.

10. `10_Error_Analysis_and_Deployment_Benchmark.ipynb`
    - Analyze false positives and false negatives with source metadata, scores, thresholds, and reconstruction/error heatmaps.
    - Measure inference time per stroke and approximate memory footprint on the intended deployment hardware if available.
    - Report hardware, software versions, batch size, preprocessing time, and model inference time separately.

11. `11_Method_Documentation_and_Citation_Cleanup.ipynb`
    - Generate final method documentation tables for architecture, STFT/image-generation parameters, filtering, and annotation protocol.
    - Quantify how many samples were removed during reconstruction-driven filtering and document the review criterion.
    - Document annotator expertise, annotation criteria, use of force/vibration signals, ambiguous-case handling, and inter-annotator agreement status.
    - Audit autoencoder citations:
      - Keep `Autoencoders` temporarily for general AE background, but replace it with the real book-chapter citation if metadata can be identified.
      - Remove `AEIntroduction` from active use.
      - Remove `BottleneckRepresentation` unless the text specifically discusses the redundancy penalty method.
      - Keep `BottleneckInvestigation` only for bottleneck-size discussion.
      - Keep `Hyper-Parameter` only if the final training procedure was guided by Smith's method.

12. `12_Publication_Quality_Figures_and_Tables.ipynb`
    - Create paper-ready figures and tables from generated report artifacts.
    - Save vector figures as PDF/SVG and high-resolution PNG copies under `reports/publication/figures`.
    - Save publication tables as CSV and LaTeX under `reports/publication/tables`.

## Minimal Python Policy

- Keep `src/spectrogram_anomaly_ae/py.typed`.
- Do not introduce broad script or package-module extraction by default.
- If notebook duplication becomes difficult to maintain, add one compact helper module such as `src/spectrogram_anomaly_ae/notebook_utils.py`.
- Avoid creating separate `paths.py`, `splits.py`, `datasets.py`, `models.py`, `scoring.py`, `metrics.py`, or `reports.py` unless the notebooks become unmaintainable.
- Do not create a `scripts/` workflow unless notebook execution proves insufficient.

## Dependency Cleanup

- Move all packages from `requirements.txt` into `pyproject.toml`.
- Use the repo's lower-bound dependency style:
  - `scikit-learn>=1.9.0`
  - `torch>=2.9.1`
  - `pillow>=12.2.0`
  - `matplotlib>=3.10.9`
  - `tqdm>=4.67.3`
  - `plotly>=6.8.0`
  - `nbformat>=5.10.4`
  - `nptdms>=1.10.0`
- Remove `requirements.txt`.
- Run `uv lock` after editing dependency metadata.
- Resolve any Python/PyTorch compatibility mismatch if `uv lock` reports that PyTorch does not support the current `requires-python` range.

## Acceptance Criteria

- `IMPLEMENTATION_PLAN.md` lists the notebooks in the natural logical order above, including publication-quality output generation.
- Split creation precedes spectrogram generation and all downstream evaluation notebooks consume frozen manifests.
- The plan no longer recommends broad script/module extraction.
- Pure Python additions are limited to minimal shared notebook helpers if needed.
- Dependencies are declared in `pyproject.toml`, `requirements.txt` is removed, and `uv.lock` is refreshed.
- No final-test labels are used to choose thresholds or tune model variants.
- Every table requested in `TODO.md` is assigned to a notebook and output location.
- Broaching results are either included from a documented data location or explicitly blocked pending dataset availability.
