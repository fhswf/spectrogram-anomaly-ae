# Open Points Requiring Author Experiments

## Required before submission

- Add baseline comparisons against simpler anomaly detection approaches, e.g. global reconstruction MSE/MAE, PCA or one-class SVM on spectral features, isolation forest on handcrafted features, and optionally a 1D time-series autoencoder.

- Add an ablation study for the Vertical Error Rate (VER). Report at least global MSE, global MAE, VER maximum segment score, and a top-k segment aggregation variant using the same train/validation/test split.

- Report numerical PR-AUC values for the broaching test set and the public turning dataset. Include F1-score, precision, recall, and confusion matrices at the selected operating point.

- Separate threshold selection from final testing. The threshold should be selected only on validation data and then frozen before evaluation on the independent test set.

- For the public turning dataset, create a separate validation/test split if the sample count permits. If not, clearly state that the experiment is a proof-of-concept rather than a fully independent benchmark.

- Quantify the sensitivity of VER to the number and width of vertical segments. Report whether the selected segmentation is robust across reasonable alternatives.

## Strongly recommended

- Provide confidence intervals or bootstrap intervals for PR-AUC, F1-score, precision, and recall, especially because anomalous samples are rare.

- Report the full autoencoder architecture in tabular form: layer type, kernel size, stride, padding, activation, output shape, latent dimension, and parameter count.

- Document all STFT and image-generation parameters: window function, window length, overlap, FFT size, retained frequency range, amplitude scaling, resizing/interpolation, and normalization strategy.

- Quantify how many samples were removed during reconstruction-driven filtering of the training set and explain the review criterion used to confirm anomalies.

- Evaluate whether the 150 x 100 image resolution materially affects performance by comparing it with at least one lower and one higher resolution.

- Test robustness to possible training-set contamination by injecting small fractions of anomalous samples into the nominal training set.

## Helpful for a stronger revision

- Add an annotation protocol: annotator expertise, annotation criteria, whether force signals were used, how ambiguous cases were handled, and whether inter-annotator agreement was assessed.

- Compare performance when using individual vibration axes versus the combined three-axis representation.

- Measure inference time per stroke and memory footprint on the intended deployment hardware.

- Analyze false positives and false negatives qualitatively to identify which anomaly types or process phases are most difficult.

- If possible, evaluate cross-machine or cross-process transfer more strictly by training on one setup/process and testing on another without retraining.

## Autoencoder Reference Cleanup

| Key | Current role | Recommendation |
| --- | --- | --- |
| `Autoencoders` | General AE background and figure source | Keep for now, but preferably replace/update with the actual book-chapter citation if the book metadata can be identified. The arXiv page itself labels it as a book chapter, not a journal paper. |
| `AEIntroduction` | General AE background | Remove from active use. It is lecture notes / an introductory paper and redundant with `Autoencoders`. |
| `BottleneckRepresentation` | Cited when introducing the CNN-AE architecture | Remove from active use unless the paper explicitly discusses redundancy reduction in the bottleneck representation. It is about a specific redundancy-penalization method that this paper does not use. |
| `BottleneckInvestigation` | Supports bottleneck-size/hyperparameter discussion | Keep if the paper discusses bottleneck size as an experimental design choice. It is directly relevant to convolutional autoencoder bottlenecks, but appears to remain arXiv-only. |
| `Hyper-Parameter` | Supports learning rate/batch size/training tuning | Optional. It is not autoencoder-specific, but it is a relevant technical report for hyperparameter tuning. Keep only if the hyperparameter search was actually guided by Smith's procedure. |

Recommended active citation pattern:

```latex
... scarce and costly to obtain. \cite{Autoencoders}

...
The presented architecture ... bottleneck size, learning rate, batch size, and training duration. \cite{BottleneckInvestigation,Hyper-Parameter}
```