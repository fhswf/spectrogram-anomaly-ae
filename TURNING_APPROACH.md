# Turning Dataset Evaluation Approach

The turning dataset is small, imbalanced, and organized as short windows from a limited number of machining runs. This makes a single frozen validation/test split useful for reproducibility, but noisy as a performance estimator. In the current split, the held-out test set contains only 27 chatter and 48 no-chatter samples. Therefore, one additional false negative changes test recall by about 3.7 percentage points, and a few borderline samples can visibly move F1-score, precision, and recall.

The bottleneck sweep already shows this instability from a second angle: repeated training with different random seeds gives noticeably different results. Those seed repeats measure model stochasticity from initialization, mini-batch order, and the internal nominal train/validation split used for early stopping. They do not measure how much the reported result depends on which machining runs happened to be placed in the validation or test set, because the external validation/test manifest remains fixed.

For the turning dataset, k-fold cross validation is therefore appropriate as a robustness analysis. The goal is not to create more independent data, but to use the available labeled windows more efficiently and to report how performance varies across different held-out subsets. The cross-validation results should be interpreted as a distribution of plausible performance estimates, not as a replacement for genuinely independent data.

## Grouped Folds

Folds should be grouped by `source_run`, not by individual image/window. Adjacent windows from the same machining run are correlated because they share process parameters, tooling state, sensor setup, and temporal context. If windows from the same run appear in both training and evaluation folds, sample-level k-fold cross validation can overestimate generalization and produce confidence intervals that look more stable than the experiment really is.

The recommended split strategy is therefore stratified grouped k-fold cross validation:

- preserve all windows from the same `source_run` in the same fold;
- balance the chatter/no-chatter class distribution as much as possible across folds;
- train anomaly detectors only on no-chatter samples from the training folds;
- score the held-out fold containing both no-chatter and chatter samples;
- repeat the folds across several random seeds when computationally feasible.

Because chatter occurs in a small number of runs, five folds is a reasonable starting point, but three folds can be more stable if a five-fold split produces poorly balanced positive folds. The fold summaries should include the number of chatter samples and chatter-bearing runs per fold.

## Autoencoder Protocol

For each grouped fold, the convolutional autoencoder is trained only on no-chatter images from the training folds. A small internal split of those nominal training images is still used for early stopping. The held-out fold is scored with reconstruction-error metrics such as global MSE, global MAE, vertical maximum error, and vertical top-k error.

Threshold-free metrics, especially PR-AUC, are the cleanest cross-validation summary because they do not require choosing an operating threshold from scarce labeled data. Thresholded metrics such as F1, precision, and recall are still useful, but the threshold protocol must be stated clearly.

The implemented grouped-CV notebook path reports fold-internal best-F1 thresholds. This is useful for comparing score functions and model variants under scarce data, but it is optimistic as an estimate of deployed thresholded performance because the threshold is selected on the same fold being summarized. For final deployment-style numbers, keep the existing discipline: select thresholds on the frozen validation split and evaluate once on the frozen held-out test split.

## Baseline Protocol

The classical baseline models should use the same grouped folds as the autoencoder. In each fold, the descriptor-based one-class SVM, isolation forest, and PCA reconstruction baseline are fitted only on no-chatter samples from the training folds and scored on the held-out fold. This keeps the comparison fair: all methods see the same nominal training runs and are evaluated on the same held-out runs.

Reporting should include both per-fold metrics and aggregated mean, standard deviation, and approximate 95% confidence intervals across fold/seed repeats. The confidence intervals should be described as cross-validation variability intervals rather than formal population confidence intervals, because folds are correlated and the dataset is small.

## Bottleneck Sweep Protocol

The bottleneck-dimension sweep should be evaluated under the same grouped-CV protocol when the goal is to compare latent dimensionalities robustly. In this mode, every bottleneck dimension is trained on the same fold/seed combinations, and performance deltas should be computed within matched `cv_seed` and `cv_fold` pairs. This paired comparison is preferable to comparing independent averages because each bottleneck dimension is evaluated on the same held-out machining runs.

The fixed validation/test bottleneck sweep remains useful as a direct companion to the final frozen-split evaluation. The grouped-CV sweep is better suited for answering whether a bottleneck choice is consistently strong across plausible held-out run partitions.

## Recommended Paper Wording

A concise description for an accompanying paper could be:

> In addition to the frozen validation/test evaluation, we performed repeated stratified grouped k-fold cross validation on the turning dataset. Groups were defined by machining run to avoid leakage between adjacent windows from the same experiment. In each fold, models were fitted only on nominal samples from the training groups and evaluated on the held-out groups containing both nominal and chatter windows. We report fold-level metrics and aggregate variability across folds and random seeds to quantify sensitivity to the small and imbalanced evaluation set.

This wording keeps the role of cross validation honest: it supports stability analysis and model comparison, while the frozen validation/test split remains the final threshold-selection and held-out-test protocol.
