# Head and backend CPU coverage

101 tests passed without skips on Python 3.12.6, Torch 2.6.0.

All tests use synthetic features, labels, durations, prototypes and checkpoints. Existing MIL regressions are included. No clinical records were read.

| Component | CPU checks |
|---|---|
| Neural `linear` | Forward, backward, probabilities, artifact reload |
| Neural `mlp` | Forward, backward, probabilities, artifact reload |
| Neural `bayesian_mlp` | Forward, backward, probabilities, artifact reload |
| Classifier `logistic_regression` | Real binary/multiclass fit, class order, probabilities, artifact reload |
| Classifier `random_forest` | Real binary/multiclass fit, class order, probabilities, artifact reload |
| Classifier `extra_trees` | Real binary/multiclass fit, class order, probabilities, artifact reload |
| Classifier `hist_gradient_boosting` | Real binary/multiclass fit, class order, probabilities, artifact reload |
| Classifier `svm` | Real binary/multiclass fit, class order, probabilities, artifact reload |
| Classifier `knn` | Real binary/multiclass fit, class order, probabilities, artifact reload |
| Classifier `gaussian_nb` | Real binary/multiclass fit, class order, probabilities, artifact reload |
| Classifier `sklearn_mlp` | Real binary/multiclass fit, class order, probabilities, artifact reload |
| Classifier `xgboost` | Real binary/multiclass fit, class order, probabilities, artifact reload |
| Classifier `lightgbm` | Real binary/multiclass fit, class order, probabilities, artifact reload |
| Classifier `catboost` | Real binary/multiclass fit, class order, probabilities, artifact reload |
| Survival `cox` | Linear/MLP gradients, censoring/tie oracles, calibrated artifact reload |
| Survival `discrete_hazard` | Linear/MLP gradients, censoring/tie oracles, calibrated artifact reload |

| MIL profile | Representation / gradient / freeze / reload | Cox + discrete head gradients |
|---|---|---|
| `abmil` | Passed | Passed |
| `gated_abmil` | Passed | Passed |
| `dsmil` | Passed | Passed |
| `dtfd_mil` | Passed | Passed |
| `acmil` | Passed | Passed |
| `asmil` | Passed | Passed |
| `clam_sb` | Passed | Passed |
| `clam_mb` | Passed | Passed |
| `varpool` | Passed | Passed |
| `panther` | Passed | Passed |
| `wikg` | Passed | Passed |
| `tad_graph` | Passed | Passed |
| `resgat` | Passed | Passed |
| `patch_gcn` | Passed | Passed |
| `slidegraph_plus` | Passed | Passed |
| `cs_mil` | Passed | Passed |
| `zoommil` | Passed | Passed |
| `hag_mil` | Passed | Passed |
| `gfr_mil` | Passed | Passed |
| `h2_mil` | Passed | Passed |
| `hipt` | Passed | Passed |
| `transmil` | Passed | Passed |
| `ilra` | Passed | Passed |
| `rrt_mil` | Passed | Passed |
| `dgr_mil` | Passed | Passed |
| `caprmil` | Passed | Passed |
| `lin_mil` | Passed | Passed |
| `mamba_mil` | Passed | Passed |
| `nnmil` | Passed | Passed |
| `mhim_v1` | Passed | Passed |
| `mhim_v2` | Passed | Passed |
| `mean_mil` | Passed | Passed |
| `max_mil` | Passed | Passed |
| `mean_std_mil` | Passed | Passed |

This verifies CPU execution and mathematical regression cases. Real-cohort accuracy, clinical validity, GPU execution and every optional upstream configuration remain untested.
