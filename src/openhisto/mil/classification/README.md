# Classification

Use `create_neural_head` for differentiable linear/MLP/Bayesian heads and `create_classifier` for fitted frozen-vector classifiers.

```python
from openhisto.mil.classification import create_neural_head, create_classifier, list_heads

head = create_neural_head("mlp", input_dim=768, num_classes=3)
forest = create_classifier("random_forest", n_estimators=200)
```

Neural profiles: `linear`, `mlp`, `bayesian_mlp`. Frozen-vector profiles: logistic regression, random forest, Extra Trees, histogram boosting, SVM, kNN, GaussianNB, sklearn MLP, XGBoost, LightGBM and CatBoost. The default random seed is 0; estimators stay on CPU. Classifiers preserve original class labels and probability-column order, including fixed-width NumPy byte-string labels.

`bayesian_mlp` uses actual variational weight posteriors and explicit KL/training-size scaling. GaussianNB is a separate Naive Bayes model. SVM probability estimation requires caller-supplied calibration folds or held-out calibration data.

See [the complete head guide](../heads/README.md) for installation, all-34-model attachment, training examples, classifier fitting, serialization, calibration and CPU build instructions. [CPU coverage](../heads/COVERAGE.md) records tested paths. Survival heads live in [`openhisto.mil.survival`](../survival/README.md).
