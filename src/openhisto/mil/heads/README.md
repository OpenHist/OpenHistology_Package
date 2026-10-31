# Classification and survival heads

Attach a new task head to any of the **34 MIL profiles**, or fit a classical classifier on frozen slide vectors. Native MIL outputs, internal classifiers, auxiliary losses, teachers and anchors remain available.

The public components live in `openhisto.mil.classification`, `openhisto.mil.survival` and `openhisto.mil.heads`. Cohort loaders, automatic splits, training orchestration, hyperparameter searches and `.h5ad` loading are deferred. [COVERAGE.md](COVERAGE.md) and [coverage.json](coverage.json) record actual synthetic CPU verification.

## Environment

From the repository root, use a Python 3.12 virtual environment:

```bash
python3.12 -m venv .venv-heads
source .venv-heads/bin/activate
# All existing MIL backbones:
python -m pip install -e ".[mil-backbones]"
# Classical backends and safe persistence, including optional boosted trees:
python -m pip install -e ".[mil-classical-all]"
```

Standalone neural heads and survival heads only need the `mil` extra. XGBoost, LightGBM and CatBoost have their own extras (`mil-classical-xgboost`, `mil-classical-lightgbm`, `mil-classical-catboost`). The full tested dependency set is captured in `requirements-tested.lock.txt`.

On macOS without OpenMP, the default XGBoost/LightGBM wheels may fail to import. The tested serial CPU build uses the existing compiler and environment-local CMake/Ninja, without changing system libraries:

```bash
bash src/openhisto/mil/classification/install_serial_boosting.sh "$VIRTUAL_ENV/bin/python"
```

This script requires an activated virtual environment and builds XGBoost 3.4.1 and LightGBM 4.7.0 with OpenMP disabled. Other platforms with working CPU wheels can use those wheels normally.

## Neural classification and joint MIL training

```python
import torch
from openhisto.mil.models.code_bases import create_model
from openhisto.mil.classification import create_neural_head
from openhisto.mil.heads import attach_head, representation_dimension

backbone = create_model("clam_sb", input_dim=768, num_classes=3)
head = create_neural_head("mlp", representation_dimension(backbone), num_classes=3)
model = attach_head(backbone, head, train_backbone=True).train()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

bag = torch.randn(32, 768)  # synthetic patch features for one slide
result = model(bag)
loss = model.head.loss(result.head, torch.tensor([1]))
# Add appropriate native auxiliary objectives explicitly when required.
optimizer.zero_grad()
loss.backward()
optimizer.step()
model.after_optimizer_step()  # forwards MHIM/ASMIL update hooks when present
```

Neural profiles are `linear`, `mlp` and `bayesian_mlp`. MLP defaults to one 256-unit ReLU hidden layer with configurable dropout. Heads accept finite `[slides,D]` floating tensors on CPU; logits contain one column per configured class, including two columns for binary tasks. Targets are integer class indices; optional `class_names` records column meanings, and `class_weights` configures cross-entropy weights.

`TaskOutput.head` contains new predictions, `TaskOutput.representation` is `[1,D]`, and `TaskOutput.mil` is the unchanged native `MILOutput`, including `aux` and `native_output`. New-head losses do not automatically train unused native classifiers or replace method-specific objectives. Cached patch features remain fixed unless callers explicitly provide differentiable features.

Use `train_backbone=False` to train only the new head. The supplied backbone's parameters are frozen, stale gradients are cleared and it stays in evaluation mode even when the wrapper trains. It will not update BatchNorm, centers or teacher/anchor state. Attachment takes ownership of the supplied modules; construct another backbone or restore intended parameter flags to change freezing policy later.

## Representations and native behavior

`get_representation_spec(backbone)` and `representation_dimension(backbone)` inspect declared boundaries without a probe forward. [README_BINDINGS.md](README_BINDINGS.md) documents every model's binding.

Attention-profile specifications include effective factory options, including defaults. Readout settings such as VarPool's activation and DTFD's selected-instance count must match when loading a head against an expected backbone, even when their tensor shapes and weights are identical.

- Class-conditioned DSMIL, CLAM-MB and MHIM/DSMIL vectors are flattened in native class order; the representation retains its original class count. No class averaging is invented.
- Graph and hierarchical models use explicit pooled-feature boundaries. Prepared graphs, coordinates and multiscale mappings remain caller supplied.
- SlideGraph+ concatenates its per-layer pooled features. This is an explicitly marked downstream-head adaptation; native prediction still sums layer-wise decision scores.
- nnMIL's downstream head uses its returned mean window feature. A nonlinear downstream head differs from the native ensemble of window logits; both outputs remain available.
- Hard selection can block gradients to earlier stages. Native intermediate losses and selectors remain available; the library does not synthesize new survival versions of classification auxiliaries.

No arbitrary pooling policy is guessed. The initial `representation` option is `"profile"`; unknown backbones or mismatched head dimensions fail before a native forward. Temporary capture hooks are removed on success and failure.

## Bayesian neural uncertainty

```python
bayes = create_neural_head("bayesian_mlp", input_dim=768, num_classes=3)
vectors = torch.randn(8, 768)
output = bayes(vectors)
loss = bayes.loss(
    output, torch.tensor([0, 1, 2, 0, 1, 2, 0, 1]), num_train_samples=400
)  # full training-set size, not this batch size
distribution = bayes.predict_distribution(vectors, mc_samples=50, seed=0)
mean_probability = distribution.probabilities
epistemic = distribution.uncertainty["mutual_information"]
```

The Bayesian head learns diagonal Gaussian weight posteriors with standard-normal priors and analytic KL. Its objective is cross-entropy plus `kl_weight * KL / num_train_samples`; training-set size must be explicit. Training samples weights; ordinary evaluation uses posterior-mean weights. `predict_distribution` averages sampled **probabilities**, returning population predictive variance, predictive/expected entropy and mutual information. It restores model modes and RNG state and accumulates moments without keeping all samples.

Gaussian Naive Bayes is a separate fitted classifier. Its class probabilities do not represent a neural weight posterior or the same uncertainty estimates.

## Frozen vectors and classical classifiers

```python
from openhisto.mil.classification import create_classifier, load_classifier
from openhisto.mil.heads import extract_slide_embeddings

vectors, specification = extract_slide_embeddings(backbone, bags)
# bags is an iterable of native inputs or (inputs, per-slide kwargs) pairs.
classifier = create_classifier("random_forest", n_estimators=200)
classifier.fit(vectors.numpy(), labels, provenance={"representation_spec": specification})
probabilities = classifier.predict_proba(test_vectors)
print(classifier.classes_)  # probability-column order, using original labels
classifier.save("forest.classifier")
restored = load_classifier("forest.classifier")
```

Classical profiles are `logistic_regression`, `random_forest`, `extra_trees`, `hist_gradient_boosting`, `svm`, `knn`, `gaussian_nb`, `sklearn_mlp`, `xgboost`, `lightgbm` and `catboost`. `mlp` aliases `sklearn_mlp` in `create_classifier`; it means the Torch MLP in `create_neural_head`.

Extraction runs the backbone in evaluation/no-grad mode, preserves iterable order and restores mixed train/eval modes afterwards. Returned vectors are CPU tensors. The representation specification records the actual frozen backbone state fingerprint and rejects state changes during export. No patient IDs or labels are inferred.

Classifiers use `fit(X,y)`, `predict(X)`, `predict_proba(X)` and `classes_`. They require fixed-width finite vectors, preserve original numeric/string label ordering and never backpropagate into MIL. Scaling for logistic regression, SVM, kNN and sklearn MLP is fitted on training rows. Implicit validation/early-stopping splits are disabled. Backend options, including supported class weighting, are explicit.

Fixed-width NumPy byte-string labels also roundtrip without decoding or changing their dtype, including labels containing non-UTF-8 bytes. Their JSON class mapping records an explicit base64 encoding; predictions still return the original labels.

### SVM calibration

Uncalibrated SVMs expose labels/decision scores; probabilities require explicit calibration:

```python
svm = create_classifier("svm", calibration="sigmoid")
svm.fit(X_train, y_train, groups=train_groups, calibration_cv=explicit_index_pairs)
# Alternatively use calibration_data=(X_calibration,y_calibration), with
# calibration_groups when training groups are supplied.
```

CV inputs are explicit disjoint train/calibration index pairs covering rows once. Every fold must contain all classes, and supplied groups must not overlap across fold sides. Scaling stays inside each fold's training pipeline. Held-out calibration keeps its scaler fitted on training data only. The library does not create patient folds or infer missing grouping.

## Survival

`openhisto.mil.survival` provides generic `cox` and `discrete_hazard` heads, with linear/MLP networks, training-only calibration and censoring-aware losses. See [the survival API](../survival/README.md) for executable examples and interval semantics.

Always supply durations and `event=1` for observed events, `0` for right-censoring. Clinical endpoints and units are not inferred. Cox returns log-relative risk until a training-only Breslow baseline is fitted. Cox loss requires a cohort/risk set; calling it independently on each one-slide forward is not a training procedure.

Discrete heads default to 20 positive-width intervals fitted on training durations, or accept explicit boundaries. Events contribute preceding survival plus their interval hazard; censoring contributes only completed intervals. Out-of-horizon labels fail unless administrative truncation is explicitly enabled. Original labels and aggregate truncation counts remain available. Discrete risk is the named native-MIL `-sum(bin survival)` proxy, **not expected survival time**.

## Artifacts and verification

```python
from openhisto.mil.heads import save_head, load_head

save_head(model, "task.milhead")  # head-only payload; current backbone fingerprint included
head = load_head("task.milhead", expected_backbone=model.backbone)
```

Head artifacts contain JSON configuration/metadata and tensor-only state. Saving an attached wrapper records the current backbone fingerprint and representation; the artifact contains the **task head**, not the full MIL backbone. Retain the matching backbone and its factory configuration separately, or save the complete wrapper's `state_dict()` for a training checkpoint. Optimizer/cohort checkpoint orchestration is deferred.

Classical artifacts use a fixed reviewed skops trust list or native XGBoost JSON, LightGBM text and CatBoost CBM. Class mapping, scalers, provenance and dependency versions accompany the payload. Native/safe formats avoid executable model-object pickles. KNN/SVM artifacts necessarily retain training/support vectors as part of their estimators. Head/classifier publication is atomic and refuses existing paths by default; loaders check structure, size limits and integrity. Checksums are integrity checks, not authentication of an artifact's author.

`load_classifier` defaults to `version_policy="strict"`, requiring recorded dependency versions. For native XGBoost/LightGBM/CatBoost payloads, opt into `version_policy="native"` to let the backend reader load across version differences; a warning lists differences and feature/class/integrity checks still apply. Verify predictions in the target environment. Sklearn/skops artifacts retain strict version matching. Re-saving a native model records the current export environment and preserves prior fit versions in `training_versions`.

```bash
python -m openhisto.mil.classification list
python -m openhisto.mil.classification info bayesian_mlp
python -m openhisto.mil.heads.validate --write-report
```

Validation executes real CPU backbones and all classical backends on synthetic inputs, including the existing MIL regression suite, head gradients/freezing/reload, Bayesian KL and MC checks, survival likelihood oracles, time-grid boundaries, calibration isolation and artifact round trips. No clinical records or real model checkpoints are used. It does not establish clinical validity, real-cohort accuracy or GPU behavior.
