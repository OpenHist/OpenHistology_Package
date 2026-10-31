"""Classification heads and fitted classifiers with lazy backend imports."""

from importlib.util import find_spec

_NEURAL = ("linear", "mlp", "bayesian_mlp")
_CLASSICAL = (
    "logistic_regression",
    "random_forest",
    "extra_trees",
    "hist_gradient_boosting",
    "svm",
    "knn",
    "gaussian_nb",
    "sklearn_mlp",
    "xgboost",
    "lightgbm",
    "catboost",
)


def head_info(name, *, kind=None):
    kind = ("neural" if name in _NEURAL else "classifier") if kind is None else kind
    if kind == "classifier" and name == "mlp":
        name = "sklearn_mlp"
    names = _NEURAL if kind == "neural" else _CLASSICAL if kind == "classifier" else ()
    if name not in names:
        raise ValueError("Unknown classification profile/kind; call list_heads()")
    dependency = (
        "torch"
        if kind == "neural"
        else name
        if name in {"xgboost", "lightgbm", "catboost"}
        else "sklearn"
    )
    return {
        "name": name,
        "kind": kind,
        "task": "binary_or_multiclass_single_label",
        "differentiable": kind == "neural",
        "input": "one fixed vector per slide",
        "dependency": dependency,
        "dependency_present": find_spec(dependency) is not None,
        "posterior_uncertainty": name == "bayesian_mlp",
        "probability_requirement": "explicit calibration data/folds"
        if name == "svm"
        else "fitted model"
        if kind == "classifier"
        else None,
    }


def list_heads(*, kind=None):
    if kind not in {None, "neural", "classifier"}:
        raise ValueError("kind must be neural or classifier")
    return (
        [head_info(name, kind="neural") for name in _NEURAL] if kind != "classifier" else []
    ) + ([head_info(name, kind="classifier") for name in _CLASSICAL] if kind != "neural" else [])


def create_neural_head(name, input_dim, num_classes, **options):
    from .neural import create_neural_head as factory

    return factory(name, input_dim, num_classes, **options)


def create_classifier(name, **options):
    from .classical import create_classifier as factory

    return factory(name, **options)


def list_classifiers():
    from .classical import list_classifiers as listing

    return listing()


def classifier_info(name):
    from .classical import classifier_info as info

    return info(name)


def classification_loss(*args, **kwargs):
    from .neural import classification_loss as loss

    return loss(*args, **kwargs)


def load_classifier(*args, **kwargs):
    from .classical import load_classifier as loader

    return loader(*args, **kwargs)


def save_head(*args, **kwargs):
    from openhisto.mil.heads.persistence import save_head as save

    return save(*args, **kwargs)


def load_head(*args, **kwargs):
    from openhisto.mil.heads.persistence import load_head as load

    return load(*args, **kwargs)


__all__ = [
    "list_heads",
    "head_info",
    "list_classifiers",
    "classifier_info",
    "create_neural_head",
    "create_classifier",
    "classification_loss",
    "save_head",
    "load_head",
    "load_classifier",
]
