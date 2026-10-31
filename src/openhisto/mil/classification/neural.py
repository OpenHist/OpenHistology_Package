"""Differentiable linear, MLP and variational Bayesian classification heads."""

from contextlib import contextmanager
import math

import torch
from torch import nn
from torch.nn import functional as F

from openhisto.mil.heads.types import ClassificationOutput


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@contextmanager
def _evaluating(module):
    modes = [(child, child.training) for child in module.modules()]
    module.eval()
    try:
        yield
    finally:
        for child, mode in modes:
            child.training = mode


class BayesianLinear(nn.Module):
    """Reparameterized diagonal Normal posterior with analytic Normal KL."""

    def __init__(self, in_features, out_features, *, prior_std=1.0, posterior_std=0.05):
        super().__init__()
        if not all(math.isfinite(value) and value > 1e-6 for value in (prior_std, posterior_std)):
            raise ValueError("prior_std/posterior_std must be finite and greater than 1e-6")
        self.in_features, self.out_features = in_features, out_features
        self.prior_std = float(prior_std)
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        rho = math.log(math.expm1(posterior_std - 1e-6))
        self.weight_rho = nn.Parameter(torch.full_like(self.weight_mu, rho))
        self.bias_rho = nn.Parameter(torch.full_like(self.bias_mu, rho))
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))

    @staticmethod
    def _std(rho):
        return F.softplus(rho) + 1e-6

    def forward(self, inputs, *, sample=True):
        weight, bias = self.weight_mu, self.bias_mu
        if sample:
            weight = weight + self._std(self.weight_rho) * torch.randn_like(weight)
            bias = bias + self._std(self.bias_rho) * torch.randn_like(bias)
        return F.linear(inputs, weight, bias)

    def kl_divergence(self):
        result = self.weight_mu.new_zeros(())
        for mu, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            std = self._std(rho)
            result = (
                result
                + (
                    torch.log(self.prior_std / std)
                    + (std.square() + mu.square()) / (2 * self.prior_std**2)
                    - 0.5
                ).sum()
            )
        return result


class NeuralClassificationHead(nn.Module):
    def __init__(
        self,
        name,
        input_dim,
        num_classes,
        *,
        hidden_dims=None,
        dropout=0.1,
        prior_std=1.0,
        posterior_std=0.05,
        seed=0,
        class_names=None,
        class_weights=None,
    ):
        super().__init__()
        self.input_dim = _positive_int(input_dim, "input_dim")
        self.num_classes = _positive_int(num_classes, "num_classes")
        if self.num_classes < 2:
            raise ValueError("Single-label classification uses at least two logits/classes")
        if name not in {"linear", "mlp", "bayesian_mlp"}:
            raise ValueError("Neural head must be linear, mlp or bayesian_mlp")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        if not math.isfinite(dropout) or not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        if hidden_dims is None:
            hidden_dims = [] if name == "linear" else [256]
        hidden_dims = [_positive_int(value, "hidden dimension") for value in hidden_dims]
        if name == "linear" and hidden_dims:
            raise ValueError("A linear head has no hidden layers")
        names = list(range(num_classes)) if class_names is None else list(class_names)
        if (
            len(names) != num_classes
            or len(set(names)) != num_classes
            or any(
                not isinstance(value, (str, int, float)) or isinstance(value, bool)
                for value in names
            )
        ):
            raise ValueError("class_names must contain one distinct JSON scalar per class")
        if any(isinstance(value, float) and not math.isfinite(value) for value in names):
            raise ValueError("Class names must be finite")
        weights = (
            None if class_weights is None else torch.as_tensor(class_weights, dtype=torch.float32)
        )
        if weights is not None and (
            weights.shape != (num_classes,)
            or not torch.isfinite(weights).all()
            or not (weights > 0).all()
        ):
            raise ValueError("class_weights must contain one finite positive weight per class")
        self.register_buffer("class_weights", weights)
        self.name = name
        self.config = dict(
            family="classification",
            name=name,
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=float(dropout),
            prior_std=float(prior_std),
            posterior_std=float(posterior_std),
            seed=seed,
            class_names=names,
            class_weights=None if weights is None else weights.tolist(),
        )
        self.metadata = {
            "task": "single_label_classification",
            "profile": name,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "classes": names,
            "posterior": "diagonal_normal" if name == "bayesian_mlp" else None,
        }
        self.layers = nn.ModuleList()
        dimensions = [input_dim, *hidden_dims, num_classes]
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            for first, second in zip(dimensions, dimensions[1:]):
                self.layers.append(
                    BayesianLinear(first, second, prior_std=prior_std, posterior_std=posterior_std)
                    if name == "bayesian_mlp"
                    else nn.Linear(first, second)
                )
        self.dropout = nn.Dropout(dropout)

    def _inputs(self, values):
        if (
            not isinstance(values, torch.Tensor)
            or values.ndim != 2
            or values.shape[1] != self.input_dim
            or not len(values)
        ):
            raise ValueError(f"Head requires nonempty [samples,{self.input_dim}] tensors")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError("Head inputs must be finite floating-point features")
        if values.device.type != "cpu":
            raise ValueError("This release requires CPU feature tensors")
        return values.to(dtype=next(self.parameters()).dtype)

    def kl_divergence(self):
        return sum(
            (layer.kl_divergence() for layer in self.layers if isinstance(layer, BayesianLinear)),
            start=next(self.parameters()).new_zeros(()),
        )

    def forward(self, inputs, *, sample=None):
        values = self._inputs(inputs)
        sample = self.training if sample is None else sample
        if not isinstance(sample, bool):
            raise ValueError("sample must be boolean or None")
        for index, layer in enumerate(self.layers):
            values = (
                layer(values, sample=sample) if isinstance(layer, BayesianLinear) else layer(values)
            )
            if index < len(self.layers) - 1:
                values = self.dropout(F.relu(values))
        if not torch.isfinite(values).all():
            raise ValueError(
                "The head produced nonfinite logits; check input scale and fitted weights"
            )
        kl = self.kl_divergence() if self.name == "bayesian_mlp" else None
        return ClassificationOutput(values, values.softmax(-1), kl=kl)

    def loss(self, output, targets, *, num_train_samples=None, kl_weight=1.0):
        return classification_loss(
            output,
            targets,
            class_weights=self.class_weights,
            num_train_samples=num_train_samples,
            kl_weight=kl_weight,
        )

    def predict_distribution(self, inputs, *, mc_samples=50, seed=0):
        _positive_int(mc_samples, "mc_samples")
        if self.name != "bayesian_mlp":
            with _evaluating(self), torch.no_grad():
                return self(inputs, sample=False)
        inputs = self._inputs(inputs)
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        devices = [inputs.device.index] if inputs.is_cuda else []
        with _evaluating(self), torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.random.default_generator.manual_seed(seed)
            mean = inputs.new_zeros((len(inputs), self.num_classes))
            m2, entropy_sum = torch.zeros_like(mean), inputs.new_zeros(len(inputs))
            for index in range(mc_samples):
                probabilities = self(inputs, sample=True).probabilities
                delta = probabilities - mean
                mean = mean + delta / (index + 1)
                m2 = m2 + delta * (probabilities - mean)
                entropy_sum -= (
                    probabilities
                    * probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny).log()
                ).sum(-1)
            entropy = -(mean * mean.clamp_min(torch.finfo(mean.dtype).tiny).log()).sum(-1)
            expected = entropy_sum / mc_samples
            return ClassificationOutput(
                None,
                mean,
                kl=self.kl_divergence(),
                uncertainty={
                    "predictive_variance": (m2 / mc_samples).clamp_min(0),
                    "predictive_entropy": entropy,
                    "expected_entropy": expected,
                    "mutual_information": (entropy - expected).clamp_min(0),
                    "mc_samples": mc_samples,
                },
            )

    def predict_proba(self, inputs, *, mc_samples=50, seed=0):
        return self.predict_distribution(inputs, mc_samples=mc_samples, seed=seed).probabilities

    def predict(self, inputs, **options):
        return self.predict_proba(inputs, **options).argmax(-1)


def classification_loss(
    output, targets, *, class_weights=None, num_train_samples=None, kl_weight=1.0
):
    logits = output.logits if isinstance(output, ClassificationOutput) else output
    if (
        not isinstance(logits, torch.Tensor)
        or logits.ndim != 2
        or not len(logits)
        or not torch.isfinite(logits).all()
    ):
        raise ValueError(
            "Loss requires finite logits [samples,classes], not predictive probabilities"
        )
    if (
        not isinstance(targets, torch.Tensor)
        or targets.dtype != torch.long
        or targets.shape != (len(logits),)
    ):
        raise ValueError("Targets must be a long vector of class indices")
    if (targets < 0).any() or (targets >= logits.shape[1]).any():
        raise ValueError("Target index is outside the configured class order")
    if not math.isfinite(kl_weight) or kl_weight < 0:
        raise ValueError("kl_weight must be finite and nonnegative")
    weights = (
        None
        if class_weights is None
        else torch.as_tensor(class_weights, device=logits.device, dtype=logits.dtype)
    )
    if weights is not None and (
        weights.shape != (logits.shape[1],)
        or not torch.isfinite(weights).all()
        or not (weights > 0).all()
    ):
        raise ValueError("Class weights must be finite and positive")
    loss = F.cross_entropy(logits, targets, weight=weights)
    if isinstance(output, ClassificationOutput) and output.kl is not None:
        _positive_int(num_train_samples, "num_train_samples for Bayesian KL scaling")
        loss = loss + kl_weight * output.kl / num_train_samples
    return loss


def create_neural_head(name, input_dim, num_classes, *, device="cpu", **options):
    if torch.device(device).type != "cpu":
        raise ValueError('This release validates CPU heads; use device="cpu"')
    return NeuralClassificationHead(name, input_dim, num_classes, **options).cpu().float()
