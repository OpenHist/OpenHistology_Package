"""Neural Cox and discrete-hazard heads for explicit bag-level embeddings."""

import torch
from torch import nn
from torch.nn import functional as F

from openhisto.mil.heads.types import SurvivalOutput
from ._validation import positive_integer
from .baseline import CoxBaseline, fit_cox_baseline
from .grid import TimeGrid, fit_time_grid
from .losses import cox_ph_loss, discrete_hazard_loss


class _Head(nn.Module):
    def __init__(
        self,
        name,
        input_dim,
        output_dim,
        *,
        network="linear",
        hidden_dims=None,
        activation="relu",
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.name = name
        self.input_dim = positive_integer(input_dim, "input_dim")
        self.network_name = network
        if network not in {"linear", "mlp"}:
            raise ValueError("network must be linear or mlp")
        if activation not in {"relu", "gelu", "silu", "tanh"}:
            raise ValueError("activation must be relu, gelu, silu or tanh")
        if (
            isinstance(dropout, bool)
            or not isinstance(dropout, (int, float))
            or not 0 <= dropout < 1
        ):
            raise ValueError("dropout must be in [0,1)")
        if not isinstance(bias, bool):
            raise ValueError("bias must be a boolean")
        hidden_dims = [128] if hidden_dims is None else list(hidden_dims)
        if network == "mlp" and not hidden_dims:
            raise ValueError("MLP heads require at least one hidden dimension")
        for width in hidden_dims:
            positive_integer(width, "hidden dimension")
        self.hidden_dims = hidden_dims if network == "mlp" else []
        self.activation, self.dropout, self.bias = activation, float(dropout), bias
        self.metadata = {
            "family": "survival",
            "name": name,
            "input_dim": self.input_dim,
            "network": network,
            "supported_devices": ["cpu"],
            "event_convention": "1=observed_event,0=right_censored",
            "time_unit": "unchanged_from_supplied_durations",
        }
        act = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU, "tanh": nn.Tanh}[activation]
        modules, width = [], input_dim
        for hidden in self.hidden_dims:
            modules.extend((nn.Linear(width, hidden, bias=bias), act(), nn.Dropout(dropout)))
            width = hidden
        modules.append(nn.Linear(width, output_dim, bias=bias))
        self.network = nn.Sequential(*modules)

    @property
    def config(self):
        return {
            "family": "survival",
            "name": self.name,
            "input_dim": self.input_dim,
            "network": self.network_name,
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
            "dropout": self.dropout,
            "bias": self.bias,
        }

    def _features(self, features):
        if not isinstance(features, torch.Tensor) or features.device.type != "cpu":
            raise ValueError("Survival heads require CPU floating-point embeddings [N,D]")
        if features.ndim == 1:
            features = features.unsqueeze(0)
        if features.ndim != 2 or features.shape[0] < 1 or features.shape[1] != self.input_dim:
            raise ValueError(f"Expected nonempty embeddings [N,{self.input_dim}]")
        if not features.is_floating_point() or not torch.isfinite(features).all():
            raise ValueError("Embeddings must be finite floating-point values")
        return features.to(dtype=next(self.network.parameters()).dtype)

    def _logits(self, features):
        logits = self.network(self._features(features))
        if not torch.isfinite(logits).all():
            raise ValueError(
                "Survival network produced nonfinite logits; check inputs and fitted weights"
            )
        return logits


class CoxHead(_Head):
    def __init__(
        self,
        input_dim,
        *,
        network="linear",
        hidden_dims=None,
        activation="relu",
        dropout=0.0,
        bias=True,
    ):
        super().__init__(
            "cox",
            input_dim,
            1,
            network=network,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )
        self.register_buffer("baseline_times", torch.empty(0, dtype=torch.float64))
        self.register_buffer("baseline_log_cumulative_hazard", torch.empty(0, dtype=torch.float64))
        self.register_buffer("baseline_risk_center", torch.zeros((), dtype=torch.float64))
        self.register_buffer("baseline_training_horizon", torch.zeros((), dtype=torch.float64))
        self.register_buffer("baseline_event_count", torch.zeros((), dtype=torch.long))

    @property
    def baseline_fitted(self):
        return bool(self.baseline_event_count > 0)

    @property
    def baseline(self):
        if not self.baseline_fitted:
            raise ValueError("Fit a training-only Cox baseline before requesting absolute survival")
        return CoxBaseline(
            self.baseline_times,
            self.baseline_log_cumulative_hazard,
            self.baseline_risk_center,
            self.baseline_training_horizon,
            int(self.baseline_event_count),
        )

    def set_baseline(self, baseline):
        if not isinstance(baseline, CoxBaseline):
            raise TypeError("baseline must be a CoxBaseline from fit_cox_baseline")
        self.baseline_times = baseline.times.clone()
        self.baseline_log_cumulative_hazard = baseline.log_cumulative_hazard.clone()
        self.baseline_risk_center = baseline.risk_center.clone()
        self.baseline_training_horizon = baseline.training_horizon.clone()
        self.baseline_event_count.fill_(baseline.event_count)
        self.metadata["baseline"] = baseline.metadata
        return self

    def fit_baseline(self, embeddings, durations, events, *, split="train"):
        """Calibrate using eval-mode predictions of the fitted head on training embeddings."""
        previous = self.training
        try:
            self.eval()
            with torch.no_grad():
                risk = self._logits(embeddings)[:, 0]
            baseline = fit_cox_baseline(risk, durations, events, split=split)
            self.set_baseline(baseline)
        finally:
            self.train(previous)
        return baseline

    def forward(self, embeddings, *, times=None, allow_extrapolation=False):
        if not isinstance(allow_extrapolation, bool):
            raise ValueError("allow_extrapolation must be an explicit boolean")
        logits = self._logits(embeddings)
        risk = logits[:, 0]
        if self.baseline_fitted:
            output = self.baseline.predict(risk, times, allow_extrapolation=allow_extrapolation)
            output.logits, output.risk = logits, risk
            return output
        if times is not None:
            raise ValueError("Fit a training-only Cox baseline before requesting survival at times")
        return SurvivalOutput(
            risk=risk,
            logits=logits,
            aux={"baseline_fitted": False, "risk_definition": "log_relative_hazard"},
        )

    def loss(self, output, durations, events, **options):
        risk = output.risk if isinstance(output, SurvivalOutput) else output
        return cox_ph_loss(risk, durations, events, **options)

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        # Calibrated event-time count varies by training cohort. Resize only
        # these declared aggregate buffers before strict key/weight loading.
        for name in ("baseline_times", "baseline_log_cumulative_hazard"):
            value = state_dict.get(prefix + name)
            if isinstance(value, torch.Tensor) and value.ndim == 1:
                self._buffers[name] = torch.empty_like(value, device="cpu")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        try:
            if self.baseline_event_count < 0:
                raise ValueError("Baseline event count cannot be negative")
            if self.baseline_fitted:
                self.baseline
            elif self.baseline_times.numel() or self.baseline_log_cumulative_hazard.numel():
                raise ValueError("Unfitted Cox state cannot contain baseline times/hazards")
        except (ValueError, RuntimeError) as error:
            error_msgs.append(f"{prefix}invalid Cox baseline state: {error}")


class DiscreteHazardHead(_Head):
    def __init__(
        self,
        input_dim,
        *,
        network="linear",
        hidden_dims=None,
        activation="relu",
        dropout=0.0,
        bias=True,
        num_bins=None,
        boundaries=None,
        administrative_truncation=False,
    ):
        grid = None if boundaries is None else TimeGrid(boundaries)
        bins = (
            (grid.num_bins if grid is not None else 20)
            if num_bins is None
            else positive_integer(num_bins, "num_bins")
        )
        if grid is not None and bins != grid.num_bins:
            raise ValueError("num_bins does not match the explicit boundaries")
        super().__init__(
            "discrete_hazard",
            input_dim,
            bins,
            network=network,
            hidden_dims=hidden_dims,
            activation=activation,
            dropout=dropout,
            bias=bias,
        )
        if not isinstance(administrative_truncation, bool):
            raise ValueError("administrative_truncation must be an explicit boolean")
        self.num_bins = bins
        self.administrative_truncation = administrative_truncation
        self.register_buffer(
            "boundaries",
            torch.empty(0, dtype=torch.float64) if grid is None else grid.boundaries.clone(),
        )
        self.register_buffer("grid_fitted_count", torch.zeros((), dtype=torch.long))

    @property
    def grid_fitted(self):
        return self.boundaries.numel() > 0

    @property
    def grid(self):
        if not self.grid_fitted:
            raise ValueError("Fit a training time grid or supply explicit boundaries first")
        return TimeGrid(self.boundaries, int(self.grid_fitted_count))

    @property
    def config(self):
        return {
            **super().config,
            "num_bins": self.num_bins,
            "boundaries": self.boundaries.tolist() if self.grid_fitted else None,
            "administrative_truncation": self.administrative_truncation,
        }

    def fit_time_grid(
        self,
        durations,
        events=None,
        *,
        boundaries=None,
        split="train",
        administrative_truncation=None,
    ):
        truncate = (
            self.administrative_truncation
            if administrative_truncation is None
            else administrative_truncation
        )
        if boundaries is None and self.grid_fitted:
            boundaries = self.boundaries
        grid = fit_time_grid(
            durations,
            events,
            num_bins=self.num_bins,
            boundaries=boundaries,
            administrative_truncation=truncate,
            split=split,
        )
        if grid.num_bins != self.num_bins:
            raise ValueError(
                "The fitted grid width differs from the neural output; construct a head with matching num_bins"
            )
        self.boundaries = grid.boundaries.clone()
        self.grid_fitted_count.fill_(grid.fitted_count)
        self.metadata["time_grid"] = grid.metadata
        return grid

    def forward(self, embeddings):
        logits = self._logits(embeddings)
        hazards = logits.sigmoid()
        # log(1-sigmoid(z)) = -softplus(z); avoids log(0) at large logits.
        cumulative = F.softplus(logits.to(torch.float64)).cumsum(dim=1)
        bin_survival = torch.exp(-cumulative)
        return SurvivalOutput(
            risk=-bin_survival.sum(dim=1),
            logits=logits,
            hazards=hazards,
            survival=bin_survival if self.grid_fitted else None,
            time_grid=self.boundaries[1:].clone() if self.grid_fitted else None,
            cumulative_hazard=cumulative,
            aux={
                "risk_definition": "negative_sum_of_bin_survival_native_MIL_convention",
                "risk_is_expected_time": False,
                "grid_fitted": self.grid_fitted,
                "grid": self.grid.metadata if self.grid_fitted else None,
                "bin_survival": bin_survival,
            },
        )

    def loss(
        self,
        output,
        durations,
        events,
        *,
        administrative_truncation=None,
        reduction="mean",
        return_targets=False,
    ):
        logits = output.logits if isinstance(output, SurvivalOutput) else output
        truncate = (
            self.administrative_truncation
            if administrative_truncation is None
            else administrative_truncation
        )
        return discrete_hazard_loss(
            logits,
            durations,
            events,
            self.grid,
            administrative_truncation=truncate,
            reduction=reduction,
            return_targets=return_targets,
        )

    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
    ):
        value = state_dict.get(prefix + "boundaries")
        if isinstance(value, torch.Tensor) and value.ndim == 1:
            self._buffers["boundaries"] = torch.empty_like(value, device="cpu")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )
        try:
            if self.grid_fitted and self.grid.num_bins != self.num_bins:
                raise ValueError("Grid interval count does not match the neural output")
            if not self.grid_fitted and self.grid_fitted_count != 0:
                raise ValueError("Unfitted grid cannot have a fitted observation count")
        except (ValueError, RuntimeError) as error:
            error_msgs.append(f"{prefix}invalid discrete time grid state: {error}")
