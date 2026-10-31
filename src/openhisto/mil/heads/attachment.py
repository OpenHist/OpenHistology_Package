"""Trainable task heads attached to explicit, preserved native MIL backbones."""

from copy import deepcopy

import torch
from torch import nn

from .bindings import _binding
from .types import ClassificationOutput, SurvivalOutput, TaskOutput


class AttachedHead(nn.Module):
    def __init__(self, backbone, head, *, representation="profile", train_backbone=True):
        super().__init__()
        if not isinstance(backbone, nn.Module) or not isinstance(head, nn.Module):
            raise TypeError("A backbone and a trainable Torch head are required")
        if not isinstance(train_backbone, bool):
            raise TypeError("train_backbone must be boolean")
        binding = _binding(backbone, representation)
        dimension = binding.spec["dimension"]
        input_dim = getattr(head, "input_dim", None)
        if isinstance(input_dim, bool) or not isinstance(input_dim, int) or input_dim != dimension:
            raise ValueError(
                f"Head input_dim must equal the declared representation dimension {dimension}"
            )
        self.backbone, self.head = backbone, head
        self.train_backbone = train_backbone
        self._binding = binding
        self.representation_spec = deepcopy(binding.spec)
        self.representation_dim = dimension
        metadata = getattr(head, "metadata", None)
        if isinstance(metadata, dict):
            metadata["representation_spec"] = deepcopy(self.representation_spec)
        mode = backbone.training
        if not train_backbone:
            self.backbone.requires_grad_(False)
            for parameter in self.backbone.parameters():
                parameter.grad = None
        self.train(mode)

    def train(self, mode=True):
        super().train(mode)
        if not self.train_backbone:
            self.backbone.eval()
        return self

    def forward(self, inputs, **forward_kwargs):
        if not self.train_backbone:
            self.backbone.eval()
        with self._binding.capture() as captured:
            if self.train_backbone:
                mil = self.backbone(inputs, **forward_kwargs)
            else:
                # Ordinary no_grad produces tensors which a trainable head may
                # save for backward; inference_mode would not permit that.
                with torch.no_grad():
                    mil = self.backbone(inputs, **forward_kwargs)
            value = self._binding.resolve(self.backbone, mil, captured)
        output = self.head(value)
        if not isinstance(output, (ClassificationOutput, SurvivalOutput)):
            raise TypeError("The attached head must return ClassificationOutput or SurvivalOutput")
        return TaskOutput(output, value, mil, deepcopy(self.representation_spec))

    def _update(self, method, *args, **kwargs):
        if not self.train_backbone:
            return None
        function = getattr(self.backbone, method, None)
        if not callable(function):
            raise AttributeError(f"The backbone does not define {method}")
        return function(*args, **kwargs)

    def after_optimizer_step(self, *args, **kwargs):
        if not self.train_backbone:
            return None
        function = getattr(self.backbone, "after_optimizer_step", None)
        if callable(function):
            return function(*args, **kwargs)
        for method in ("update_teacher", "update_anchor"):
            function = getattr(self.backbone, method, None)
            if callable(function):
                return function(*args, **kwargs)
        return None

    def update_teacher(self, *args, **kwargs):
        return self._update("update_teacher", *args, **kwargs)

    def update_anchor(self, *args, **kwargs):
        return self._update("update_anchor", *args, **kwargs)


def attach_head(backbone, head, representation="profile", train_backbone=True):
    return AttachedHead(
        backbone, head, representation=representation, train_backbone=train_backbone
    )


def extract_slide_embeddings(
    backbone, inputs_iterable, *, representation="profile", forward_kwargs=None
):
    """Extract CPU rows in iterable order and return ``(matrix, spec)``.

    Entries are native inputs or ``(inputs, per_slide_forward_kwargs)`` pairs.
    Caller-supplied shared kwargs are copied, then overridden by per-slide kwargs.
    No identifiers, labels, graph structures, or patient splits are inferred.
    """
    binding = _binding(backbone, representation)
    from .provenance import state_fingerprint

    fingerprint = state_fingerprint(backbone)
    modes = [(module, module.training) for module in backbone.modules()]
    rows = []
    try:
        backbone.eval()
        with torch.no_grad():
            for item in inputs_iterable:
                kwargs = dict(forward_kwargs or {})
                if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], dict):
                    inputs, current = item
                    kwargs.update(current)
                else:
                    inputs = item
                with binding.capture() as captured:
                    mil = backbone(inputs, **kwargs)
                    row = binding.resolve(backbone, mil, captured)
                rows.append(row.detach().cpu().clone())
    finally:
        # Restore mixed modes exactly, including intentionally frozen teachers.
        for module, training in modes:
            module.training = training
    matrix = (
        torch.cat(rows, dim=0)
        if rows
        else torch.empty((0, binding.spec["dimension"]), dtype=torch.float32)
    )
    if state_fingerprint(backbone) != fingerprint:
        raise RuntimeError("Backbone state changed during embedding extraction")
    spec = deepcopy(binding.spec)
    spec["backbone_provenance"]["state_sha256"] = fingerprint
    return matrix, spec
