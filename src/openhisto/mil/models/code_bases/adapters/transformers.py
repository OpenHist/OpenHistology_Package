"""Native transformer/state-space MIL models with portable CPU execution.

The wrappers preserve native computation, parameter names below ``native``,
and method-specific training tensors. MHIM additionally owns its frozen teacher
and exposes explicit EMA hooks; callers remain responsible for their optimizer
and for adding the returned auxiliary losses to their training objective.
"""

from contextlib import contextmanager
from copy import deepcopy

import torch
from torch import nn

from ..base import MILAdapter, as_bag, output_from_logits
from ..compat.loader import load_native
from ..compat.transformer_imports import replacements


def _native(slug, module, roots=("",)):
    return load_native(slug, module, roots=roots, replacements=replacements(slug))


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@contextmanager
def _capture(module):
    values = []
    handle = module.register_forward_pre_hook(lambda _, args: values.append(args[0]))
    try:
        yield values
    finally:
        handle.remove()


class NativeTransformer(MILAdapter):
    def __init__(self, name, input_dim, num_classes, **options):
        super().__init__()
        self.name = name
        self.input_dim = _positive(input_dim, "input_dim")
        self.num_classes = _positive(num_classes, "num_classes")
        self.metadata = {
            "profile": name,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "one_bag_per_call": True,
            "adaptations": [],
            "options": dict(options),
        }
        slug = name
        if name == "transmil":
            landmarks = options.pop("num_landmarks", None)
            if options:
                raise TypeError(f"Unsupported TransMIL options: {sorted(options)}")
            self.native = _native(slug, "TransMIL", ("models",)).TransMIL(n_classes=num_classes)
            if input_dim != 1024:
                self.native._fc1[0] = nn.Linear(input_dim, 512)
            if landmarks is not None:
                for layer in (self.native.layer1, self.native.layer2):
                    layer.attn.num_landmarks = _positive(landmarks, "num_landmarks")
            self.head = "_fc2"
            self.metadata["adaptations"] = [
                "device_local_class_token",
                "configurable_input_projection",
            ]
        elif name == "ilra":
            self.native = _native(slug, "ilra", ("models",)).ILRA(
                feat_dim=input_dim, n_classes=num_classes, **options
            )
            self.head = "classifier"
        elif name == "rrt_mil":
            if options.get("pool", "attn") not in {"attn", "avg"}:
                raise ValueError("RRT pool must be attn or avg")
            if options.get("pos", "none") == "sincos":
                raise ValueError(
                    "RRT pos='sincos' requires native four-dimensional region geometry, "
                    "which the feature-bag API does not supply; use pos='none', 'peg', or 'ppeg'"
                )
            if options.get("pos", "none") not in {"none", "peg", "ppeg"}:
                raise ValueError("RRT pos must be 'none', 'peg', or 'ppeg' for feature bags")
            self.native = _native(slug, "modules.rrt").RRTMIL(
                input_dim=input_dim, n_classes=num_classes, **options
            )
            self.head = "predictor"
            self.metadata["adaptations"] = [
                "projection_honors_mlp_dim",
                "average_pool_over_token_axis",
            ]
        elif name == "dgr_mil":
            self.native = _native(slug, "models.dgrmil").DGRMIL(
                in_features=input_dim, num_classes=num_classes, **options
            )
            self.head = "classifier"
            self.metadata["training_note"] = (
                "Native center updates run during training; centers and lesion embeddings are returned for auxiliary losses"
            )
        elif name == "caprmil":
            options.setdefault("hidden_dim", 512)
            self.native = _native(slug, "model_caprmil", ("src",)).CAPRMIL(
                input_dim=input_dim, num_classes=num_classes, **options
            )
            self.head = "classifier"
        elif name == "lin_mil":
            module = _native(slug, "lin_mil.Lin_MIL")
            config = options.pop("config", None)
            heads = options.pop("heads", 8)
            pos_enc = options.pop("pos_enc", None)
            if config is None:
                config = module.Config(**options)
            elif isinstance(config, dict):
                if options:
                    raise TypeError("Pass Lin-MIL config once")
                config = module.Config(**config)
            elif options:
                raise TypeError("Pass Lin-MIL config once")
            self.native = module.Lin_MIL(
                num_classes=num_classes,
                input_dim=input_dim,
                config=config,
                heads=heads,
                pos_enc=pos_enc,
            )
            self.head = "mlp_head"
        elif name == "mamba_mil":
            options.setdefault("dropout", 0.0)
            options.setdefault("act", "relu")
            if options.get("survival", False):
                raise ValueError(
                    "The mamba_mil profile is classification; use native survival training separately"
                )
            if options.get("type", "SRMamba") not in {"SRMamba", "Mamba", "BiMamba"}:
                raise ValueError("Mamba type must be SRMamba, Mamba, or BiMamba")
            _positive(options.get("layer", 2), "layer")
            _positive(options.get("rate", 10), "rate")
            self.native = _native(slug, "models.MambaMIL").MambaMIL(
                in_dim=input_dim, n_classes=num_classes, **options
            )
            self.head = "classifier"
            self.metadata.update(
                kernel_backend="torch_reference",
                adaptations=[
                    "unfused_exact_causal_convolution_selective_scan",
                    "native_sr_transpose_branch_preserved",
                    "preserve_batch_axis_after_pooling",
                ],
            )
        elif name == "nnmil":
            self.native = _native(slug, "simple_mil", ("network_architecture/models",)).SimpleMIL(
                input_dim=input_dim, pred_num=num_classes, **options
            )
            self.head = "cls"
            self.metadata["adaptations"] = ["population_dispersion_for_single_eval_window"]
        else:
            raise ValueError("Unknown transformer MIL profile")
        self.metadata["repository"] = slug
        self.metadata["native_entrypoint"] = (
            type(self.native).__module__ + "." + type(self.native).__name__
        )

    def _bag(self, inputs):
        return as_bag(inputs, self.input_dim, batched=True).float()

    def forward(self, inputs, **kwargs):
        bag = self._bag(inputs)
        attention, aux = None, {}
        head = getattr(self.native, self.head)
        with _capture(head) as embeddings:
            if self.name == "transmil":
                if kwargs:
                    raise TypeError("TransMIL takes feature tensors only")
                native = self.native(data=bag)
                logits = native["logits"]
            elif self.name == "rrt_mil":
                # Native average pooling has no attention-return signature.
                want_attention = (
                    hasattr(self.native.pool_fn, "attention")
                    or self.native.pool_fn.__class__.__name__ == "DAttention"
                )
                native = self.native(bag, return_attn=want_attention, **kwargs)
                if want_attention:
                    logits, attention = native
                else:
                    logits = native
            elif self.name == "dgr_mil":
                native = self.native(bag, **kwargs)
                logits, attention, instances = native[:3]
                aux["instance_embeddings"] = instances
                if len(native) > 3:
                    aux.update(
                        positive_center=native[3],
                        normal_center=native[4],
                        lesion_embeddings=native[5],
                    )
            elif self.name == "nnmil":
                native = self.native(bag, return_WSI_attn=True, return_WSI_feature=True, **kwargs)
                logits = native["logits"]
                attention = native.get("WSI_attn")
                aux = {
                    key: value
                    for key, value in native.items()
                    if key not in {"logits", "WSI_attn", "WSI_feature"}
                }
            elif self.name in {"ilra", "lin_mil", "mamba_mil"}:
                native = self.native(bag, **kwargs)
                logits = native[0]
                if self.name == "lin_mil":
                    attention = native[1]
            else:
                native = self.native(bag, **kwargs)
                logits = native
        embedding = embeddings[-1] if embeddings else None
        if self.name == "nnmil":
            embedding = native.get("WSI_feature", embedding)
        if embedding is not None and embedding.ndim == 3 and embedding.shape[1] == 1:
            embedding = embedding.squeeze(1)
        return output_from_logits(
            logits, embedding=embedding, attention=attention, aux=aux, native=native
        )


class MHIMAdapter(MILAdapter):
    """Native MHIM student, mask/distillation path, and checkpointed EMA teacher."""

    def __init__(
        self,
        name,
        input_dim,
        num_classes,
        *,
        teacher_momentum=0.999,
        teacher_state_dict=None,
        **options,
    ):
        super().__init__()
        self.input_dim = _positive(input_dim, "input_dim")
        _positive(num_classes, "num_classes")
        self.version = name
        if not 0 <= teacher_momentum <= 1:
            raise ValueError("teacher_momentum must be in [0,1]")
        self.teacher_momentum = teacher_momentum
        # A positive high-attention fraction requires a nonzero sampling ratio.
        options.setdefault("mask_ratio_hr", 1.0)
        for key in (
            "mask_ratio",
            "mask_ratio_h",
            "mask_ratio_l",
            "mask_ratio_hr",
            "merge_ratio",
            "merge_mm",
        ):
            if key in options and not 0 <= options[key] <= 1:
                raise ValueError(f"{key} must be in [0,1]")
        if options["mask_ratio_hr"] == 0:
            raise ValueError("mask_ratio_hr must be positive")
        slug = "mhim_mil_v1" if name == "mhim_v1" else "mhim_mil"
        cls = _native(slug, "modules.mhim").MHIM
        if name == "mhim_v1":
            self.native = cls(n_classes=num_classes, **options)
            self.native.patch_to_emb[0] = nn.Linear(input_dim, options.get("mlp_dim", 512))
        else:
            self.native = cls(input_dim=input_dim, n_classes=num_classes, **options)
        self.teacher = deepcopy(self.native)
        if name == "mhim_v2":
            # The upstream training coordinator also keeps teacher merging off.
            self.teacher.merge_test = False
        if teacher_state_dict is not None:
            self.teacher.load_state_dict(teacher_state_dict, strict=True)
        self.teacher.requires_grad_(False).eval()
        self.register_buffer("teacher_updates", torch.zeros((), dtype=torch.long))
        self.metadata = {
            "profile": name,
            "repository": slug,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "one_bag_per_call": True,
            "native_entrypoint": "modules.mhim.MHIM",
            "teacher_momentum": teacher_momentum,
            "options": options,
            "adaptations": [
                "frozen_checkpointed_teacher",
                "explicit_teacher_ema_hook",
                "at_least_one_retained_patch",
                "integer_mask_indices",
            ],
            "training_note": "Add aux.distillation_loss to classification loss; call update_teacher after optimizer.step; native v2 merge EMA remains active",
        }
        if name == "mhim_v2":
            self.metadata["adaptations"] += [
                "identity_indices_for_zero_mask",
                "argument_compatible_disabled_merge",
                "registered_frozen_merge_query",
                "nonempty_merge_input",
            ]
        if self.native.baseline == "dsmil":
            self.metadata["adaptations"] += [
                "dual_dsmil_heads_retained",
                "dsmil_configured_class_count",
                "dsmil_eval_unpack",
            ]
            self.metadata["training_note"] += (
                "; DSMIL returns bag logits plus aux.max_instance_logits for its two-head classification objective"
            )

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    @torch.no_grad()
    def reset_teacher(self):
        self.teacher.load_state_dict(self.native.state_dict())
        self.teacher_updates.zero_()

    @torch.no_grad()
    def update_teacher(self, momentum=None):
        momentum = self.teacher_momentum if momentum is None else momentum
        if not 0 <= momentum <= 1:
            raise ValueError("Teacher momentum must be in [0,1]")
        source = self.native.state_dict()
        seen = set()
        for key, target in self.teacher.state_dict().items():
            identity = (target.data_ptr(), target.storage_offset(), tuple(target.shape))
            if identity in seen:
                continue
            seen.add(identity)
            current = source[key]
            if target.is_floating_point() or target.is_complex():
                target.mul_(momentum).add_(current, alpha=1 - momentum)
            else:
                target.copy_(current)
        self.teacher_updates.add_(1)

    def after_optimizer_step(self, momentum=None):
        self.update_teacher(momentum)

    def forward(self, inputs, *, iteration=0):
        bag = as_bag(inputs, self.input_dim, batched=True).float()
        head = (
            self.native.online_encoder.b_classifier.fcc
            if self.native.baseline == "dsmil"
            else self.native.predictor
        )
        if self.training:
            teacher_feature, attention = self.teacher.forward_teacher(bag)
            with _capture(head) as embeddings:
                native = self.native(
                    bag, attn=attention, teacher_cls_feat=teacher_feature, i=iteration
                )
            logits, distillation, count, kept = native
            aux = {
                "distillation_loss": distillation,
                "original_patch_count": count,
                "retained_patch_count": kept,
                "teacher_embedding": teacher_feature,
                "teacher_updates": self.teacher_updates.clone(),
            }
        else:
            with _capture(head) as embeddings:
                native = self.native.forward_test(bag, return_attn=True)
            logits, attention = native
            aux = {}
        if self.native.baseline == "dsmil":
            logits, max_instance_logits = logits
            aux["max_instance_logits"] = max_instance_logits
            aux["class_embeddings"] = embeddings[-1]

            def probability(value):
                return value.sigmoid() if value.shape[-1] == 1 else value.softmax(-1)

            aux["ensemble_probabilities"] = (
                probability(logits) + probability(max_instance_logits)
            ) / 2
        return output_from_logits(
            logits,
            embedding=embeddings[-1] if embeddings else None,
            attention=attention,
            aux=aux,
            native=native,
        )


def build(name, input_dim=None, num_classes=None, **options):
    input_dim = (768 if name == "ilra" else 1024) if input_dim is None else input_dim
    num_classes = 2 if num_classes is None else num_classes
    cls = MHIMAdapter if name in {"mhim_v1", "mhim_v2"} else NativeTransformer
    return cls(name, input_dim, num_classes, **options).cpu().float()
