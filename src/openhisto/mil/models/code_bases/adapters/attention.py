"""Feature-bag wrappers retaining native attention, CLAM, and statistical MIL.

Native sources are imported in isolated namespaces. Prediction wrappers expose
training-relevant auxiliary tensors; their existence does not replace upstream
multi-stage optimizers, sampling schedules, or teacher update procedures.
"""

from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from openhisto._common import json_safe

from ..base import MILAdapter, as_bag, output_from_logits
from ..compat.attention_imports import replacements
from ..compat.loader import load_native


def _native(slug, module, *, roots=("",)):
    return load_native(slug, module, roots=roots, replacements=replacements(slug))


def _positive(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


class _Adapter(MILAdapter):
    def __init__(self, name, slug, input_dim, num_classes):
        super().__init__()
        self.input_dim = _positive(input_dim, "input_dim")
        self.num_classes = _positive(num_classes, "num_classes")
        self.metadata = {
            "profile": name,
            "repository": slug,
            "input_dim": input_dim,
            "num_classes": num_classes,
            "one_bag_per_call": True,
            "adaptations": [],
        }

    def _bag(self, inputs, *, batched=False):
        bag = as_bag(inputs, self.input_dim, batched=batched)
        if bag.device.type != "cpu":
            raise ValueError("These initial MIL adapters require CPU feature tensors")
        return bag.float()


@contextmanager
def _inputs(module):
    captured = []
    handle = module.register_forward_pre_hook(lambda model, args: captured.append(args[0]))
    try:
        yield captured
    finally:
        handle.remove()


class ABMILAdapter(_Adapter):
    def __init__(self, name, input_dim, num_classes):
        super().__init__(name, "attention_deep_mil", input_dim, num_classes)
        module = _native("attention_deep_mil", "model")
        self.native = module.GatedAttention() if name == "gated_abmil" else module.Attention()
        self.projection = (
            nn.Identity()
            if input_dim == 500
            else nn.Sequential(nn.Linear(input_dim, 500), nn.ReLU())
        )
        self.binary = num_classes in {1, 2}
        if not self.binary:
            self.native.classifier = nn.Sequential(nn.Linear(500, num_classes))
        self.metadata.update(
            native_entrypoint="model.GatedAttention"
            if name == "gated_abmil"
            else "model.Attention",
            native_hidden_dim=500,
            native_binary_head=self.binary,
        )
        self.metadata["adaptations"] = [
            "bypass_image_cnn",
            "identity_projection" if input_dim == 500 else "learned_feature_projection",
        ]
        if not self.binary:
            self.metadata["adaptations"].append("multiclass_classifier_head")

    def forward(self, inputs):
        features = self.projection(self._bag(inputs))
        if hasattr(self.native, "attention_V"):
            raw = self.native.attention_w(
                self.native.attention_V(features) * self.native.attention_U(features)
            ).T
        else:
            raw = self.native.attention(features).T
        attention = raw.softmax(dim=1)
        embedding = attention @ features
        score = self.native.classifier[0](embedding)
        if self.binary:
            probability = score.sigmoid()
            native = (probability, (probability >= 0.5).float(), attention)
            logits = (
                torch.cat([torch.zeros_like(score), score], dim=-1)
                if self.num_classes == 2
                else score
            )
        else:
            logits = score
            native = (logits, logits.softmax(-1), attention)
        return output_from_logits(
            logits,
            embedding=embedding,
            attention=attention,
            aux={"raw_attention": raw},
            native=native,
        )


class DSMILAdapter(_Adapter):
    def __init__(self, input_dim, num_classes, *, dropout=0.0, nonlinear=True, passing_v=False):
        super().__init__("dsmil", "dsmil", input_dim, num_classes)
        module = _native("dsmil", "dsmil")
        self.native = module.MILNet(
            module.FCLayer(input_dim, num_classes),
            module.BClassifier(
                input_dim, num_classes, dropout_v=dropout, nonlinear=nonlinear, passing_v=passing_v
            ),
        )
        self.metadata.update(
            native_entrypoint="dsmil.MILNet",
            probability_note="Native inference ensemble: mean of independent bag/max-instance sigmoid scores; logits retain the bag stream",
            probability_semantics="independent_per_class_sigmoid_ensemble",
        )

    def forward(self, inputs):
        native = self.native(self._bag(inputs))
        instances, logits, attention, embedding = native
        maximum = instances.max(dim=0, keepdim=True).values
        ensemble = (logits.sigmoid() + maximum.sigmoid()) / 2
        output = output_from_logits(
            logits,
            embedding=embedding,
            attention=attention,
            aux={
                "instance_logits": instances,
                "max_instance_logits": maximum,
                "native_bag_probabilities": logits.sigmoid(),
                "native_ensemble_probabilities": ensemble,
            },
            native=native,
        )
        output.probabilities = ensemble
        return output


class DTFDAdapter(_Adapter):
    def __init__(
        self,
        input_dim,
        num_classes,
        *,
        hidden_dim=512,
        num_groups=4,
        distill="AFS",
        instances_per_group=1,
        partition_seed=0,
        dropout=0.0,
    ):
        super().__init__("dtfd_mil", "dtfd_mil", input_dim, num_classes)
        hidden_dim = _positive(hidden_dim, "hidden_dim")
        self.num_groups = _positive(num_groups, "num_groups")
        self.instances_per_group = _positive(instances_per_group, "instances_per_group")
        if distill not in {"AFS", "MaxS", "MaxMinS"}:
            raise ValueError("distill must be AFS, MaxS, or MaxMinS")
        if isinstance(partition_seed, bool) or not isinstance(partition_seed, int):
            raise ValueError("partition_seed must be an integer")
        self.distill, self.partition_seed = distill, partition_seed
        attention = _native("dtfd_mil", "Model.Attention")
        network = _native("dtfd_mil", "Model.network")
        self.native = nn.ModuleDict(
            {
                "dim_reduction": network.DimReduction(input_dim, hidden_dim),
                "attention": attention.Attention_Gated(hidden_dim),
                "classifier": network.Classifier_1fc(hidden_dim, num_classes, dropout),
                "att_classifier": attention.Attention_with_Classifier(
                    L=hidden_dim, num_cls=num_classes, droprate=dropout
                ),
            }
        )
        self.metadata.update(
            native_entrypoint="Model.Attention + Model.network",
            distill=distill,
            num_groups=num_groups,
            partition_seed=partition_seed,
            adaptations=["native_two_tier_coordinator", "deterministic_seeded_eval_partition"],
        )

    def forward(self, inputs):
        bag = self._bag(inputs)
        if len(bag) < self.num_groups:
            raise ValueError("DTFD requires at least num_groups instances")
        generator = None if self.training else torch.Generator().manual_seed(self.partition_seed)
        groups = torch.tensor_split(torch.randperm(len(bag), generator=generator), self.num_groups)
        features = self.native["dim_reduction"](bag)
        raw = self.native["attention"](features, isNorm=False)[0]
        distilled, predictions, weights = [], [], []
        for indices in groups:
            current = features[indices]
            current_weights = raw[indices].softmax(0)
            weighted = current * current_weights[:, None]
            pooled = weighted.sum(0, keepdim=True)
            predictions.append(self.native["classifier"](pooled))
            weights.append(current_weights)
            if self.distill == "AFS":
                distilled.append(pooled)
            else:
                scores = (weighted @ self.native["classifier"].fc.weight.T).softmax(-1)[:, -1]
                order = scores.argsort(descending=True)
                count = min(self.instances_per_group, len(order))
                chosen = (
                    order[:count]
                    if self.distill == "MaxS"
                    else torch.cat([order[:count], order[-count:]])
                )
                distilled.append(current[chosen])
        distilled = torch.cat(distilled)
        tier2 = self.native["att_classifier"]
        with _inputs(tier2.classifier) as embedding:
            logits = tier2(distilled)
        return output_from_logits(
            logits,
            embedding=embedding[-1],
            attention=weights,
            aux={
                "pseudo_bag_logits": torch.cat(predictions),
                "pseudo_bag_indices": groups,
                "distilled_features": distilled,
                "first_tier_raw_attention": raw,
            },
            native={"first_tier_logits": torch.cat(predictions), "second_tier_logits": logits},
        )


class ACMILAdapter(_Adapter):
    def __init__(
        self,
        input_dim,
        num_classes,
        *,
        hidden_dim=512,
        n_token=5,
        n_masked_patch=10,
        mask_drop=0.6,
        dropout=0.0,
    ):
        super().__init__("acmil", "acmil", input_dim, num_classes)
        hidden_dim = _positive(hidden_dim, "hidden_dim")
        n_token = _positive(n_token, "n_token")
        if not 0 <= mask_drop <= 1 or n_masked_patch < 0:
            raise ValueError("Invalid ACMIL masking settings")
        module = _native("acmil", "architecture.transformer")
        conf = SimpleNamespace(
            D_feat=input_dim, D_inner=hidden_dim, n_class=num_classes, n_token=n_token
        )
        self.native = module.ACMIL_GA(
            conf,
            n_token=n_token,
            n_masked_patch=n_masked_patch,
            mask_drop=mask_drop,
            droprate=dropout,
        )
        self.metadata.update(native_entrypoint="architecture.transformer.ACMIL_GA", n_token=n_token)

    def forward(self, inputs):
        with _inputs(self.native.Slide_classifier) as embeddings:
            native = self.native(self._bag(inputs, batched=True))
        branches, logits, raw = native
        return output_from_logits(
            logits,
            embedding=embeddings[-1],
            attention=raw.softmax(-1),
            aux={"branch_logits": branches, "raw_attention": raw},
            native=native,
        )


class ASMILAdapter(_Adapter):
    def __init__(
        self, input_dim, num_classes, *, hidden_dim=512, n_token=5, n_drop=1, temperature=1.0
    ):
        super().__init__("asmil", "asmil", input_dim, num_classes)
        hidden_dim = _positive(hidden_dim, "hidden_dim")
        n_token = _positive(n_token, "n_token")
        if (
            hidden_dim % 8
            or not isinstance(n_drop, int)
            or not 0 <= n_drop < n_token
            or temperature <= 0
        ):
            raise ValueError(
                "ASMIL requires hidden_dim divisible by eight, 0 <= n_drop < n_token, positive temperature"
            )
        module = _native("asmil", "architecture.mytransformer")
        self.sigmax = module.sigmax
        conf = SimpleNamespace(D_feat=input_dim, D_inner=hidden_dim, n_class=num_classes)
        self.native = module.ACMIL_MYMHA(conf, n_token=n_token, n_drop=n_drop)
        self.anchor = module.ACMIL_MYMHA_EMA(conf, n_token=n_token)
        self.anchor.requires_grad_(False)
        self.temperature = float(temperature)
        self.metadata.update(
            native_entrypoint="architecture.mytransformer.ACMIL_MYMHA",
            n_token=n_token,
            adaptations=[
                "unused_missing_import_removed",
                "explicit_matching_finite_state_ema_helper",
            ],
            training_note="Call update_anchor after optimizer steps; forward does not update anchor",
            attention_semantics="Student patch attention uses native softmax; anchor_attention uses native normalized sigmoid (sigmax)",
        )

    @torch.no_grad()
    def update_anchor(self, decay=0.999):
        if not 0 <= decay <= 1:
            raise ValueError("EMA decay must be in [0,1]")
        student = self.native.state_dict()
        for key, value in self.anchor.state_dict().items():
            if (
                key in student
                and value.shape == student[key].shape
                and torch.isfinite(student[key]).all()
            ):
                value.copy_(value * decay + student[key] * (1 - decay))

    def forward(self, inputs):
        bag = self._bag(inputs, batched=True)
        with _inputs(self.native.Slide_classifier) as embeddings:
            native = self.native(bag)
        branches, logits, raw = native
        aux = {"branch_logits": branches, "raw_attention": raw}
        if self.training:
            with torch.no_grad():
                anchor = self.anchor(bag)
                target = self.sigmax(anchor / self.temperature, dim=-1)
            aux.update(anchor_attention=target, anchor_loss=-(target * raw.log_softmax(-1)).mean())
        return output_from_logits(
            logits, embedding=embeddings[-1], attention=raw.softmax(-1), aux=aux, native=native
        )


class CLAMAdapter(_Adapter):
    def __init__(
        self, name, input_dim, num_classes, *, k_sample=8, dropout=0.0, subtyping=False, gate=True
    ):
        super().__init__(name, "clam", input_dim, num_classes)
        if num_classes < 2:
            raise ValueError("Native CLAM requires at least two classes")
        module = _native("clam", "models.model_clam")
        cls = module.CLAM_SB if name == "clam_sb" else module.CLAM_MB
        self.native = cls(
            embed_dim=input_dim,
            n_classes=num_classes,
            k_sample=_positive(k_sample, "k_sample"),
            dropout=dropout,
            subtyping=subtyping,
            gate=gate,
        )
        self.metadata.update(native_entrypoint="models.model_clam." + cls.__name__)

    def forward(self, inputs, *, label=None, instance_eval=False):
        bag = self._bag(inputs)
        if instance_eval and (label is None or len(bag) < self.native.k_sample):
            raise ValueError(
                "CLAM instance evaluation requires a label and at least k_sample patches"
            )
        native = self.native(bag, label=label, instance_eval=instance_eval, return_features=True)
        logits, probabilities, predictions, raw, results = native
        output = output_from_logits(
            logits,
            embedding=results.get("features"),
            attention=raw.softmax(-1),
            aux={**results, "predictions": predictions, "raw_attention": raw},
            native=native,
        )
        output.probabilities = probabilities
        return output


class VarPoolAdapter(_Adapter):
    def __init__(
        self,
        input_dim,
        num_classes,
        *,
        n_var_pools=100,
        separate_attn=False,
        gated=True,
        act_func="sqrt",
    ):
        super().__init__("varpool", "varpool", input_dim, num_classes)
        module = _native("varpool", "var_pool.nn.arch.VarPool")
        self.native = module.AttnMeanAndVarPoolMIL(
            encoder_dim=input_dim,
            head=num_classes,
            n_var_pools=_positive(n_var_pools, "n_var_pools"),
            separate_attn=separate_attn,
            gated=gated,
            act_func=act_func,
        )
        self.metadata.update(
            native_entrypoint="var_pool.nn.arch.VarPool.AttnMeanAndVarPoolMIL",
            n_var_pools=n_var_pools,
            gated_argument=gated,
        )

    def forward(self, inputs):
        bag = self._bag(inputs, batched=True)
        captured = []
        hook = self.native.enc_and_attend.register_forward_hook(
            lambda model, args, output: captured.append(output[0])
        )
        try:
            with _inputs(self.native.head) as embeddings:
                native = self.native(bag)
        finally:
            hook.remove()
        raw = captured[-1]
        attention = (
            [item.reshape(1, len(bag[0]), 1).softmax(1) for item in raw]
            if isinstance(raw, list)
            else raw.reshape(1, len(bag[0]), 1).softmax(1)
        )
        return output_from_logits(
            native, embedding=embeddings[-1], attention=attention, native=native
        )


class PANTHERAdapter(_Adapter):
    def __init__(
        self,
        input_dim,
        num_classes,
        *,
        prototype_path=None,
        n_proto=None,
        em_iter=3,
        tau=10.0,
        out_type="allcat",
        ot_eps=0.1,
        fix_proto=True,
    ):
        super().__init__("panther", "panther", input_dim, num_classes)
        if prototype_path is None:
            raise ValueError("PANTHER requires an explicit training-fitted prototype_path (.npy)")
        path = Path(prototype_path)
        if path.suffix != ".npy":
            raise ValueError("Use a numeric .npy prototype matrix; pickle loading is not enabled")
        prototypes = np.load(path, allow_pickle=False)
        if (
            prototypes.ndim != 2
            or prototypes.shape[1] != input_dim
            or prototypes.dtype.kind not in "fi"
            or not np.isfinite(prototypes).all()
        ):
            raise ValueError("Prototype matrix must be finite numeric [P,input_dim]")
        n_proto = _positive(n_proto or len(prototypes), "n_proto")
        if len(prototypes) != n_proto:
            raise ValueError("n_proto does not match the prototype matrix")
        if out_type not in {"allcat", "weight_all_cat", "weight_avg_all", "weight_avg_mean"}:
            raise ValueError("Unsupported PANTHER representation type")
        module = _native("panther", "mil_models.PANTHER.layers", roots=("src",))
        self.native = module.PANTHERBase(
            input_dim,
            p=n_proto,
            L=_positive(em_iter, "em_iter"),
            tau=tau,
            out=out_type,
            ot_eps=ot_eps,
            load_proto=True,
            proto_path=str(path),
            fix_proto=fix_proto,
        ).float()
        self.classifier = nn.Linear(self.native.outdim, num_classes)
        self.metadata.update(
            native_entrypoint="mil_models.PANTHER.layers.PANTHERBase",
            n_proto=n_proto,
            prototype_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            out_type=out_type,
            embedding_dim=self.native.outdim,
            adaptations=["lazy_unrelated_package_imports", "added_downstream_linear_classifier"],
        )

    def representation(self, inputs):
        embedding, assignments = self.native(self._bag(inputs, batched=True))
        return {"repr": embedding, "qq": assignments}

    def forward(self, inputs):
        native = self.representation(inputs)
        logits = self.classifier(native["repr"])
        return output_from_logits(
            logits,
            embedding=native["repr"],
            attention=native["qq"],
            aux={"prototype_assignments": native["qq"]},
            native=native,
        )


def build(name, input_dim=None, num_classes=None, **options):
    input_dim = 1024 if input_dim is None else input_dim
    num_classes = 2 if num_classes is None else num_classes
    if name in {"abmil", "gated_abmil"}:
        if options:
            raise TypeError("ABMIL profiles take only input_dim and num_classes")
        cls, arguments = ABMILAdapter, (name, input_dim, num_classes)
    elif name in {"clam_sb", "clam_mb"}:
        cls, arguments = CLAMAdapter, (name, input_dim, num_classes)
    else:
        cls = {
            "dsmil": DSMILAdapter,
            "dtfd_mil": DTFDAdapter,
            "acmil": ACMILAdapter,
            "asmil": ASMILAdapter,
            "varpool": VarPoolAdapter,
            "panther": PANTHERAdapter,
        }.get(name)
        if cls is None:
            raise ValueError("Unknown attention/statistical profile")
        arguments = (input_dim, num_classes)
    configuration = inspect.signature(cls).bind(*arguments, **options)
    configuration.apply_defaults()
    model = cls(*arguments, **options)
    # Include effective defaults as well as explicit arguments: tensor state
    # alone cannot distinguish operations such as VarPool's sqrt/log readout.
    model.options = json_safe(dict(configuration.arguments))
    model.options.pop("name", None)
    if name == "panther":
        model.options["n_proto"] = model.metadata["n_proto"]
    model.metadata["options"] = model.options
    return model.cpu().float()
