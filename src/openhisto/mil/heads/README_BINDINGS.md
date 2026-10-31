The attachment layer exposes one fixed vector per slide while retaining the
complete original MIL model and its outputs. It supports all 34 profiles in
`openhisto.mil.models.code_bases`. Dimensions are inspected from constructed modules;
binding inspection never runs a training forward.

```python
from openhisto.mil.models.code_bases import create_model
from openhisto.mil.classification import create_neural_head
from openhisto.mil.heads import attach_head, representation_dimension

backbone = create_model("abmil", input_dim=1024, num_classes=2)
head = create_neural_head("mlp", representation_dimension(backbone), 3)
model = attach_head(backbone, head, train_backbone=True)
result = model(patch_features)
# result.head: ClassificationOutput or SurvivalOutput from the new head
# result.representation: [1, declared_dimension]
# result.mil: the original, unmodified MILOutput, including native aux losses
# result.representation_spec: JSON-safe dimensions, axes, policy and provenance
```

| Profiles | Representation policy |
| --- | --- |
| ABMIL, gated ABMIL, DTFD, ACMIL, ASMIL, CLAM-SB, VarPool, PANTHER | Existing native slide embedding, including the native second-tier or prototype representation where applicable |
| DSMIL, CLAM-MB, MHIM with DSMIL baseline | Flatten the ordered class-by-feature axes; retain every class embedding and record native classifier index order |
| TransMIL, ILRA, RRT, DGR, CAPRMIL, Lin-MIL, MambaMIL, MHIM attention/self-attention | Existing native pooled embedding |
| nnMIL | Native pooled feature averaged across evaluation windows; a downstream nonlinear head differs from averaging native per-window predictions, and the spec records this adaptation |
| WiKG | Normalized graph readout before `fc` |
| ResGAT | Mean graph readout before the classifier MLP (`fc1`) |
| TAD | Causal graph readout before `clf.pred_layer`, excluding its separate trivial stream |
| Patch-GCN | Native gated pooled `path_rho` vector before its survival classifier |
| SlideGraph+ | Concatenate the configured native feature pooling of each layer; this is recorded as an adaptation from its native sum of layer prediction scores |
| CS-MIL | Native paired-resolution fusion and bag-attention vector before `fc6` |
| ZoomMIL | Native sum of the three magnification readouts |
| HAG-MIL | Native finest selected hierarchy stage before `fc5` |
| GFR-MIL | Native fine class-attention readout before `classfier` |
| H2-MIL | Native sum of the two hierarchy-pooling readouts before `lin1` |
| HIPT | Native global hierarchy readout before `classifier`, for LGP, GP and None variants |
| Mean, max, mean/std baselines | Their native statistical feature pooling |

Every graph or hierarchy still requires its native caller-prepared inputs.
Bindings do not create graph edges, pair magnifications, infer parent mappings,
average class axes or infer a patient split. Graph capture hooks exist only
during the call and are removed even if the native forward raises.

`train_backbone=False` freezes its parameters, keeps it in evaluation mode even
after `model.train()`, and computes its representation under `torch.no_grad()`.
The new head remains trainable. `train_backbone=True` preserves pre-existing
parameter freeze flags, including native frozen teachers and prototypes.

Original classifier, auxiliary modules, teachers and EMA buffers remain in the
wrapper's `state_dict` under `backbone`. The new head is under `head`. A loss
on the new head does not train an original classifier which it bypasses.
Native auxiliary objectives remain available through `result.mil`; hard
selection stages still require their native stage objectives. Statistical
pooling and frozen-prototype representations can have no trainable parameters
before the attached head.

Call `model.after_optimizer_step()` to forward a backbone's explicit teacher or
anchor update hook; `update_teacher` and `update_anchor` are also forwarded.
Frozen backbones skip these updates. No implicit optimizer or loss weighting is
introduced. The representation spec is copied into a head's dictionary
`metadata` for persistence; its configuration and input shape are unchanged.

`extract_slide_embeddings(backbone, inputs_iterable, ...)` returns a CPU tensor
`[slides,D]` and its representation spec. Input entries are native inputs or
`(inputs, forward_kwargs)` pairs, preserving iterable order. It uses eval mode
and no gradients, then restores the exact prior training flags, including
mixed frozen-teacher modes. Empty iterables return `[0,D]`. It does not infer
sample IDs or labels, and it collects the requested slide matrix in memory.
The returned spec includes the actual backbone tensor-state SHA256. The state
is hashed before and after extraction; a change fails the export. Ordinary
binding inspection and training forwards do not hash model weights.
