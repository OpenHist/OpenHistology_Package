# MIL code bases and CPU adapters

This directory contains **34 model profiles**, backed by **28 upstream repositories in 29 pinned working trees**, plus three original statistical-pooling baselines. MHIM v1 and v2 use separate branches/checkouts; ABMIL/gated ABMIL and CLAM-SB/MB share their respective sources.

The common API runs one feature bag, graph or prepared hierarchy at a time on **CPU float32**. It retains native model components, outputs and training auxiliaries. [COVERAGE.md](COVERAGE.md) reports actual synthetic CPU execution; [coverage.json](coverage.json) records the tested environment, commits and implementation fingerprint.

## Install and inspect

Use Python 3.12 in a separate environment. From the repository root:

```bash
python3.12 -m venv .venv-mil
source .venv-mil/bin/activate
python -m pip install -e ".[mil-backbones]"
python -m openhisto.mil.models.code_bases list
python -m openhisto.mil.models.code_bases info mhim_v2
```

The `mil` extra installs the original pooling baselines. Each family has its own extra: `mil-attention`, `mil-graph-multiscale` and `mil-transformers`; `mil-backbones` installs all three. The combined CPU set was tested in a fresh Python 3.12 environment with Torch 2.6.0, torchvision 0.21.0, NumPy 2.2.6 and PyG 2.6.1. `requirements-tested.lock.txt` captures the full tested environment. Historical upstream environment files remain preserved as references; they are not installation instructions for these adapters.

Listing and inspecting models does not import Torch/upstream model code, request credentials or download artifacts. Construction imports only the selected family/model. The public factory currently accepts CPU devices only; GPU parity and performance have not been validated.

## Feature-bag usage

```python
import torch
from openhisto.mil.models.code_bases import create_model, list_models, model_info

model = create_model("clam_sb", input_dim=768, num_classes=3, k_sample=8)
patch_features = torch.randn(32, 768)  # [patches, feature dimensions]
model.eval()
with torch.no_grad():
    output = model(patch_features)
print(output.logits.shape)  # [1, 3]
```

`create_model(name, input_dim=None, num_classes=None, device="cpu", **model_options)` returns a PyTorch module. Omitted dimensions/class counts use each native profile's defaults; constraints remain model-specific. Most feature models accept `[N,D]` or `[1,N,D]`. Empty bags and invalid feature dimensions are rejected. Configurable native options pass through explicitly; different models need different bag sizes and hyperparameters.

`MILOutput` has `logits`, `probabilities`, `embedding`, `attention`, `aux` and `native_output`. Unavailable fields are `None`; native tuples/dictionaries and gradients are preserved. Attention shapes and meanings remain model-specific, with raw scores retained where exposed. Survival outputs are hazards/survival/risk in `aux`, rather than class probabilities. DSMIL uses its native sigmoid ensemble; CS-MIL retains independent sigmoid heads.

Use normal `.train()`, `.eval()`, `.parameters()` and `.state_dict()` interfaces. Save the entire adapter state, including teachers, anchors and projections, with its metadata/configuration. Saving only the student omits method-specific state. Local checkpoint loaders use explicit paths, CPU mapping and strict state loading; missing or incompatible requested weights fail rather than silently initializing a pretrained model at random.

You can pass features read through the existing feature encoder manually:

```python
from openhisto.feature_encoder import FeatureReader

with FeatureReader("slide.h5ad") as reader:
    features = reader.features("phikon", mag=20)  # select run_key if ambiguous
bag = torch.from_numpy(features).float()
output = create_model("abmil", input_dim=bag.shape[1], num_classes=2)(bag)
```

Automatic `.h5ad` loading and feature/run selection are deferred. Extraction magnification, patch order and geometry remain caller responsibilities.

## Profiles and input contracts

| Family | Profile IDs | Inputs / requirements |
|---|---|---|
| Attention | `abmil`, `gated_abmil`, `dsmil`, `dtfd_mil`, `acmil`, `asmil`, `clam_sb`, `clam_mb` | Feature bags; DTFD needs at least `num_groups` patches; CLAM instance evaluation needs labels and sufficient patches |
| Statistical | `mean_mil`, `max_mil`, `mean_std_mil`, `varpool`, `panther` | Feature bags; PANTHER additionally requires explicit `.npy` prototypes |
| Transformer/framework | `transmil`, `ilra`, `rrt_mil`, `dgr_mil`, `caprmil`, `lin_mil`, `mamba_mil`, `nnmil`, `mhim_v1`, `mhim_v2` | Feature bags, preserving native order/sampling/masking; Mamba profile is classification-only |
| Learned graph | `wikg` | Feature bag; at least `topk` instances, no coordinates required by the released network |
| Prepared graph | `tad_graph`, `resgat`, `patch_gcn`, `slidegraph_plus`, `h2_mil` | Caller-prepared PyG objects, valid feature-aligned edges and one graph per call |
| Multiscale/hierarchy | `cs_mil`, `zoommil`, `hag_mil`, `gfr_mil`, `hipt` | Native region correspondence, hierarchy or grouping described below |

Aliases include `MeanMIL`, `MaxMIL`, `MeanStdMIL`, `ilra_mil`, `r2t_mil`, `h2mil` and `mhim_mil` (v1). Canonical names are returned by `list_models()`.

For prepared graphs, `x` is `[N,D]`, edges are long `[2,E]`, and `batch` is omitted or a long all-zero vector. Graph construction, spatial clustering and inference of neighbourhoods are deferred. Specific contracts:

ResGAT and Patch-GCN also accept edgeless graphs (`edge_index` shaped `[2,0]`), including a single node; their native self/residual paths remain available. Other graph profiles retain their model-specific nonempty edge requirements.

- **TAD-Graph:** normalized distance/angle `edge_attr[E,2]` in `[0,1]`; preserves learned causal/trivial graphs and information loss. Native `task="survival_prediction"` exposes survival outputs.
- **Patch-GCN:** native discrete-survival head; `num_classes` controls time bins. Supply `edge_latent` for latent-edge aggregation when selected.
- **SlideGraph+:** prepared clustered-region graph. The profile uses the released EdgeConv/max-neighbour aggregation configuration; GIN is an explicit alternative.
- **H²-MIL:** supply `edge_index_tree_8nb`, long `node_type` values 0/1/2, `node_tree` and normalized `x_y_index[N,2]`. Nodes are ordered root/coarse/fine, with exactly one root and at least one fine child per coarse parent. Native hierarchical pooling and auxiliary outputs remain available.
- **CS-MIL:** `{"clusters": [tensor,...], "mask": tensor}`; each cluster is `[P_i,D,3,1]`, with three corresponding scales; mask is optional `[1,sum(P_i)]`.
- **ZoomMIL:** `{"levels": (low,middle,high)}` in contiguous parent-major order with integer child-count ratios.
- **HAG-MIL:** the same three-level structure, requiring four contiguous children per parent at each level. All three native IAT models and their selection cascade are retained.
- **GFR-MIL:** dictionary containing low `features`, `coords`, `high_features`, `high_coords`, and long `region_labels` assigning each high patch to a low-region row. Native coordinate quantization remains 448/3584 pixels; use its required coordinate convention. `task="survival"` retains native survival computation.
- **HIPT:** default `variant="lgp"` requires raster-ordered `[regions,256,384]` native fine-patch embeddings. `variant="gp"` takes `[regions,192]` region features; `variant="none"` is the explicitly named no-transformer ablation with `[regions,patches,D]`. Generic embeddings cannot silently replace these native stages.

RRT's feature-bag API supports `pos="none"`, `"peg"`, and `"ppeg"`. The pinned upstream `"sincos"` ablation requires four-dimensional region geometry that `[N,D]` feature bags do not provide, so construction rejects it with an actionable error. No row-to-grid mapping is inferred.

Synthetic shape examples for every profile are executable `make_case` fixtures in the three family test modules. They use fabricated values and do not infer graph/hierarchy structure from patient data.

## Training components and explicit adaptations

Shared trainers, data splits, cohort processing and evaluation pipelines are deferred. Native training scripts are retained. The adapters expose the components a future trainer must use:

- DSMIL instance/bag streams, DTFD pseudo-bag logits and distillation variants, ACMIL branch logits/masking, CLAM instance losses and DGR lesion/center tensors remain accessible.
- ASMIL exposes its frozen anchor and `update_anchor(decay)`; call this explicitly after optimizer steps. MHIM keeps its teacher in the state dictionary, returns `aux["distillation_loss"]`, and exposes `update_teacher()` / `after_optimizer_step()`. Combine native auxiliary objectives with the appropriate slide loss and retain the upstream schedules.
- nnMIL preserves training feature sampling and evaluation window ensembling. Its automatic dataset/split planner is not invoked.
- Original ABMIL's grayscale CNN is bypassed. A declared learned `D→500` ReLU projection is used when needed, followed by native attention/classifier components. Two-class logits preserve native binary sigmoid probabilities; larger class heads are explicit adaptations.
- PANTHER uses its native prototype/EM representation and an identified downstream linear classifier. Supply training-fitted finite numeric prototypes through `prototype_path=`; prototypes are never fitted from an evaluation bag implicitly.
- Mean/Max/MeanStd baselines perform featurewise pooling. MeanStd uses population standard deviation (`correction=0`) so singleton bags remain finite.
- MambaMIL uses exact differentiable Torch convolution/scan operators and preserves native SR padding/transposition. This CPU path is slower than fused CUDA and does not truncate bags. Lin-MIL's default linear attention is validated; optional CUDA attention ablations are outside this CPU profile.

Review `compat/*_repairs.json` for changes and limitations. Repairs include H²-MIL normalization/device corrections, HIPT width/checkpoint fixes, modern PyG compatibility, ZoomMIL backward arity, ASMIL EMA updates, nnMIL singleton uncertainty and MHIM boundary/state repairs. H²-MIL's repaired normalization state names differ from historical checkpoints; incompatible checkpoints require an explicit migration and are not partially loaded.

## Sources, provenance and verification

The `upstream/` directories publish verified **source-only vendored snapshots** of the recorded commits. Local acquisition also retains sparse Git checkouts, whose `.git` metadata is not part of the outer package repository. A fresh package clone already contains the sources and needs no submodule initialization. Python/native source, scripts, configuration, requirements, documentation and licenses are included; slide data, clinical/split tables, notebooks with embedded outputs, weights and LFS payloads are excluded.

Source selection can omit an explicitly named, unused upstream reference without changing the source commit. `source_selection.py` and each affected record's `excluded_files` document those paths and reasons; acquisition, restoration and snapshot inventories use the same selection. The incomplete CS-MIL `models/swav.py` reference is excluded so published Python sources compile, while the supported CS-MIL module and all included source bytes remain unchanged.

`compat/sources/*.json` records each repository URL, branch, exact SHA, retrieval time, included license paths and checkout. [SOURCE_CATALOG.json](SOURCE_CATALOG.json) links all profiles to their sources. Preserve each upstream's notices and consult its supplied terms; missing license files are recorded rather than assigned a new license.

Compatibility code is kept separately, and exact import overlays leave upstream source files unchanged. Repository-local absolute imports are placed into private namespaces; no global `models`/`utils` aliases are installed. First import checks the recorded pin and exact file inventory against `compat/snapshot_integrity.json`. Missing, changed or additional files are rejected; Git metadata and generated caches are excluded. When local `.git` metadata exists, its repository root, HEAD and tracked source cleanliness are checked too. The outer package Git repository is never mistaken for an upstream checkout. Changes require an explicit new revision/repair rather than silently mixing provenance.

```bash
# Idempotently verify/recover the recorded source checkouts:
python -m openhisto.mil.models.code_bases.sources
# Run real synthetic tests and refresh coverage only on full success:
python -m openhisto.mil.models.code_bases.validate --write-report
```

Source acquisition refuses unmanaged or edited directories. A missing managed checkout is restored to its recorded commit, even if the remote branch has advanced. It never silently upgrades an existing pin.

Existing vendored snapshots are verified without network access or recloning.
Maintainers can regenerate inventories with
`python -m openhisto.mil.models.code_bases.snapshot`; generation requires all original
repository-local checkouts to be pristine at their recorded commits and never
trusts an arbitrary vendored directory as a new baseline.

CPU validation covers all 34 public profiles using real native implementations: finite forward/backward, strict state round trips, model-specific auxiliaries, Mamba reference mathematics, hierarchy constraints, import isolation, changed-source rejection and acquisition reruns/restoration. Prototype and checkpoint fixtures are synthetic. These tests establish executable compatibility; pretrained accuracy, full-slide throughput, GPU parity and published benchmark reproduction remain untested.

Coverage uses small synthetic configurations and focused boundary cases; it does not certify every upstream hyperparameter combination or optional ablation.
