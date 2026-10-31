# Stain transfer

Normalize **input patches → a destination appearance** using Vahadane, Macenko,
Reinhard, histogram matching, MultiStain CycleGAN, or pix2pix. The module accepts
individual images, recursive image folders, CSV/JSONL manifests, and native
`patch_extraction` HDF5 archives. It also trains CycleGAN and pix2pix generators.

Python 3.12 is the validation target. Existing source modules under
`AIM_Lab_Stuff` are not needed at runtime.

## Installation and quick start

From the repository root:

```bash
python3.12 -m venv .venv-stain
source .venv-stain/bin/activate
python -m pip install -e ".[stain-transfer]"
# Vahadane / Macenko:
python -m pip install -e ".[stain-transfer-spams]"
# CycleGAN / pix2pix training or inference:
python -m pip install -e ".[stain-transfer-gan]"
python -m openhisto.stain_transfer methods

python -m openhisto.stain_transfer normalize input.png normalized.png \
  --reference reference.png
python -m openhisto.stain_transfer normalize patches/ normalized/ \
  --method reinhard --reference reference.png
python -m openhisto.stain_transfer normalize patches.h5 normalized.h5 \
  --method histogram --reference reference.png
```

The default method is **Vahadane** (`default` is an alias), seed is `256`, and
brightness standardization is off. No reference image or pretrained weights are
bundled. A destination reference/checkpoint is required. `hist` is an alias for
`histogram`. SPAMS is imported only for Vahadane/Macenko; Torch
is imported only for GAN workflows. Install one compatible OpenCV distribution
in an environment, rather than mixing headless and GUI distributions.

Vahadane retains the legacy positive sparse SPAMS objective. Fitting now uses an
explicit seed-derived dictionary, single-thread batch updates and a fixed 100
iterations for reproducibility. These numerical settings differ from the old
implicit online/time-limited defaults; legacy pixel-for-pixel equivalence is not
promised. Provenance records solver settings and library versions. Supported
classical options are `luminosity_threshold` (0.8), `stain_regularizer` (0.1),
`concentration_regularizer` (0.01), and `iterations` (100). Histogram matching uses
per-channel quantiles over occupied reference intensities, with quantile clipping
`(0.01,0.99)`. Empty histogram bins do not shift colors toward the next occupied
intensity. Provenance identifies this mapping as `occupied_rgb_quantiles_v1`.
The input/output contract remains 8-bit RGB for every method.

On macOS, SPAMS executes in a persistent local worker so its OpenMP runtime can
coexist with Torch in mixed jobs. Calls have a 300-second response timeout and
a 256 MiB message limit; use patch-sized reference images. Other platforms use
SPAMS directly. Workers close automatically after file/archive jobs.

## Python API and jobs

```python
from openhisto.stain_transfer import DomainSpec, TransferConfig, create_normalizer, normalize

appearance = DomainSpec(method="reinhard", references=("reference.png",))
# rgb is a uint8 NumPy array shaped [height, width, 3].
with create_normalizer(appearance) as normalizer:
    result = normalizer.transform(
        rgb, valid_bounds=(0, 0, rgb.shape[1], rgb.shape[0]), patch_id="slide/patch-0"
    )
report = normalize("patches.h5", "normalized.h5", appearance, TransferConfig(standardize=False))
print(report.to_dict())
```

The normalizer fits references or loads models once. Statistical methods exclude
pixels outside `valid_bounds` from their calculations, and every method preserves
those pixels on output. Empty valid regions and patches whose valid pixels are
all at least 245 in all channels are returned unchanged and counted as blank.
Unusable references fail during initialization. Degenerate source stain estimates
are preserved; input corruption and model errors fail with patch context.
Use a context manager or `normalizer.close()` to release backend resources after
direct API use. File/archive jobs close their normalizers automatically.

JSON jobs contain `input`, `output`, `domain` and optional `config`:

```json
{
  "input": "patches/slide.h5",
  "output": "normalized/slide.h5",
  "domain": {"method": "vahadane", "references": ["reference.png"]},
  "config": {"seed": 256, "standardize": false, "overwrite": false}
}
```

```bash
python -m openhisto.stain_transfer normalize --job src/openhisto/stain_transfer/examples/normalize.json
```

Every path in a job is relative to the job JSON, including checkpoint/reference
paths and training paths. Copy the examples and replace their synthetic paths.
CLI flags cannot override a job silently. Successful commands print JSON reports;
errors print a JSON error to stderr and return a nonzero exit status.

### Explicit mixtures

A single destination is used normally. Multiple references or method/reference
candidates require `TransferConfig(mix=True)` / `--mix`. Candidates are selected
with equal probability using a SHA256 mapping of seed and stable patch identity:
processing order does not change the selected method. A direct `transform` call
must supply a nonempty `patch_id` for a mixture.

```python
mixture = DomainSpec(
    candidates=(
        DomainSpec(method="reinhard", references=("reference-a.png",)),
        DomainSpec(method="histogram", references=("reference-b.png",)),
    )
)
report = normalize("patches/", "mixed/", mixture, TransferConfig(mix=True))
```

Multiple references for one method are another shorthand; see
[examples/mixture.json](examples/mixture.json). Selection uses relative file paths
for folders, manifest patch IDs (or supplied paths), and original archive indexes.
Keep these identities stable for reproducibility. Each candidate is fitted/loaded
once, so a mixture of several GAN checkpoints occupies the sum of their model
memory.

## Inputs, outputs, and recovery

- **Single image:** output must end in `.png`; provenance is `image.png.json`.
- **Folder:** recursively mirror relative paths with `.png` extensions; write
  `stain_transfer.json`. Two inputs such as `a.jpg` and `a.png` that map to the
  same output are rejected before writing.
- **Manifest:** CSV/JSONL requires `image_path` (accepts legacy `path`); paths are
  relative to the manifest. Optional unique `patch_id`, `slide_id`, `mag`, bounds,
  and other fields are retained. Write a same-basename manifest and lossless
  images under the output directory. Supply `patch_id`, `slide_id`, and `mag`
  when the result is intended for `feature_encoder`. Do not reuse IDs within
  one manifest. `valid_bounds=[x0,y0,x1,y1]` or separate `valid_x0` etc. fields
  identify output-pixel bounds with exclusive right/bottom edges.
- **Archive:** input/output `.h5` or `.hdf5` uses the existing `PatchReader` schema.
  Preserve count, order, coordinates, footprints, valid bounds and extraction
  metadata; output images are PNG. Normalization metadata is nested under
  `stain_transfer`. Existing normalization history is retained.

RGB/grayscale 8-bit files are accepted, with transparency composited on white.
High-bit-depth, floating-point, and unsupported color-space files require explicit
conversion. Image buffers are bounded to one patch at a time; directory/manifest
indexes still use memory proportional to patch count. `workers=0` and
`batch_size=1` are currently required for normalization; unsupported values fail
explicitly. Run independent slides/jobs in separate processes for parallelism.

Existing output files are refused unless `overwrite=True` / `--overwrite` is
explicit. Inputs cannot be overwritten. Every image publishes atomically;
archives publish atomically only after completion. A failed folder/manifest job
can leave completed image files, without a success provenance record. Inspect
these before rerunning with overwrite. Archive failures preserve the existing
published output and remove handled temporary work. Simultaneous writers to the
same destination and normalization resume are not supported.

Normalized archives can be consumed by `feature_encoder` directly. Because
normalization changes pixel hashes, use a distinct feature source ID or a fresh
feature output for each appearance; an existing encoding deliberately rejects
changed source pixels.

## GAN inference

```bash
python -m openhisto.stain_transfer normalize patches.h5 normalized.h5 \
  --method cyclegan --checkpoint training/generator_AtoB.pth --device cpu
```

New exports have an adjacent `.json` containing method, architecture, generator
width, normalization, input preprocessing and effective source/destination names
and direction. Keep it with the generator. A raw legacy checkpoint instead needs
explicit settings:

```bash
python -m openhisto.stain_transfer normalize patches/ normalized/ \
  --method cyclegan --checkpoint latest_net_G_A.pth \
  --architecture unet_256 --input-mode grayscale --direction AtoB \
  --options '{"ngf":64,"norm":"instance","dropout":false}'
```

MultiStain CycleGAN training used replicated three-channel grayscale input;
its previous standalone inference script incorrectly supplied RGB. Use declared
grayscale preprocessing for matching trained models. Explicit `rgb` settings
remain available for raw models trained to expect RGB. pix2pix inputs depend on
the paired data used during training; metadata records that choice.

Inference runs in evaluation/inference mode without random crops, flips or color
jitter. Bottom/right replicate padding meets each generator's stride/minimum
size; cropping restores original dimensions. Optional `options.inference_size`
resizes internally and then restores the original output dimensions. No output
geometry changes are made. Supported CycleGAN architectures are `unet_256`,
`unet_128`, `resnet_6blocks`, and `resnet_9blocks`; pix2pix uses
`pix2pix_unet_256`. CPU, CUDA and explicit MPS devices are supported; `auto`
selects CUDA when available, otherwise CPU. No checkpoints are downloaded.

## GAN training

```python
from openhisto.stain_transfer import TrainConfig, train

report = train(
    TrainConfig(
        method="cyclegan",
        source="lab-a/",
        target="lab-b/",
        output="training/",
        source_name="lab-a",
        target_name="lab-b",
        device="cpu",
    )
)
print(report.to_dict())
```

```bash
python -m openhisto.stain_transfer train --job src/openhisto/stain_transfer/examples/cyclegan-train.json
python -m openhisto.stain_transfer train --job src/openhisto/stain_transfer/examples/pix2pix-train.json
```

CycleGAN accepts unpaired recursive image folders or CSV manifests. Flat CSVs
use `path` and optional `mask_path`. Existing slide manifests use `slide_dir` and
`mask_dir`, expanding images matched to `*_mask` files. Manifest paths are
relative to their CSV. Masks must match their image dimensions and undergo the
same spatial transforms with nearest-neighbor interpolation.

pix2pix accepts source/target folders matched by exact relative filename, a CSV
with `source_path,target_path`, or concatenated A|B images supplied with
`paired_ab`. Unmatched files, differently sized pairs, and odd-width A|B images
fail. `grayscale_source=true` with only `target` derives grayscale/color pairs
on the fly. No duplicate images or automatic train/validation splits are made.
Training inputs use image folders/manifests; native HDF5 is a normalization input.

| Training setting | CycleGAN | pix2pix |
| --- | --- | --- |
| Generator | `unet_256`, width 64, instance norm, no dropout | eight-level U-Net, width 64, batch norm, dropout |
| Discriminator width | 32 | 64 |
| Input preprocessing | grayscale | RGB, or explicit grayscale self-pairing |
| Batch size | 1 | 4 |
| Training geometry | resize/crop 256 | resize 286, crop 256 |
| Epochs + linear decay | 50 + 50 | 30 + 30 |
| Learning rate | `1e-5` | `2e-4` |
| Objective | LSGAN; cycle 10; identity 0.5 | LSGAN; L1 100 |

Override these with `TrainConfig` fields. Widths may be reduced for smoke tests.
Backend `options` allow `norm`, `dropout`, `pool_size`, `color_jitter`,
`brightness`, `contrast`, `saturation`, `beta1`, and CUDA `amp`; unknown keys fail.
CycleGAN auxiliary segmentation/translated-mask consistency is enabled only by
`lambda_mask>0`, and requires usable masks. `lambda_real_seg` defaults to 1.
Paired geometry is synchronized; unpaired domains use independent transforms.
Loss JSONL logs and sample PNGs are written locally without a visualization server.

Checkpoints retain networks, optimizers, schedulers, epoch/global step, random
states, replay pools and optional AMP scalers. Set `resume` to a full training
checkpoint to continue; use `warm_start` for generator weights without training
state. Generator-only legacy files cannot supply full resume. Both CycleGAN
directions are exported as `generator_AtoB.pth` / `generator_BtoA.pth` with JSON;
pix2pix exports its single A→B generator. Keep domain mapping explicit: old
`direction=BtoA` jobs swapped the original domain inputs before training.
The latest complete state is `training_latest.pth`; set both `resume` and
`output` explicitly in a resume job. Fresh training refuses existing training
artifacts. The optional `max_steps` debug limit applies to one invocation;
partial-epoch continuation requires `workers=0`. Resume inherits omitted settings;
explicit values, including `workers=0`, `device="auto"` and `checkpoint_every=1`,
override the saved values. A checkpoint created with multiple workers can thus
continue with `TrainConfig(resume="training_latest.pth", workers=0, ...)`.

## Container and cluster use

From the repository root, on a Linux host with Apptainer/Singularity:

```bash
apptainer build stain-transfer.sif src/openhisto/stain_transfer/Singularityfile.def
apptainer run --bind /data stain-transfer.sif methods
apptainer run --bind /data stain-transfer.sif normalize --job /data/job.json
# CUDA jobs need a compatible host NVIDIA driver:
apptainer run --nv --bind /data stain-transfer.sif train --job /data/train.json
```

The recipe includes all six backends using Python 3.12 and pinned Torch.
SPAMS wheel availability depends on the host/build platform;
Linux x86-64 is the container target. The scheduler example
[examples/slurm-normalize.sh](examples/slurm-normalize.sh) assigns independent
job JSONs with distinct outputs to array tasks. Set `STAIN_DATA_ROOT`,
`STAIN_SIF`, and optionally `STAIN_JOB_LIST`; adjust array size and GPU requests
for your scheduler. The recipe cannot be built natively on this macOS workspace.

## Validation

```bash
python -m unittest discover -s src/openhisto/stain_transfer/tests -t src -v
```

Tests use generated stain textures, temporary archives and synthetic GAN weights.
Install the optional dependencies to execute their tests; unavailable backends
skip their integration tests. The suite checks actual SPAMS transforms,
repeatability, padding, all input formats, atomic publication, downstream archive
consumption, GAN shape/preprocessing/direction compatibility, reduced-width CPU
training, masks, generator exports and full-state resume. Tests validate software
behavior; no trained research checkpoint or real dataset is bundled.

The complete 62-test suite passed on macOS arm64 with Python 3.12.6,
NumPy 2.2.6, OpenCV 4.14.0, SPAMS-bin 2.6.14, Torch 2.14.1 and the feature
encoder's AnnData 0.13.3.post0 environment. Runtime warnings were treated as
errors. CUDA and a Linux container build were unavailable on this host.

Classical routines retain the [StainTools MIT notice](CLASSICAL_LICENSE.txt).
Vendored GAN definitions retain the [upstream license](licenses/CycleGAN-BSD.txt).
