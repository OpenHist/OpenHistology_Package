# Semantic segmentation

Train, evaluate, and predict binary or multiclass semantic masks through one
Python API or JSON-configured CLI. Models and datasets remain local. The
`instance/` directory is reserved; exported connected regions are semantic
polygons, not independently trained object instances.

Requires Python 3.12 or newer. From the repository root:

```bash
python -m pip install -e ".[segmentation-transformers]"
python -m openhisto.segmentation models
python -m openhisto.segmentation --help
```

Install the `segmentation-torch` extra for DeepLabV3/SegDiff without Transformers.
SAM2 and SegDiff have their own extras (`segmentation-sam2`, `segmentation-segdiff`);
SAM2's pinned official source is installed in a separate step below.
the optional SegDiff Mamba network additionally needs a CUDA build of `mamba-ssm`.
Install a PyTorch/torchvision pair appropriate for your CPU/CUDA platform. Core data
and export operations need only the `segmentation` extra; catalogue inspection imports
no Torch or Transformers and downloads nothing. The `feature_encoder` adapter
additionally needs the `feature-encoder` extra (including AnnData).

## What the reference pipeline does

The source is `AIM_Lab_Stuff/segmentation/train_segmentation`. Its CSV rows name
image and mask directories, matching `name.png` with `name_mask.png`. It resizes
patches, converts grayscale masks to binary using `>=128`, applies horizontal
flips and brightness/contrast, and delegates loss computation to the selected
model. AdamW trains the model; exponential learning-rate decay runs each epoch.
Validation computes loss; checkpoints and qualitative overlays are saved. The
original prediction script requires masks and writes binary PNGs.

This module preserves the model families and optimization defaults while fixing
class loss, 0/1-mask decoding, rectangular dimensions, image/label augmentation
alignment, first-item-only prediction, and ground-truth-derived inference
prompts. It adds explicit geometry, multiple formats, metrics, and resumable
checkpoints. The research scripts themselves are unchanged.

| Model / accepted source alias | Default initialization and objective |
| --- | --- |
| `mask2former` / `M2F` | `facebook/mask2former-swin-tiny-ade-semantic`; native query classification, matched mask BCE and Dice |
| `segformer` / `SegFormer` | `nvidia/segformer-b5-finetuned-ade-640-640`; pixel cross-entropy |
| `upernet` / `UPerNet` | Random ConvNeXt, matching the source; an explicit model path selects pretrained weights; main-head cross-entropy |
| `deeplabv3` / `DLV3` | torchvision pretrained ResNet50 with a replaced class head; cross-entropy |
| `sam2` / `SAM2` | Official SAM2 image model; binary prompt-conditioned BCE + Dice |
| `segdiff` / `SegDiff` | Random embedding-conditioned mask diffusion; noise MSE + 0.1 CE + 0.1 Dice |

For Mask2Former, ignore/padded pixels require valid-pixel matching and mask loss:
the native HF mask criterion treats these pixels as negatives. Fully labeled,
unpadded inputs use the native criterion. Background class zero is retained.
The optional DeepLab auxiliary loss is disabled by default; enabling it uses a
configurable weight (default 0.4). DeepLab/UPerNet automatically use running
BatchNorm statistics for singleton training batches; `freeze_batch_norm=True`
freezes them for every batch, and `False` requests the backend's native behavior.
No patches are dropped to avoid singleton batches.

## Python workflow

```python
import numpy as np
from openhisto.segmentation import (
    Sample,
    MaskSpec,
    ModelConfig,
    TrainConfig,
    PredictConfig,
    prepare_model,
    train,
    load_model,
    predict,
    evaluate,
    export_prediction,
)

# A sample's image can also be a PNG/JPEG/TIFF/NumPy path.
image = np.zeros((128, 192, 3), dtype=np.uint8)
mask = np.zeros((128, 192), dtype=np.uint8)
mask[32:96, 48:144] = 1
samples = [Sample(image, mask, sample_id="example")]

# Explicit preparation may download the selected pretrained weights.
model = prepare_model(ModelConfig(model="SegFormer", image_size=(128, 192)))
result = train(
    samples,
    samples,
    "experiment",
    model=model,
    train_config=TrainConfig(num_epochs=2, batch_size=1),
)

model = load_model("experiment/best")  # bundled local weights and preprocessing
# Prediction ignores annotations, even if supplied in the input samples.
for prediction in predict(model, samples, config=PredictConfig(batch_size=2)):
    export_prediction(
        prediction, "predictions", formats=("labels", "probabilities", "overlay", "qupath")
    )
print(evaluate(model, samples))
```

Use independent training/validation/test sets for research; the repeated sample
above only illustrates the API. `predict` yields results incrementally. Each
result has an integer `[H,W]` `labels` array, optional float32 `[C,H,W]` arrays,
the original RGB image, sample/slide identity, class names, and geometry.
Default prediction retains probabilities; request `scores=True` or `logits=True`
through `PredictConfig` when needed. Set probabilities false to retain only labels.

Training defaults are 30 epochs, batch size 5, image size `(1000,1000)`, learning
rate `5e-4`, AdamW weight decay `0.01`, scheduler gamma `0.95`, seed zero, and zero
loader workers. `device="auto"` selects CUDA when available, otherwise CPU.
Explicit CPU/CUDA/MPS selection is available where the chosen backend supports it.
All size tuples are **height, width**; coordinates are **x, y**.

## Masks and annotations

`prepare_dataset(source, mask_spec=..., num_classes=...)` returns lazy `Sample`
records. `Sample.mask_spec` overrides the dataset default. Internal labels are
contiguous integers `0..num_classes-1`; zero is background and `-100` is ignored
by default. Set `ModelConfig.ignore_index` to change the ignored label.

| Mask | Specification |
| --- | --- |
| Boolean / 0–1 / 0–255 binary | `MaskSpec(kind="binary")`; auto recognizes binary encodings when there are two classes |
| Class-ID PNG/TIFF/NPY | `MaskSpec(kind="indexed")`; integer precision is preserved |
| Sparse/raw or instance IDs | `value_map={0:0, 42:1, 1300:2}`; instance kind requires this explicit mapping |
| RGB/RGBA class colors | `MaskSpec(kind="rgb", palette={"0,0,0":0, "255,0,0":1})` |
| HWC/CHW binary channels | `MaskSpec(kind="channels", channel_axis=0, channel_map={0:1, 1:2})` |
| QuPath/GeoJSON | `MaskSpec(kind="geojson", class_map={"tumor":1, "stroma":2})` |

Use `ignore_values=[255]` for a raw void label in an indexed mask. Auto binary
mode interprets 255 as foreground, so choose the mask kind explicitly when that
value is ambiguous. Unknown colors, unmapped values, ambiguous axes, noninteger
labels and mismatched dimensions raise errors. Overlapping channel classes need
`class_priority`, with the first entry having highest priority.

GeoJSON accepts Polygon and MultiPolygon, including interior holes. QuPath
classes default to `properties.classification.name`; `class_property` can name
another dotted property. With no class map, all polygons become foreground class
one. Later features overwrite earlier features unless `class_priority` is
supplied. Annotations are sampled at output-pixel centers; only the requested
patch is rasterized. Invalid geometries are rejected rather than repaired silently.

## Files, manifests, and patch archives

A standard CSV contains:

```csv
sample_id,slide_id,image_path,mask_path
patch_a,slide_a,images/a.png,masks/a_mask.png
patch_b,slide_a,images/b.png,masks/b_mask.npy
```

Paths resolve relative to the CSV. Omit `mask_path` for prediction. JSON lists of
sample records are supported too. Additional fields include `mask_spec`,
`xy_level0`, `footprint_level0`, `valid_bounds`, `prompts`, and `embedding`;
structured CSV fields contain JSON. Direct Python `Sample` records accept arrays.
Legacy `slide_dir,mask_dir` rows are supported, with strict missing/orphan pair
checks. Background-only patches are retained. IDs must be unique and usable as
filename components for export.

Image inputs decode palette PNG colors and composite transparency onto white.
Palette masks retain their integer class IDs with `kind="indexed"` or `"auto"`;
`kind="rgb"` explicitly decodes their colors before applying the class palette.

Existing extraction archives supply the geometry:

```python
from openhisto.segmentation import prepare_dataset, MaskSpec

samples = prepare_dataset(
    {
        "archive_path": "slide_patches.h5",
        "slide_id": "slide_a",
        "annotation": "slide_annotations.geojson",
        "mask_spec": MaskSpec(kind="geojson", class_map={"tumor": 1}),
    }
)
```

The archive is consumed through `patch_extraction.PatchReader`; its schema is
unchanged. Sampling uses each row's level-0 origin, footprint, and valid bounds,
with private file handles per process. A selected archive row can also be supplied
as `Sample(image={"archive_path":"slide_patches.h5", "index":7}, ...)`.
For a slide-level raster mask, supply both `MaskSpec.origin=(x,y)` and
`downsample=(sx,sy)` in level-0 pixels, plus image geometry when it is not supplied
by an archive. Neither magnification nor coordinates are parsed from filenames.
Out-of-mask and padded target pixels are ignored. TIFF sources are memory-mapped
when possible; compressed TIFF fallback may decompress the source mask.

## SAM2 and SegDiff

SAM2 is binary only. Prediction and reported accuracy require explicit points or
boxes in the **native image's pixel coordinates**:

```python
sample = Sample(
    "patch.png",
    sample_id="a",
    prompts={
        "point_coords": [[120, 90]],
        "point_labels": [1],
        "boxes": [[80, 50, 170, 140]],
    },
)
```

Prompts are transformed with the image. Multiple prompt groups are merged as
foreground. Training may simulate prompts from labels; validation without
explicit prompts reports conditional loss only, with no segmentation accuracy.
Fine-tuning uses the official differentiable `SAM2Train`, not the inference
predictor. The resolved architecture and weights are bundled for offline reload.
Install the SAM2 extra and the separate pinned official source requirements:

```bash
python -m pip install -e ".[segmentation-sam2]" setuptools wheel
SAM2_BUILD_CUDA=0 python -m pip install --no-build-isolation -r src/openhisto/segmentation/requirements-sam2.txt
```

The requirements file preserves the official code revision used by the tests.
It stays outside package dependency metadata because PyPI rejects direct URL
dependencies. CUDA is recommended for full-resolution training.
The command above runs the CPU acceptance path without the optional CUDA
extension. On a CUDA host, omit `SAM2_BUILD_CUDA=0` to build that extension
against the installed Torch. The file constrains Torch, torchvision and
TensorDict to the tested compatibility line; the upstream development extra
is unnecessary for this adapter and would also install GUI OpenCV.

SegDiff requires **one embedding per patch**; RGB pixels supply dimensions, not
image conditioning. Supply a vector in `Sample.embedding`, or a reference:

```python
# A completed sibling feature_encoder store; explicitly select model/run and row.
embedding = {
    "kind": "feature_store",
    "path": "features.h5ad",
    "model": "my_encoder",
    "mag": 20,
    "row": "observation_id",
    "run_key": "selected_run",
}
# Legacy HDF5: an explicit row or unique xy coordinate is required.
embedding = {
    "kind": "legacy",
    "path": "features.h5",
    "row": 7,
    "features_key": "features/20x",
    "coords_key": "coords/20x",
}
# Array files support an optional row or NPZ key.
embedding = {"path": "embedding.npy"}
```

Training discovers and validates the embedding dimension before constructing the
optimizer. For explicit `prepare_model`, supply `options={"feature_dim": D}`.
Other options include `timesteps` (1000), `base_channels` (64), `channel_mults`
([1,2,4]), `time_emb_dim` (256), `embed_dim` (256), `beta_schedule`,
`network_style` (`none`, `low_dim`, `attention`, `mamba`), and loss weights.
Fixed embeddings disable augmentation. Inference is stochastic unless
`inference_seed` is set. The full reverse diffusion chain can be expensive.

The source's effective global conditioning and bottleneck transformations are
preserved. Its nominal multilevel attention/Mamba pooled branches did not affect
the decoder; those unused computations are omitted. Single-token conditioning
acts as a broadcast embedding addition. This port does not introduce a new
spatial-attention architecture. Mamba is optional and generally needs its own
supported CUDA environment.

## CLI configuration, checkpoints, and exports

Save this as `experiment.json`:

```json
{
  "model": {"model":"SegFormer", "num_classes":3,
            "class_names":["background","tumor","stroma"], "image_size":[512,512]},
  "mask": {"kind":"indexed"},
  "train": {"num_epochs":30, "batch_size":5, "seed":0},
  "predict": {"batch_size":4, "probabilities":true},
  "exports": ["labels","probabilities","qupath"]
}
```

```bash
python -m openhisto.segmentation prepare-model local_model --config experiment.json
python -m openhisto.segmentation prepare-data train.csv prepared_data --config experiment.json
python -m openhisto.segmentation train train.csv val.csv run --config experiment.json
python -m openhisto.segmentation train train.csv val.csv run --config experiment.json --resume run/latest
python -m openhisto.segmentation evaluate run/best test.csv --config experiment.json
python -m openhisto.segmentation predict run/best images.csv predictions --config experiment.json
```

`prepare-model` explicitly downloads/initializes a backend; `--random-init` avoids
pretrained downloads. To train the prepared bundle, pass `model=load_model(...)`
in Python. `prepare-data` materializes native RGB images and canonical NPY masks,
plus a portable relative-path CSV; it refuses existing destinations. It does not
extract patches from whole-slide files.

Bundles contain a versioned `bundle.json`, model configuration, mask defaults,
normalization/processor state, architecture and weights. `latest`, `best`, and
periodic epoch bundles include optimizer/scheduler, RNG, epoch and history state;
`final` is for inference. Resume uses the checkpoint's model and requires the same
optimizer, batch, worker and augmentation settings; increase `num_epochs` to the
new total. Dataset identities/content must remain unchanged when resuming.
`history.json` persists train/validation loss and metrics. Metrics exclude ignored
pixels; absent classes have null values and are excluded from macro averages.
`foreground_dice` additionally excludes background. All-ignore targets are rejected.

Available export names are `labels` (default), `labels_png`, `labels_tiff`,
`binary`, `probabilities`, `scores`, `logits`, `palette`, `overlay`, `geojson`, and
`qupath`. Label PNG uses uint8/uint16; larger IDs use uint32 TIFF. Binary exports
are per foreground class. A metadata JSON accompanies every export. Existing
files are refused unless `overwrite=True`/`--overwrite` is supplied.

Dense model probabilities use softmax, SAM2 uses `[1-sigmoid, sigmoid]`, and
Mask2Former normalizes its summed query-class/mask scores across classes. These
scores are not calibrated confidence estimates. Mask2Former has no native dense
semantic logits, so requesting them raises `CapabilityError`. NumPy outputs retain
class axes; classes are never silently merged. Polygon exports preserve holes and
use level-0 coordinates when geometry is supplied, otherwise native image pixels.
Padded bounds are excluded from foreground visualization and polygon products.

## Verification

```bash
python -m unittest discover -s src/openhisto/segmentation/tests -t src -v
# Select optional real model tests after installing their dependencies:
python -m unittest discover -s src/openhisto/segmentation/tests -t src -p 'test_standard.py' -v
python -m unittest discover -s src/openhisto/segmentation/tests -t src -p 'test_special.py' -v
```

Recorded CPU validation on macOS used Python 3.12.6, PyTorch 2.5.1, torchvision
0.20.1, Transformers 4.52.4, and the pinned SAM2 revision: **53 tests passed**, with
actual backward and offline reload checks for all six adapters. Optional dependencies were installed
in an isolated temporary target for this verification; the corresponding base
environment reported nine explicit optional-dependency skips. The Mamba CUDA
variant was not exercised in that CPU validation environment.

Tests use generated images/annotations, small random-weight model configurations,
and temporary files. They exercise actual model backward passes, local weight
round trips, class preservation, ignored-pixel losses, mask formats, geometry,
exports, unlabeled batches, and exact interrupted/resumed CPU training. Optional
backends report explicit skips when missing. No patient data or pretrained weights
are downloaded by tests. Real scanner and cohort validation remains dataset-specific.

Whole-slide extraction/stitching, automatic SAM2 mask generation, native instance
training, and distributed training are outside this module's first version.
