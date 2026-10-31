# OpenHisto

**Created and maintained by Abteen Arab.**

We develop OpenHisto for computational pathology workflows, from whole-slide
images to slide-level classification and risk/survival models. Our modules cover
dataset acquisition, patch extraction, stain transfer, feature encoding,
semantic segmentation, vision-language descriptions and multiple-instance
learning (MIL).

Use the components independently or connect them through explicit patch
identities, geometry and metadata. Each module guide documents its public API,
command line, outputs and current limits.

## Components

| Module | What we provide |
|---|---|
| [Dataset acquisition](src/openhisto/default_datasets/README.md) | Discover and verify public TCGA/CAMELYON16 assets, select KPMP files from an Atlas export, and register approved local BRACS slides with their metadata. |
| [Patch extraction](src/openhisto/patch_extraction/README.md) | Grid or mask-centered RGB patches in one indexed HDF5 archive per slide, with coordinates, footprints and valid bounds. |
| [Stain transfer](src/openhisto/stain_transfer/README.md) | Reinhard, histogram, Macenko and Vahadane normalization, plus CycleGAN/Pix2Pix training and transfer. |
| [Feature encoding](src/openhisto/feature_encoder/README.md) | Selected local or API embedding profiles, ordered patch vectors, incremental updates and standard AnnData `.h5ad` stores. |
| [Semantic segmentation](src/openhisto/segmentation/README.md) | Binary or multiclass training, evaluation, prediction and geometry-aware exports, with distinct automatic, prompted and conditioned backends. |
| [Vision-language descriptions](src/openhisto/vlas/README.md) | Prompted patch descriptions and result readers, with selected models running in separate environments. Availability and validation status are documented per backend. |
| [MIL backbones](src/openhisto/mil/models/code_bases/README.md) | 34 CPU model profiles with [classification/task heads](src/openhisto/mil/heads/README.md), [classical classifiers](src/openhisto/mil/classification/README.md) and [survival heads](src/openhisto/mil/survival/README.md). Graph and hierarchy inputs follow each model's native contract. |

## Install from source

Use **Python 3.12 or newer**. Clone a checkout you can access, create an
environment and install the core package:

```bash
git clone https://github.com/OpenHist/OpenHistology_Package.git
cd OpenHistology_Package
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
```

The core is enough for the offline archive example below. Install the extras
for the workflows you need, from the repository root:

```bash
python -m pip install -e ".[patch-extraction-opencv,feature-encoder-local]"
```

| Workflow | Extra names |
|---|---|
| Dataset acquisition | `default-datasets` |
| Patch extraction | `patch-extraction-opencv`; use `patch-extraction` when reusing a compatible existing `cv2` installation |
| Stain transfer | `stain-transfer`, `stain-transfer-spams`, `stain-transfer-gan` |
| Feature encoding | `feature-encoder`, `feature-encoder-local`, `feature-encoder-google`, `feature-encoder-cohere`, `feature-encoder-aws` |
| Semantic segmentation | `segmentation`, `segmentation-torch`, `segmentation-transformers`, `segmentation-segdiff`, `segmentation-sam2` |
| Vision-language coordinator/readers | `vlas`; model-worker requirements are installed separately |
| MIL and task heads | `mil`, `mil-attention`, `mil-transformers`, `mil-graph-multiscale`, `mil-backbones` |
| Classical classifiers | `mil-classical`, `mil-classical-xgboost`, `mil-classical-lightgbm`, `mil-classical-catboost`, `mil-classical-all` |

We test MIL and stain-transfer GAN workflows on the Torch 2.6 / torchvision 0.21
compatibility line. The MIL guides include exact reference lock files. GPU
backends still need an appropriate Torch/runtime installation. The shared MIL
factory currently uses CPU float32.

SAM2 needs a separate pinned source installation described in the segmentation
guide. Legacy VLA workers may require Python 3.10; those requirements apply to
their isolated interpreters, not the main package.

## Quickstart: an offline patch archive

This example writes and reads one generated RGB patch in a temporary directory.
It uses no slide files, model weights or network requests.

```python
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from openhisto.patch_extraction import PatchArchiveWriter, PatchReader

rgb = np.random.default_rng(1729).integers(0, 256, (16, 16, 3), dtype=np.uint8)

with TemporaryDirectory() as directory:
    path = Path(directory) / "patches.h5"
    with PatchArchiveWriter(
        path, patch_size=16, metadata={"source_mag": 40, "target_mag": 20}
    ) as writer:
        writer.append(rgb, (0, 0), (32, 32), (0, 0, 16, 16))

    with PatchReader(path) as reader:
        np.testing.assert_array_equal(reader[0].image, rgb)
        print(len(reader), reader[0].xy_level0, reader[0].footprint_level0)
        # 1 (0, 0) (32, 32)
```

For a real WSI, follow the [extraction guide](src/openhisto/patch_extraction/README.md).
Source magnification must come from scanner metadata or a known explicit
override. Keep clinical targets and cohort splits in external manifests.

Before preparing an encoder or VLA model, inspect its catalogue and access
requirements. Selected inference/preparation may download weights. Some models
need approval, and cloud encoders send patches to a provider and can incur
charges. Public checkpoint access does not replace its model-specific terms.

## Tests and contributions

We welcome focused fixes, new adapters, documentation improvements and
reproducible bug reports. Start with [CONTRIBUTING.md](CONTRIBUTING.md) for
environment setup, relevant checks and source/provenance conventions.

Our functional tests use generated inputs and mocks rather than patient data or
real pretrained checkpoints. Install the relevant module extras, then run its
suite from the repository root, for example:

```bash
python -m unittest discover -s src/openhisto/patch_extraction/tests -t src -v
```

The MIL guides link synthetic CPU coverage reports. Those records establish
the tested functional paths, not real-cohort accuracy or GPU parity.

## License and attribution

We distribute our original code under the [MIT License](LICENSE), copyright
2026 Abteen Arab. Vendored model sources and adapted third-party implementations
retain their own licenses, attribution and source inventories. Our project
license does not replace those notices or the terms of external model weights
and datasets. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the
recorded source/license inventory and any unresolved notice information.
