"""OpenHisto: computational pathology tools from whole-slide images to slide-level models.

Each subpackage is imported on its own and has its own optional dependencies
(see the ``[project.optional-dependencies]`` extras):

- ``openhisto.default_datasets``: public slide acquisition and dataset metadata
- ``openhisto.patch_extraction``: whole-slide patches in indexed HDF5 archives
- ``openhisto.stain_transfer``: reference- and model-based stain transfer
- ``openhisto.feature_encoder``: multi-model, multi-magnification patch features
- ``openhisto.segmentation``: binary and multiclass semantic segmentation
- ``openhisto.vlas``: prompted patch descriptions from vision-language models
- ``openhisto.mil``: multiple-instance learning backbones and task heads

This module deliberately imports nothing: isolated model-worker interpreters
import ``openhisto`` without the scientific dependencies installed.
"""

__version__ = "0.1.0"
