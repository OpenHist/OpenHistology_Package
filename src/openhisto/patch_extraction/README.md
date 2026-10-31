# Patch extraction

Extract whole-slide RGB patches into **one indexed HDF5 file per slide**. The
archive holds independently compressed images, their coordinates, and technical
extraction settings. Ordered iteration and random reads use the same index.

PNG is lossless by default. Pixel preservation refers to the extracted RGB patch
after magnification resampling, not to unchanged scanner pixels. JPEG is available
only when explicitly requested. Consolidation reduces inode use; a PNG archive
can have a similar logical size to separate PNG files, since PNG already compresses
the images. Filesystem block usage and HDF5 metadata also affect the result.

## Installation

From the repository root:

```bash
python -m pip install -e ".[patch-extraction-opencv]"
python -m openhisto.patch_extraction --help
```

The extra includes `openslide-bin`, which supplies OpenSlide's native
library alongside the Python bindings. If you already maintain OpenSlide through
a system package manager, you can use that installation instead. See the
[OpenSlide installation documentation](https://openslide.org/api/python/#installing).
The importable storage reader does not need to open OpenSlide. The
`patch-extraction` extra includes scikit-image for Otsu foreground filtering and
tifffile for TIFF masks. Mask-centered extraction additionally needs `cv2`.
The `patch-extraction-opencv` extra installs headless OpenCV for a fresh
environment. If a compatible OpenCV distribution is already installed, use
`.[patch-extraction]` to reuse it without installing a second distribution.

## Python API

```python
from openhisto.patch_extraction import ExtractionConfig, PatchReader, extract_grid

config = ExtractionConfig(
    patch_size=512,
    target_mag=20,
    # source_mag=40,  # supply when objective-power metadata is unavailable
    foreground_method="HSV",
)
summary = extract_grid("input.svs", "patches.h5", config)
print(summary.to_dict())

with PatchReader("patches.h5") as patches:
    print(len(patches), patches.metadata)
    coordinates = patches.get_coordinates()  # ordered [N, 2] level-0 (x, y)
    if len(patches):
        record = patches[0]  # patches.read(0) is equivalent
        rgb = record.image  # uint8 [height, width, 3]
        print(record.index, record.xy_level0, record.footprint_level0)
        print(record.valid_bounds)  # output-pixel (x0, y0, x1, y1)
    for record in patches:
        pass  # only one image is decoded by each read
```

`get_coordinates(start, stop)` reads a coordinate slice without decoding images.
Indexing supports negative indices. Each reader opens a private file handle in
the process that uses it; live HDF5 handles are excluded when the reader is
pickled. Training code can construct a reader inside each worker. Close readers
when finished, preferably with a context manager. Readers consume completed,
immutable files; concurrent reading of an active writer is not supported.

### Mask-centered patches

```python
from openhisto.patch_extraction import ExtractionConfig, extract_mask_centered

summary = extract_mask_centered(
    "input.svs",
    "binary-mask.tif",  # alternatively a two-dimensional binary NumPy array
    "objects.h5",
    ExtractionConfig(patch_size=512, min_box_size=4),
    mask_downsample=(8, 8),  # level-0 pixels per mask pixel, X and Y
    mask_origin=(0, 0),  # level-0 location of the mask's top-left boundary
)
```

Masks select regions; this version does not save paired training masks. External
connected-component contours supply bounding boxes, whose centers determine the
square patches. Mask coordinates are converted using the explicit scale and
origin, never inferred from a filename or shape. Anisotropic mask scales are
supported. Use a two-dimensional binary mask, with boolean, 0/1 or 0/255 values;
select a channel yourself if a TIFF contains multiple classes or planes.

`min_box_size` is the minimum width **and** height in mask pixels (default 0).
Mask-selected patches bypass the grid foreground filter unless
`filter_mask_patches=True`. Component discovery holds the supplied mask and
contours in memory; use a suitably downsampled mask for very large slides.

## Command line

```bash
python -m openhisto.patch_extraction grid input.svs patches.h5 \
  --patch-size 512 --target-mag 20 --source-mag 40

python -m openhisto.patch_extraction mask input.svs binary-mask.tif objects.h5 \
  --mask-downsample 8 8 --min-box-size 4 --edge pad

# Explicitly select lossy image storage:
python -m openhisto.patch_extraction grid input.svs smaller.h5 \
  --codec jpeg --jpeg-quality 95
```

Successful commands print a JSON extraction summary. Use the subcommand's
`--help` for all options. Existing files are refused unless `--overwrite` is set.

## Coordinates, resolution and defaults

| Setting | Default / meaning |
| --- | --- |
| `patch_size` | 512 output pixels per side |
| `stride` | Patch size, measured in target-resolution pixels |
| `target_mag` | 20×; cannot exceed the source magnification |
| `source_mag` | Explicit override, otherwise scanner objective-power metadata |
| `foreground_method` | `HSV`; also `SL`, `luminance`/`otsu`, or `none` |
| `sat_thresh`, `tissue_fraction` | 0.05 and 0.05 for HSV filtering |
| `lum_thresh`, `sl_tissue_fraction` | 220 and 0.2 for SL filtering |
| `edge` | `drop`; `pad` composites out-of-slide pixels onto white |
| `codec`, `png_compression` | `png`, level 6 (configurable 0–9) |
| `jpeg_quality` | 95; used only with `codec="jpeg"`, 4:4:4 sampling |

All stored origins and footprints are integer **level-0** pixels. Patch `i` uses
row `i` in every coordinate/geometry dataset. Grid patches are ordered by Y then
X; mask components are sorted by position. Fractional scale factors are preserved
when generating grid origins, avoiding accumulated integer-step drift. Footprints
are rounded to integer source pixels; their recorded values make the effective
sampling geometry explicit. Resolution here uses objective magnification rather
than calibrated physical MPP.

Extraction selects a pyramid level at least as fine as the requested resolution
and resizes RGB data to the output dimensions. The edge check uses the entire
level-0 footprint. With padding, origins can be negative and `valid_bounds`
identifies the usable rectangle in output pixels; padding is excluded from
foreground statistics. Overlapping patches remain distinct indexed records.

## Archive layout and resource use

| Dataset | Contents |
| --- | --- |
| `images/bytes` | Concatenated independently encoded PNG/JPEG byte streams |
| `images/offsets` | `N+1` byte offsets; image `i` occupies offsets `i:i+2` |
| `coords` | `[N,2]` level-0 origins |
| `footprints` | `[N,2]` level-0 width and height |
| `valid_bounds` | `[N,4]` usable output-pixel rectangle, exclusive right/bottom |

Root attributes identify schema version, completion state, image size/codec and
JSON metadata. Extraction metadata contains technical settings and counts rather
than original paths, slide identifiers, arbitrary scanner properties, or label
images. Callers choose their own output filename.

Patches are read, filtered, encoded and appended incrementally. Image buffers
scale with patch/read size rather than the number of extracted patches; the
mask-based workflow additionally needs mask/component memory. Encoded bytes are
not gzip-compressed again. A writer owns a temporary file beside the destination,
validates offsets and aligned dataset lengths, and publishes only after success.
Handled failures remove the temporary file. A forcibly killed process can leave
an unpublished temporary file, which can be removed after confirming no writer
is using it. Resume and simultaneous writers to the same archive are not supported.

No CSV or per-patch image sidecars are produced by default.

## Validation and storage comparison

From the repository root:

```bash
python -m unittest discover -s src/openhisto/patch_extraction/tests -t src -v
python -m openhisto.patch_extraction.benchmark --count 64 --patch-size 512
```

Tests use generated images, mocked slide readers and temporary archives. They
cover exact PNG round trips, order/coordinate alignment, fractional scales,
pyramid reads, masks and offsets, edge policies, empty extraction, file integrity,
publication failures and process-local readers. Native OpenSlide smoke tests
generate tiled TIFFs and exercise the CLI; these tests skip if the native library
is unavailable. No patient files or sample-slide downloads are needed. Testing
your scanner's actual slide format remains a separate step.

The optional benchmark generates deterministic synthetic texture, writes both
separate PNGs and an archive, measures logical/allocated bytes and write/read
throughput, and verifies decoded pixels. Temporary benchmark outputs are removed.
Results depend on image content, filesystem caching and hardware, and do not
predict a fixed compression ratio for real slides.

One measured run on macOS arm64 (Python 3.12.6, Pillow 11.0.0, h5py 3.13.0),
using 64 synthetic 512×512 patches, seed 0 and PNG level 6:

| Measurement | Separate PNGs | HDF5 archive |
| --- | ---: | ---: |
| Regular files | 64 | 1 |
| Logical bytes | 26,876,429 | 26,986,600 |
| Allocated bytes | 27,000,832 | 27,762,688 |
| Encode/write seconds | 3.286 | 3.260 |
| Ordered reads, patches/second | 148.4 | 143.9 |
| Shuffled reads, patches/second | 147.7 | 143.6 |

Every decoded image matched its original pixel hash. This run had about 0.4%
logical archive overhead; allocated-space overhead was larger on this filesystem.
The improvement was file count, with comparable throughput and compressed
lossless storage. Directory/inode storage is excluded from the byte measurements.
