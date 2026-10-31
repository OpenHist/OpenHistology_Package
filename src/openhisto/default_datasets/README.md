# Default datasets

Acquire native whole-slide images and preserve their associated clinical,
pathology-label and split metadata through one Python API and command line.

**Select → discover a plan → download or register local files → verify → export.**
Importing this module performs no network requests or downloads. Discovery reads
catalogs/metadata only; image transfers happen through the separate download
operation. The pipeline stops at a local slide manifest ready for preprocessing.

## Installation

Python 3.12 or newer is required. From the repository root:

```bash
python -m pip install -e ".[default-datasets]"
python -m openhisto.default_datasets --help
```

`requests` handles HTTP and `openpyxl` reads XLSX files. JSON, CSV and SQLite use
the standard library. Keep dataset roots outside the code directories. Each root
belongs to one dataset; use a separate root for each cohort/release you want to
retain. The latest saved plan defines the current exported cohort. Files from
earlier selections are retained, not automatically deleted.

## Supported providers

| Provider | Acquisition | Associated data |
| --- | --- | --- |
| TCGA | Explicit GDC projects, case UUIDs or file UUIDs; public diagnostic SVS slides by default | Original file associations and nested case clinical entities, linked by case UUID |
| KPMP | Public files selected in an Atlas repository CSV export; approved files can also be registered locally | Available public clinical tables, participant links and original export fields |
| CAMELYON16 | Anonymous official AWS inventory and TIFF downloads | Official reference labels, training/test information, optional XML/mask references and release license |
| BRACS | Approved local SVS files with supplied XLSX/CSV metadata | WSI labels, original split and supplied patient grouping; ROIs are excluded |

The providers retain native WSI formats. TIFFs are not renamed or converted to
SVS. CAMELYON16 and BRACS exports explicitly mark clinical data unavailable for
the supported releases rather than generating demographics or outcome columns.

Public downloads are supported without login, cookies or tokens. Obtain gated
data through the provider first, then use local import. The package does not
register accounts, accept access agreements, or implement authenticated FTP.
Standard Requests transport settings such as `HTTPS_PROXY`, `HTTP_PROXY`,
`NO_PROXY` and `REQUESTS_CA_BUNDLE` remain available. Implicit `.netrc` origin
authentication is disabled, including after redirects. Supplying a Session
preserves its transport configuration without changing its authentication state.

Official references: [GDC API](https://docs.gdc.cancer.gov/API/Users_Guide/Downloading_Files/),
[KPMP Atlas](https://www.kpmp.org/help-docs/atlas-repository),
[CAMELYON AWS release](https://registry.opendata.aws/camelyon/),
[BRACS dataset details](https://www.bracs.icar.cnr.it/details/).

## TCGA example

The example selects at most five slides. Edit the selection before use. Omitting
`limit` selects all slides matching the explicit project/case/file constraints.

```bash
python -m openhisto.default_datasets discover TCGA \
  --selection src/openhisto/default_datasets/examples/tcga.json --plan tcga-plan.json

python -m openhisto.default_datasets download tcga-plan.json --root /path/to/data/tcga-kirc
python -m openhisto.default_datasets verify --root /path/to/data/tcga-kirc
```

The discovery summary reports asset counts, expected bytes and unknown sizes.
Plans contain linkage metadata; the summary does not print clinical records.

TCGA selection accepts `projects`, `case_ids`, `file_ids`, `diagnostic_only` and
`limit`. Identifiers are explicit lists; supplied constraints are combined.
Defaults restrict discovery to open-access biospecimen Slide Image/SVS files in
the TCGA program. Set `diagnostic_only=false` to include additional eligible slide
images. Clinical retrieval expands demographic, diagnoses, treatments, follow-ups
and exposures for the selected case UUIDs. Multiple cases, diagnoses and
follow-ups are retained; survival targets are not derived. API schema names were
checked against GDC's schema endpoints without retrieving case/file records.

## CAMELYON16 example

```bash
python -m openhisto.default_datasets discover CAMELYON16 \
  --selection src/openhisto/default_datasets/examples/camelyon16.json --plan camelyon-plan.json
python -m openhisto.default_datasets download camelyon-plan.json --root /path/to/data/camelyon16
```

Select `split` (`train`, `test`, `all`), `splits`, or explicit `slide_ids`; `limit`
is optional. Set `include_annotations=true` to include related XML/mask assets.
They remain separate assets with their supplied coordinate semantics.

The official mirror uses `CAMELYON16/images/`, not separate training/testing
directories. The adapter preserves reference-table values and the documented
training filename labels. Test labels are not inferred from test filenames.
Conflicting label/split evidence is flagged and source values retained. Unknown
reference-table headers remain explicitly unrecognized; they do not become
guessed labels. No patient grouping is inferred from CAMELYON16 filenames.
Only published MD5 values are treated as source checksums; S3 ETags are retained
as remote identity metadata and are not assumed to be MD5 values.

## KPMP: Atlas export and clinical tables

Filter the Atlas repository to the desired files and download its CSV export.
Set `repository_csv` in `examples/kpmp.json` to that file. Public light-microscopic
WSIs are selected; controlled assets are excluded from download. Clinical files
present in that export are included unless `include_clinical=false`.

```bash
python -m openhisto.default_datasets discover KPMP \
  --selection src/openhisto/default_datasets/examples/kpmp.json --plan kpmp-plan.json
python -m openhisto.default_datasets download kpmp-plan.json --root /path/to/data/kpmp
```

The export needs Access, Participant ID, File Name, Internal Package ID and
Experimental Strategy. File UUID, Data Category, Data Format and Workflow Type
are retained when present. Re-export if required columns were hidden. Full
UUID-prefixed filenames are used for download URLs. Human-readable Size cells
are retained as source metadata; they are not treated as exact byte counts.

Optional selection fields include `participant_ids`, `file_ids`,
`clinical_path`, and `clinical_column_mapping`. A clinical table may be CSV,
JSON row objects, JSONL, or XLSX. The default participant column is `Participant ID`
or `participant_id`; use `{"subject_id": "Your source column"}` for an explicit
mapping. Published clinical files in other formats are preserved but require
conversion/import before linked manifest generation. Public binned values such
as age and eGFR remain unchanged.

For approved local files, use the same Atlas export. Files are matched by their
full original filenames or their explicit planned paths:

```bash
python -m openhisto.default_datasets import-local KPMP \
  --root /path/to/existing/kpmp --metadata /path/to/atlas-export.csv
```

An `--options` JSON file can additionally provide `clinical_path` and the column
mapping. No slide copies are made.

## BRACS: approved local import

After obtaining BRACS through its official portal, keep SVS files under a dataset
root and point the importer to the supplied workbook:

```bash
python -m openhisto.default_datasets import-local BRACS \
  --root /path/to/existing/bracs --metadata /path/to/bracs-summary.xlsx
```

The importer uses `WSI_Information` when present, or the only sheet in a workbook.
For other layouts, pass `--options src/openhisto/default_datasets/examples/bracs-options.json`
after adapting the directory, sheet and headers to your workbook. Canonical
mapping keys are `slide_id`, `label`, `subject_id`, and `split`. Known WSI headers
are recognized conservatively; unknown identifiers require an explicit mapping.

Missing metadata, missing slides, duplicate stems and conflicting labels/splits/
subjects are retained and flagged. Unknown columns and every metadata row remain
in source records. No arbitrary first row resolves a conflict. Official split
strings are preserved rather than reassigned. Native WSI labels are separate from
ROI labels; ROI PNGs are not imported.

TCGA/CAMELYON16 local imports accept a saved acquisition plan instead of a
workbook. They match the planned paths or unique original filenames under the
root, retaining the plan's explicit metadata links:

```bash
python -m openhisto.default_datasets import-local TCGA \
  --root /path/to/existing/tcga --metadata tcga-plan.json
```

## Python API

```python
from openhisto.default_datasets import discover, download, import_local, verify, build_manifest

plan = discover("TCGA", {"projects": ["TCGA-KIRC"], "limit": 5})
print(plan.expected_bytes, plan.unknown_size_count)
plan.save("tcga-plan.json")
report = download(plan, "/path/to/data/tcga-kirc")
print(report.to_dict())

# These are offline operations:
verification = verify("/path/to/data/tcga-kirc")
manifest_path = build_manifest("/path/to/data/tcga-kirc")
local_report = import_local(
    "BRACS",
    "/path/to/existing/bracs",
    {
        "path": "/path/to/bracs-summary.xlsx",
        "slides_dir": "WSIs",
    },
)
```

Discovery also accepts an injectable requests-compatible session for testing.
Downloads accept `retries` (additional attempts, default 3), `timeout` (default
connect/read 10/60 seconds), and `chunk_size` (default 1 MiB). CLI successful
operations print aggregate JSON; exit 1 indicates failed/missing assets and
exit 2 indicates an invalid command/configuration or provider request error.

## Outputs and integrity

| Output | Purpose |
| --- | --- |
| `acquisition_plan.json` | Selection, durable source IDs, original metadata and provenance |
| `acquisition.sqlite` | One shared transfer/verification ledger |
| `slides/` | Downloaded native images; local imports keep existing paths |
| `manifest.jsonl` | Complete common per-slide records |
| `slides.csv` | Readable summary; list/dictionary cells use JSON |
| `clinical.jsonl` | Original linked clinical entities without arbitrary flattening |
| `source_metadata/records.jsonl` | Preserved provider rows/objects, including extra fields |
| `source_metadata/imports/` | Content-addressed copies of imported metadata tables, not slides |
| `provenance.json` | Selection, sources, release information, timestamps and counts |

Each manifest slide has source and namespaced identifiers, supplied subject/
sample associations, root-relative local path, native format, status, expected/
actual size, checksums, labels and their status, original split, clinical
references/availability, optional source magnification and annotation references.
Subject/group identifiers exist only when the source supplies them. Unknown
magnification stays null. Identical clinical records from several locations share
one reference while retaining all provenance sources. Conflicting content under
the same source key is retained with distinct references.

Integrity levels are explicit:

- `checksum_verified`: downloaded/local bytes match a published source checksum.
- `size_verified`: the size matches a declared source/HTTP size, with a local SHA-256 baseline.
- `local_hash_only`: a local SHA-256 baseline exists without an external size/checksum guarantee.
- `failed` / `missing`: verification failed or a file is absent.

Reruns rehash existing payloads before reuse. Previously recorded hashes remain
the baseline on verification failure. Changing an asset's URL, path, expected
size or checksum invalidates stale transfer state. Files are streamed to one
`.part` file per active asset. Resume uses validated ranges and remote validators;
servers that ignore ranges or cannot supply a usable validator trigger a safe
restart. Completed payloads are published atomically. Interrupted partial files
remain available for the next run. Source versions are not silently upgraded.

Use one writer per dataset root. Concurrent writers, authenticated acquisition,
archive unpacking, calibrated clinical harmonization, survival endpoint derivation,
and automatic patch extraction are outside this version.

## Bridge to patch extraction

```python
import hashlib
import json
from pathlib import Path
from openhisto.patch_extraction import ExtractionConfig, extract_grid

root = Path("/path/to/data/tcga-kirc")
for line in (root / "manifest.jsonl").open():
    slide = json.loads(line)
    if slide["status"] != "complete":
        continue
    extract_grid(
        root / slide["local_path"],
        root
        / "patches"
        / (hashlib.sha256(slide["namespaced_slide_id"].encode()).hexdigest() + ".h5"),
        ExtractionConfig(source_mag=slide["source_mag"]),
    )
```

The hashed output name is filesystem-safe; keep a separate slide-to-patch index
if you run this bridge. The extractor uses scanner objective-power metadata when source
magnification is unknown; supply a known override when that metadata is absent.
Keep clinical labels/identifiers in the dataset manifest rather than copying
them into patch archives.

## Tests and validation limits

```bash
python -m unittest discover -s src/openhisto/default_datasets/tests -t src -v
```

Tests use generated provider responses, generated workbooks, synthetic slide
bytes and a local HTTP server. Coverage includes pagination, selection limits,
clinical associations, official labels/splits, conflicts, missing records,
interruptions, ranges, changed validators, retries, checksum failures, reruns,
local registration, path protection and a manifest-to-patch-extractor integration.
No real slide or clinical record is required. The XLSX test skips when openpyxl
is absent. GDC's schema was checked separately; real dataset payloads and all
provider release layouts have not been exercised end to end.
