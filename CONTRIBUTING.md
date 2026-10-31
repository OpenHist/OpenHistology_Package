# Contributing to OpenHisto

We welcome bug fixes, documentation improvements, new adapters and test cases.
Please describe the concrete workflow you want to improve and keep changes
focused enough to review. For a new model/backend, include its native input
contract, source revision, dependencies, access requirements and license.

## Set up a checkout

Use Python 3.12+ and an editable installation. Install only the module extras
needed for your change; the [README](README.md#install-from-source) lists their
names. VLA model workers have separate environments described in their guide.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[patch-extraction-opencv]"
python -m pip install ruff==0.17.3
```

Create a branch for your change and explain the problem, resulting behavior and
validation in the pull request. Small synthetic reproductions help us review
bugs without requiring a dataset download or model account.

## Test the affected module

Run the relevant suite from the repository root after installing its extras:

```bash
python -m unittest discover -s src/openhisto/patch_extraction/tests -t src -v
```

The same pattern applies to `default_datasets`, `feature_encoder`,
`stain_transfer`, `segmentation`, `vlas`, `mil/heads`, `mil/classification` and
`mil/survival`. MIL backbone family tests live under
`src/openhisto/mil/models/code_bases/tests`.

Our module tests use controlled inputs and mocks. Installing dependencies may
require network access, but functional tests do not require patient data, paid
API calls or real pretrained checkpoint downloads. State which checks you ran
and any dependency-related skips. Keep actual checkpoint/GPU acceptance separate
from synthetic correctness evidence; do not replace a missing backend with
unlabelled fabricated output.

For behavior changes, add a regression test that demonstrates the bug or checks
a meaningful invariant. Keep schema, geometry, model-run identity and failure/
publication behavior consistent across Python and CLI entry points. Update the
affected module guide and examples when their contract changes.

## Formatting and lint

We use Ruff 0.17.3 for maintained Python code and scripts:

```bash
ruff format --check src/openhisto scripts
ruff check src/openhisto scripts
```

Apply formatting with `ruff format src/openhisto scripts`, then rerun the checks.
The configuration excludes pristine upstream snapshots and copied GAN network
definitions. Do not reformat or remove their notices as part of an unrelated
change.

## Build the package

Build and inspect both distribution formats from the repository root:

```bash
python -m pip install build packaging
python -m build
python scripts/check-distribution.py dist/*.whl dist/*.tar.gz
```

The distribution check verifies metadata, license inclusion, Python syntax and
artifact contents. These commands build local files; they do not publish a release.

## MIL sources and validation reports

We preserve pinned upstream source snapshots, licenses and provenance under
`src/openhisto/mil/models/code_bases`. A fresh clone already includes the source
snapshots; no submodule initialization is needed. Put compatibility changes in
our adapter/compatibility layer where practical. Changes to source selection or
pins must update the corresponding recorded inventories and documentation.
Keep [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) aligned with source or
license-notice changes, preserving recorded gaps rather than guessing a license.

MIL coverage records include implementation fingerprints. After changing MIL
implementation bytes, format first and run both report writers in an environment
with all required backends installed:

```bash
python -m pip install -e ".[mil-backbones,mil-classical-all]"
python -m openhisto.mil.models.code_bases.validate --write-report
python -m openhisto.mil.heads.validate --write-report
```

These commands execute synthetic CPU validation and refresh their reports only
on full success. Include the resulting `COVERAGE.md` and `coverage.json` changes
from both module directories when they change. Do not edit results by hand or
describe skipped/failed cases as validated. See the
[backbone guide](src/openhisto/mil/models/code_bases/README.md#sources-provenance-and-verification)
and [head guide](src/openhisto/mil/heads/README.md) for native dependencies and
environment details.

## Reproductions, data and attribution

Use synthetic examples in issues, tests and pull requests. Remove credentials,
patient identifiers, clinical records and private paths from logs before sharing
them; a small shape/dtype/configuration example is usually enough. Do not attach
model weights or slide datasets to a bug report.

Preserve upstream authorship, citations, license files and source URLs when
adding or adapting code. Explain modifications and their source revision.
OpenHisto's original code is MIT, copyright 2026 Abteen Arab; third-party
implementations and external artifacts keep their own terms. Our
[license statement](README.md#license-and-attribution) describes this boundary.
