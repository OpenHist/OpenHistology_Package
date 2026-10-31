# Feature encoding

Encode patch archives or explicit patch manifests into **one standard AnnData `.h5ad` per slide**, with all selected models and extraction magnifications in that file. Rerunning a job reuses completed features, appends new patch identities and computes missing features for the models you select. No model runs unless selected.

## Environment

Use Python 3.12+ in a separate virtual environment. From the repository root:

```bash
python3.12 -m venv .venv-features
source .venv-features/bin/activate
# Install only the backend(s) you need; each includes the feature-encoder core.
python -m pip install -e ".[feature-encoder-local]"
# python -m pip install -e ".[feature-encoder-google]"
# python -m pip install -e ".[feature-encoder-cohere]"
# python -m pip install -e ".[feature-encoder-aws]"
```

The tested core uses AnnData `0.13.3.post0` (the published patch of 0.13.3), NumPy 2.2.6, pandas 3.1.0 and h5py 3.13.0. Local GPU packages must also match your CUDA/runtime installation. Model libraries and cloud SDKs are imported only for selected, uncached work; core AnnData dependencies are required to import this module.

## Models and access

```bash
python -m openhisto.feature_encoder models
python -m openhisto.feature_encoder models conch15_titan
python -m openhisto.feature_encoder access uni_v1 --prompt
python -m openhisto.feature_encoder prepare phikon
```

`models` reports dimensions, dependencies, preprocessing, terms/approval links, authentication and an unverified local cache inspection. It does not download weights or look up credentials. `access` performs a lightweight check; cloud credential presence does not establish API entitlement. `prepare` explicitly loads/downloads the selected model or initializes its client without image inference.

| Profile ID | Default dimensions | Representation / access |
|---|---:|---|
| `phikon` | 768 | Official checkpoint processor, CLS; public Hugging Face weights |
| `phikon_v2` | 1024 | Official checkpoint processor, CLS; public Hugging Face weights |
| `uni_v1` | 1024 | Checkpoint timm transform, CLS; Hugging Face approval required |
| `conch15_titan` | 768 | TITAN CONCH v1.5 attentional pooling, 448-pixel transform; Hugging Face approval required |
| `gemini_embedding_2` | 3072 | One RGB image per request; Gemini API key |
| `cohere_embed_5_pro` | 2048 | Separate image input objects; Cohere API key |
| `cohere_embed_5_fast` | 2048 | Separate image input objects; Cohere API key |
| `nova_multimodal` | 3072 | One image per `SINGLE_EMBEDDING` request; AWS credential chain |

Local defaults pin immutable Hub commits. `models MODEL` shows those commits and linked terms. CONCH v1.5 uses TITAN's patch encoder and preprocessing; it does not substitute the original CONCH image embedding or a TITAN slide embedding. Supplying a token does **not** grant gated-model approval: request access at the descriptor's approval URL using the same Hugging Face account.

Credentials resolve from the named environment variables (`HF_TOKEN`, `GEMINI_API_KEY`, `COHERE_API_KEY`), an existing Hugging Face login, or opt-in hidden input with `--prompt`. Do not put credential values in job JSON. Python callers may pass an `AuthConfig` with in-memory credentials. Newly entered values are neither persisted nor included in run metadata/fingerprints. Complete cached jobs and complete gated checkpoint bundles do not require credential resolution. AWS uses the SDK's normal profile, role and environment chain; `aws_profile` and `aws_region` are optional.

Cloud inference sends each selected patch to its provider and can incur charges. Each patch receives its own vector; the pipeline rejects fused responses, invalid shapes and nonfinite vectors. Clients stay alive across bounded batches. Transient retries respect short retry hints; authentication failures and malformed responses stop immediately. No real checkpoint downloads or paid image requests were used in the automated tests.

## Encode a slide

Copy [examples/local.json](examples/local.json), replace its synthetic paths/slide ID, then run:

```bash
python -m openhisto.feature_encoder encode --job src/openhisto/feature_encoder/examples/local.json
python -m openhisto.feature_encoder status features/synthetic-slide.h5ad
python -m openhisto.feature_encoder inspect features/synthetic-slide.h5ad --metadata
```

Paths in a job are relative to the job file. Use one stable `slide_id` when combining archives. Give each archive/extraction its own stable `source_id`; keep it unchanged if the file is renamed or moved. The archive's original patch index is retained. Extraction magnification comes from archive metadata, or an explicit `mag` when absent; conflicting values fail. Magnification is never inferred from a filename.

```python
from openhisto.feature_encoder import EncodeConfig, EncoderSpec, SourceSpec, encode_slide

report = encode_slide(
    inputs=[
        SourceSpec("patches/slide-20x.h5", source_id="grid-20x"),
        SourceSpec("patches/slide-10x.h5", source_id="grid-10x"),
    ],
    output="features/slide.h5ad",
    models=[EncoderSpec("phikon"), EncoderSpec("phikon_v2")],
    config=EncodeConfig(slide_id="synthetic-slide", batch_size=32),
)
print(report.to_dict())
```

Defaults: batch size 32, automatic CUDA/MPS/CPU selection, evaluation/inference mode, supported CUDA mixed precision, native model normalization, float32 features and gzip level 4. `normalize=True` explicitly adds L2 normalization. `workers` controls parallel patch-image loading; `cloud_concurrency` bounds independent Gemini/Nova requests. Changing execution precision/device intent, revision, dimensions, normalization or model options creates a separate run identity. Changing batch size, compression or worker counts does not invalidate completed vectors.

If an existing run needs more rows but its resolved runtime, libraries or provider region no longer match, the update stops to prevent mixing representations. Restore the original environment or choose an explicit new configuration/`options.cache_tag`. Fully cached runs remain usable without preparing a backend.

For API models, start from [examples/cloud.json](examples/cloud.json). Supply only the models you want; that example contains one. Backend-specific options are validated rather than silently ignored. Cloud model revisions are recorded when available and explicitly unknown otherwise. To intentionally refresh a mutable cloud deployment, use a new `options.cache_tag`; cached features are never silently upgraded.

### Explicit patch manifests

CSV and JSONL rows require `patch_id`, `image_path`, `slide_id` and `mag` (also accepts `magnification`). A common slide ID and missing magnification may instead be supplied explicitly in configuration/`SourceSpec`. Paths are relative to the manifest. PNG/JPEG and other Pillow-readable 8-bit images are accepted, converted to RGB, with transparency composited on white. High-bit-depth images require an explicit conversion before encoding.

```json
{"patch_id":"patch-a","image_path":"images/a.png","slide_id":"synthetic-slide","mag":20,"x_level0":100,"y_level0":200,"width_level0":512,"height_level0":512,"valid_bounds":[0,0,512,512],"quality":"example"}
```

Optional geometry fields are `x_level0`, `y_level0`, `width_level0`, `height_level0`, `source_mag`, `patch_index`, and output-pixel `valid_bounds` (JSONL) or `valid_x0`, `valid_y0`, `valid_x1`, `valid_y1`. Missing coordinates/footprints remain NaN; original patch index is -1 if absent. The supplied row and extra metadata are preserved in `obs.metadata_json`. Reordering manifest rows does not reorder existing features. Patch IDs must remain stable, and no two patches in one source may share an identity.

## Incremental output and recovery

Existing observation order stays fixed; new identities append. Before updating, the pipeline checks decoded RGB hashes and supplied geometry against saved rows. Changed existing patches stop the operation and preserve published output: write a new output for revised source patches. Images are rechecked when loaded for encoding to catch changes during the run. This means even a cached job scans image content, but never loads a model or recomputes matching vectors.

Optional metadata on existing rows follows `EncodeConfig.metadata_policy` (or
`config.metadata_policy` in a job): `"preserve"` keeps the first supplied values
and reports how many rows differ; `"update"` applies corrections; `"error"`
stops when values differ. The default is `"preserve"`. To correct quality or
label fields in a patch manifest, edit the manifest and run:

```bash
python -m openhisto.feature_encoder encode --job job.json --metadata-policy update
```

The CLI flag overrides the job setting. Python callers can pass
`EncodeConfig(metadata_policy="update", slide_id="synthetic-slide")`.
Metadata corrections preserve completed vectors and run identities, without
loading a model or encoding those rows again. Content hashes, source identities,
magnification and geometry remain immutable under every policy. Output reports
include `metadata_updated` and `metadata_preserved`, counting existing patch
rows with differing optional metadata rather than patch/run pairs. Metadata on
omitted rows is retained.

When you select one model after adding patches, other existing models remain pending on eligible new rows. A later job selecting them fills those rows. Include the original patch sources for all rows that need computation; the feature file stores metadata and features, not patch pixels. Omitted old rows remain in the output.

One sibling `SLIDE.partial.h5ad` holds resumable work, and `SLIDE.h5ad.lock` prevents simultaneous writers. Updates temporarily require space for a copy of the published feature file. Feature bytes flush before their completion masks. Successful validation atomically publishes the partial, leaving one feature file; an exception preserves committed partial batches and the prior published file. Rerun the same job to resume. Locks are released on handled errors. After a process crash, inspect the PID in the lock and remove it manually only after establishing that no writer is active. Incompatible partials are retained with an error rather than discarded.

Report `completed` and `reused` count patch/run pairs; `patches` counts global rows. `pending` counts all outstanding eligible patch/run pairs, including unselected existing runs. `failed` is nonzero for an unsuccessful operation. CLI failures return a nonzero status and retain compatible staged work.

## Read features

```python
from openhisto.feature_encoder import FeatureReader

with FeatureReader("features/slide.h5ad") as reader:
    print(reader.list_runs())
    metadata = reader.patch_metadata(mag=20)
    vectors = reader.features("phikon", mag=20, rows=metadata.index[:10])
    feature_metadata = reader.model_metadata("phikon", mag=20)
    for batch, patches in reader.iter_batches("phikon", mag=20, batch_size=128):
        pass
```

Row selectors address the **global** observation axis (positions, IDs, slices or boolean masks); a model query filters them to its magnification while retaining order. Queries require complete eligible rows by default. Use `require_complete=False` to inspect pending NaNs, and use `run_key=` when a model/magnification has multiple configurations. `list_runs()` and `model_metadata()` expose the logical `feats → magnification → model → runs` hierarchy, including feature names/dimensions, checkpoint, transform, pooling, normalization, library versions and aggregate usage. Reader handles are process-local and reopened after pickling; use a separate reader per training worker.

The file layout is standard AnnData:

| Element | Contents |
|---|---|
| `obs` | Stable observation ID, source/slide/patch identities, extraction `mag`, content hash, geometry, valid bounds, supplied metadata, boolean `encoded_<run_key>` columns |
| `obsm[run_key]` | Chunked float32 matrix with the global observation axis; NaN where pending or not applicable |
| `uns["feats"]["20x"][model]["runs"][run_key]` | Run references, feature-axis information and serialized technical metadata |
| `X`, `var` | `X=None`, empty global feature axis |

Different encoders can have different dimensions. The completion mask plus `obs.mag` distinguishes pending rows, other-magnification rows and successfully encoded vectors. Model input resizing/cropping is recorded separately from extraction magnification. AnnData's public element writers create encoded elements, so `anndata.read_h5ad(path)` works normally. Prefer `FeatureReader` for large files: ordinary AnnData backed mode can eagerly load `obsm`, whereas this reader selects HDF5 rows without reading unrelated feature matrices. Metadata uses memory proportional to patch count; decoded images and feature-processing buffers are bounded by batch size, and feature payloads are never loaded wholesale during an update.

Run metadata also records its first preparation time in UTC, Python/core versions and backend runtime details. Aggregate usage accumulates across updates and includes retries; an interrupted cloud batch may need to be requested again if its returned vectors were not committed before interruption.

## Validation

```bash
python -W ignore::FutureWarning -m unittest discover -s src/openhisto/feature_encoder/tests -t src -v
```

The synthetic suite covers AnnData round trips, different magnifications/dimensions, archive and manifest inputs, alignment, no-op caching, appends, pending rows, new runs, source changes, interruption/resume, publication failures, locks, independent reader handles, selective HDF5 reads, authentication redaction and mocked cloud request/response behavior. Optional SDK schema tests run when those SDKs are installed. Real checkpoint execution and paid cloud inference remain opt-in; backend availability, account entitlement and hardware performance require validation in the deployment environment.
