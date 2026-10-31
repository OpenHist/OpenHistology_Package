# Pathology patch descriptions

`openhisto.vlas` adds prompted text generation to the OpenHisto workflow. It reads
the same indexed HDF5 patch archives or CSV/JSONL patch manifests as
`feature_encoder`, preserves their patch keys and geometry, and produces one
description for each selected patch and model configuration.

The core package requires Python 3.12+. Isolated model environments use the
Python versions specified below, including Python 3.10 for legacy backends.
Run the commands below from the repository root.

## Start with Patho-R1

Use a small core environment for reading patches/results, and a separate model
environment on a compatible NVIDIA GPU host. Core and protocol tests have also
been validated on macOS. The model environment does not need OpenHisto installed:
the runner starts its worker with only the `openhisto` package added to that
interpreter's imports.

```bash
python3.12 -m venv .venv-vlas
.venv-vlas/bin/python -m pip install -e ".[vlas]"

# Run on Linux with a CUDA-compatible PyTorch installation:
python3.10 -m venv .venv-patho-r1
.venv-patho-r1/bin/python -m pip install -r src/openhisto/vlas/requirements-patho-r1.txt

.venv-vlas/bin/python -m openhisto.vlas models
.venv-vlas/bin/python -m openhisto.vlas models patho_r1_7b
.venv-vlas/bin/python -m openhisto.vlas prepare patho_r1_7b \
  --python-executable .venv-patho-r1/bin/python
```

`models` inspects static provenance and access information without Torch imports,
credential lookup, or downloads. `prepare` explicitly loads/downloads just the
selected model and reports its actual runtime. Existing Hugging Face login or
`HF_TOKEN` supplies authorization when needed. Never put token values in jobs.
Choose a PyTorch wheel compatible with the server's driver/CUDA runtime.

Copy [examples/patho-r1.json](examples/patho-r1.json), set your archive, stable
slide/source IDs, backend Python, output directory, and prompt, then run:

```bash
.venv-vlas/bin/python -m openhisto.vlas describe --job src/openhisto/vlas/examples/patho-r1.json
.venv-vlas/bin/python -m openhisto.vlas status descriptions/synthetic-slide
```

Job paths are relative to the job file. `python_executable` may be an explicit
path or a command on `PATH`; an omitted executable uses the runner's Python.
An explicit `checkpoint` and `options.*_checkpoint` are **local artifacts**.
Omit `checkpoint` to use a catalogue model's pinned Hugging Face revision.
Remote revision overrides must be immutable commit IDs.

Defaults: CUDA, batch size 1, automatic BF16/FP16 where supported by the selected
backend, greedy generation, 2048 new tokens, seed zero, no quantization. Native
legacy backends default to FP16. Set `max_new_tokens` higher if answers truncate.
`workers` parallelizes image decoding; it does not create additional model copies.
`timeout` and `startup_timeout` bound worker response waits, in seconds.

## Models, runtimes, and access

| ID | Native inference path | State |
| --- | --- | --- |
| `patho_r1_7b` | Qwen2.5-VL processor, image preparation and conditional generation | Implemented; real checkpoint/CUDA validation required |
| `pathgen_llava` | **13B** Vicuna/LLaVA, PathGen-CLIP, explicit `llava_v1` conversation | Implemented; separate native environment |
| `quilt_llava_7b` | Official Quilt LLaVA implementation; released CLIP-L/14-336 configuration | Implemented; separate native environment |
| `pa_llava` | PLIP, native AnyRes processing, QFormer projector, Llama-3 | Experimental; prepared stage-2 artifacts and GPU smoke required |
| `pathchat_plus` | Authors' complete checkpoint and matching implementation required | Unavailable; selecting it explains the access requirement |

The catalogue pins upstream/model revisions and records access terms and needed
artifacts. Runtime metadata records actual preprocessing, precision, libraries,
and resolved checkpoint/code identities. Installed native code uses its own
distribution's VCS provenance; source checkouts use their tracked Git revision.
An enclosing application's Git commit is never used for virtualenv code.
A published architecture alone is not
a substitute for its trained checkpoint.

Patho-R1's [pinned model card](https://huggingface.co/WenchuanZhang/Patho-R1-7B/blob/7a69eb299bde72a4c3b8ec26fe0b17515346ef73/README.md)
adds research-use conditions, including prior written approval for generating
datasets from its outputs. Obtain that approval if using saved descriptions as
a dataset. Public download availability does not waive those conditions.
PathGen, Quilt, and PA checkpoints also retain their respective model and
base-model terms; inspect `models MODEL` and the linked upstream model cards.
[PathChat+ requires direct access from its authors](https://github.com/mahmoodlab/viper/blob/main/docs/REPRODUCIBILITY.md#limited-access-models).

### Legacy environments

Install only the requirements for the chosen backend, each in its own Python
3.10 environment. PathGen and Quilt both import a package named `llava` and use
incompatible Torch/Transformers versions.

```bash
python3.10 -m venv .venv-pathgen
.venv-pathgen/bin/python -m pip install -r src/openhisto/vlas/requirements-pathgen.txt
python3.10 -m venv .venv-quilt
.venv-quilt/bin/python -m pip install -r src/openhisto/vlas/requirements-quilt.txt
```

The requirement files install the pinned native source. For a maintained local
checkout, set `options.upstream_path` and preserve its revision; source contents
are hashed in the run identity. PathGen needs its associated local visual encoder
at `options.vision_checkpoint`. Download the [language/connector checkpoint](https://huggingface.co/jamessyx/PathGen-LLaVA)
and [PathGen visual encoder](https://huggingface.co/jamessyx/pathgenclip-vit-large-patch14-hf)
at the revisions shown in the catalogue/backend before an offline job. The
adapter overrides its vision path in memory and retains native preprocessing.
Quilt follows its [released checkpoint](https://huggingface.co/wisdomik/Quilt-Llava-v1.5-7b),
including the configured visual encoder; its separate encoder must also be
cached for offline preparation. See [examples/legacy.json](examples/legacy.json).

### PA-LLaVA preparation

Use [the authors' conversion workflow](https://github.com/ddw2AIGROUP2CQUPT/PA-LLaVA)
before inference. Raw `.pth` training releases cannot be passed to a generic
Transformers loader and are never converted implicitly by `describe`.

1. Obtain the pinned PA-LLaVA source revision shown by `models pa_llava`, create
   its dedicated Python 3.10 environment, and install
   `src/openhisto/vlas/requirements-pallava.txt`. Set `upstream_path` to that checkout, which
   contains `xtuner_add/pallava`.
2. Extract the instruction-tuning release. Set these absolute paths, and run
   conversion from a preparation directory containing the extracted PLIP assets
   under `./plip/config.json` and `./plip/plip_80w_196token.bin` (a symlink to the
   PLIP directory also works). The native training constructor requires the
   literal `plip` value and reads that relative directory:

   ```bash
   PA_UPSTREAM=/absolute/path/to/vendor/PA-LLaVA
   PA_PYTHON=/absolute/path/to/.venv-pallava/bin/python
   PA_BASE=/absolute/path/to/Meta-Llama-3-8B-Instruct
   PA_STAGE2_PTH=/absolute/path/to/extracted/instruction_tuning_weight.pth
   PA_PREPARED=/absolute/path/to/pa-stage2-prepared
   PYTHONPATH="$PA_UPSTREAM/xtuner_add${PYTHONPATH:+:$PYTHONPATH}" \
     "$PA_PYTHON" "$PA_UPSTREAM/xtuner_add/pallava/model_converters/pth_to_hf.py" \
     "$PA_UPSTREAM/pallava_instruction_tuning.py" \
     "$PA_STAGE2_PTH" "$PA_PREPARED" \
     --cfg-options model.visual_encoder=plip \
       model.llm.pretrained_model_name_or_path="$PA_BASE" \
       tokenizer.pretrained_model_name_or_path="$PA_BASE" \
       model.use_activation_checkpointing=False
   ```

   Use the actual extracted checkpoint filename in `PA_STAGE2_PTH`. The prepared
   output must include learned `projector/`
   weights/config and, for the unmerged base workflow, `llm_adapter/` PEFT
   weights/config. This adapter requires the stage-2 PEFT adapter and its matching
   unmerged base model; a plain Llama model or merged-only export is insufficient.
3. Obtain the authorized local `Meta-Llama-3-8B-Instruct` model/tokenizer and the
   authors' extracted PLIP directory containing `config.json` and
   `plip_80w_196token.bin`. Point `base_checkpoint` and `vision_checkpoint` to them.
4. Use [examples/pallava.json](examples/pallava.json). Its `checkpoint` is the
   converted root; `projector_checkpoint` and `adapter_checkpoint` can override
   the corresponding child directories. Initial online preparation also caches
   CLIP-B/16 preprocessing and BERT configuration assets required by native code.

PA validates learned weights and feature dimensions and preserves the native
global/local image feature flow. Its environment/conversion still needs testing
with the actual author artifacts on the target GPU.

## Python API and feature alignment

```python
from openhisto.vlas import SourceSpec, ModelSpec, DescribeConfig, describe_slide

sources = [SourceSpec("patches/slide-20x.h5", source_id="grid-20x")]
report = describe_slide(
    inputs=sources,
    output="descriptions/slide",
    models=[ModelSpec("patho_r1_7b", python_executable=".venv-patho-r1/bin/python")],
    prompt="Describe the visible tissue architecture and nuclear morphology.",
    config=DescribeConfig(slide_id="slide-a", batch_size=1),
)
print(report.to_dict())
```

Use **the same `slide_id`, `source_id`, and patch sources** for `encode_slide`
and `describe_slide`. Existing feature `SourceSpec` objects are accepted
structurally. No feature model runs during description generation.

```python
import pandas as pd
from openhisto.feature_encoder import FeatureReader
from openhisto.vlas import DescriptionReader

with DescriptionReader("descriptions/slide") as descriptions:
    run_key = descriptions.list_runs()[0]["run_key"]  # choose one model/prompt run
    text = pd.DataFrame(descriptions.records(run_key)).set_index("patch_key")
with FeatureReader("features/slide.h5ad") as features:
    patch_metadata = features.patch_metadata()
    joined = patch_metadata.join(text[["description", "status"]], how="left")
    vectors = features.features("phikon", mag=20, rows=text.index.tolist())
```

Filter description keys to the desired magnification before requesting vectors
when combining magnifications. A key is the exact feature observation hash of
`[slide_id, source_id, patch_id]`; coordinates and row positions are not join keys.
`config.patch_keys` optionally selects a subset using these observation keys.
Unknown keys fail rather than being silently dropped.

Archives retain original patch indices and all supplied geometry. CSV/JSONL
patch manifests require `patch_id`, `image_path`, `slide_id`, and `mag` (or
`magnification`), with slide ID/magnification alternatively supplied explicitly.
Image paths resolve against the manifest. Optional geometry and extra metadata
follow the feature encoder's format. Missing geometry is JSON `null`; alpha is
composited on white; high-bit-depth images require explicit conversion.

The `manifest.jsonl` written by `default_datasets` contains **whole slides**, not patches. Run
`patch_extraction` first and retain a slide-to-archive mapping. For new workflows,
use the dataset's `namespaced_slide_id` consistently in both feature and VLA jobs;
for existing features, keep the IDs already used by those feature jobs. The
archive itself deliberately omits slide labels and logical IDs.

## Results and recovery

Each output directory contains:

| File | Contents |
| --- | --- |
| `metadata.json` | Stable patch metadata, exact prompts, model/run configuration, runtime provenance, pending failures and interruptions |
| `descriptions.jsonl` | Authoritative, committed successful/truncated records, unique on `(patch_key, run_key)` |
| `descriptions.csv` | Atomic readable export of committed records |

Failed/unreturned batches remain pending and are reported in run metadata; they
are not committed as empty successful descriptions. Rerunning retries only
missing results. Truncated descriptions are retained and reused; increase
`max_new_tokens` to create a new run if needed. Patho returns the final answer
portion rather than model reasoning; an absent final answer fails the batch.

`DescriptionReader.records()` materializes requested results; use
`iter_records()` to stream text for large outputs. Patch indexing and completion
sets use memory proportional to patch/result count; decoded images and worker
requests are bounded by batch size. Models run sequentially and stay loaded
between batches. Their subprocess protocol transfers lossless PNGs in memory,
without creating per-patch image files.

Run identity includes the prompt, model/source revisions, local artifact hashes,
preprocessing and generation configuration. Changing any creates a separate run;
batch size, decoding workers, timeouts, cache location, and offline mode do not.
Hashing large local checkpoints adds an input-integrity scan to each job. Adding
missing results to an existing run checks its prepared runtime metadata; an
environment mismatch stops before mixing outputs. Fully cached jobs do not load
models. Every supplied existing patch is checked for changed pixels or geometry,
including on cached runs.

For an unexpected worker exit or timeout, enable `--worker-diagnostics` on
`prepare` or `describe` to include a bounded, redacted stderr tail in CLI failure
output. JSON jobs and Python callers can set `DescribeConfig(worker_diagnostics=True)`.
Worker failures report their exit status when available. In Python, catch
`VLAError` (including `DescriptionFailure`) and inspect `error.worker_diagnostics`
for `exit_status` and the opted-in `stderr_tail`. Diagnostics remain separate
from error messages and saved result metadata; changing this setting preserves
description run identity. Redaction covers credential values from the runtime
environment and recognized Hugging Face token strings, so review diagnostics
before sharing logs that may contain other private paths or data.

JSONL batches flush before completion is recognized. A torn final unterminated
line is discarded under the writer lock and retried; corrupt complete lines are
errors. CSV is a snapshot and can lag JSONL after a hard crash; rerunning refreshes
it. One sibling `OUTPUT.lock` prevents concurrent writers. After a process crash,
verify its recorded PID is no longer active before removing a stale lock.
Atomic metadata/CSV writes and ledger fsyncs use the local filesystem's guarantees;
use one writer on shared storage too.

`completed` counts newly saved nontruncated answers, `truncated` counts newly
saved truncated answers, `reused` counts matching saved answers, and `pending`
counts missing selected patch/run pairs. CLI exit codes are 0 success, 1 generation
failure with a report, 2 invalid configuration/preparation, and 130 interruption.
Result directories can contain several models and prompts; always select a run
before joining one description column to feature rows.

## Validation

```bash
python -m unittest discover -s src/openhisto/vlas/tests -t src -v
```

The suite uses synthetic archives/images, fake model handles, mocked native
loading/generation, and actual isolated subprocess protocol checks. Optional
feature-core tests exercise real AnnData writes and `FeatureReader` joins; they
skip if feature encoding dependencies are absent. No automated test downloads
weights or submits pathology images to external services.

For actual checkpoint acceptance on a compatible NVIDIA GPU host, use a job
containing at least two patches and exactly one model:

```bash
python -m openhisto.vlas.smoke --job my-gpu-job.json --report gpu-smoke.json
python -m openhisto.vlas.smoke --job my-gpu-job.json \
  --reference native-reference.jsonl --report gpu-acceptance.json
```

The check runs A, B, A through one persistent model, reloads B offline, checks
stable independent responses and complete answers, then compares against the
optional native reference. Generate that reference independently using the
official model's inference path with identical prompt/template, revision,
preprocessing, precision and generation settings. Each line has `patch_key` and
`description` (Patho's final answer). The report says `verified: true` only when
that comparison passes. Exact equality may expose nondeterministic GPU kernels;
investigate differences rather than silently accepting them. PA remains
experimental until this real-artifact test passes.
