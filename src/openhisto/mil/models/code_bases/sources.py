"""Pinned, code-only Git acquisition. Never fetches LFS objects or datasets."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

from .snapshot import validate_source_tree
from .source_selection import source_exclusions

ROOT = Path(__file__).resolve().parent
REPOSITORIES = {
    "attention_deep_mil": ("AMLab-Amsterdam/AttentionDeepMIL", None),
    "dsmil": ("binli123/dsmil-wsi", None),
    "dtfd_mil": ("hrzhang1123/DTFD-MIL", None),
    "acmil": ("dazhangyu123/ACMIL", None),
    "asmil": ("Linfeng-Ye/ASMIL", None),
    "clam": ("mahmoodlab/CLAM", None),
    "wikg": ("WonderLandxD/WiKG", None),
    "tad_graph": ("HKU-MedAI/TAD-Graph", None),
    "resgat": ("maiziezhoulab/ResGAT", None),
    "patch_gcn": ("mahmoodlab/Patch-GCN", None),
    "slidegraph": ("wenqi006/SlideGraph", None),
    "cs_mil": ("hrlblab/CS-MIL", None),
    "zoommil": ("histocartography/zoommil", None),
    "hag_mil": ("BearCleverProud/HAG-MIL", None),
    "gfr_mil": ("longger-z/GFR-MIL", None),
    "h2_mil": ("lin-lcx/H2-MIL", None),
    "hipt": ("mahmoodlab/HIPT", None),
    "varpool": ("mahmoodlab/varpool", None),
    "panther": ("mahmoodlab/PANTHER", None),
    "transmil": ("szc19990412/TransMIL", None),
    "ilra": ("jinxixiang/low_rank_wsi", None),
    "rrt_mil": ("DearCaat/RRT-MIL", None),
    "dgr_mil": ("ChongQingNoSubway/DGR-MIL", None),
    "caprmil": ("mandlos/CAPRMIL", None),
    "lin_mil": ("charlotterchtr/Lin-MIL", None),
    "mamba_mil": ("isyangshu/MambaMIL", None),
    "nnmil": ("Luoxd1996/nnMIL", None),
    "mhim_mil": ("DearCaat/MHIM-MIL", "master"),
    "mhim_mil_v1": ("DearCaat/MHIM-MIL", "v1"),
}


def _git(*args, cwd=None):
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1", "GIT_TERMINAL_PROMPT": "0"}
    result = subprocess.run(
        ["git", *map(str, args)], cwd=cwd, env=env, text=True, capture_output=True, timeout=300
    )
    if result.returncode:
        raise RuntimeError(f"Git operation failed: {result.stderr.strip()[:600]}")
    return result.stdout.strip()


def _source_file(name, *, slug=None):
    path = Path(name)
    if path.as_posix() in source_exclusions(slug):
        return False
    lower = path.name.lower()
    if path.suffix.lower() == ".json":
        # Explicit configuration only; split/patient/sample JSON is data.
        return any("config" in part.lower() for part in path.parts) or lower in {
            "package.json",
            "pyproject.json",
        }
    return (
        path.suffix.lower()
        in {
            ".py",
            ".pyi",
            ".c",
            ".cc",
            ".cpp",
            ".cu",
            ".cuh",
            ".h",
            ".hpp",
            ".sh",
            ".r",
            ".m",
            ".toml",
            ".cfg",
            ".ini",
            ".yaml",
            ".yml",
            ".json",
            ".md",
            ".rst",
        }
        or lower.startswith(
            (
                "license",
                "licence",
                "copying",
                "notice",
                "readme",
                "requirements",
                "environment",
                "makefile",
            )
        )
        or lower
        in {
            "setup.py",
            "setup.cfg",
            ".gitmodules",
            ".gitattributes",
            ".gitignore",
            "cmakelists.txt",
        }
    )


def acquire_one(slug):
    owner, branch = REPOSITORIES[slug]
    destination = ROOT / "upstream" / slug
    record_path = ROOT / "compat" / "sources" / (slug + ".json")
    pinned = json.loads(record_path.read_text()) if record_path.is_file() else None
    if pinned is not None and pinned["repository"] != owner:
        raise RuntimeError(f"Source catalogue origin changed: {slug}")
    if destination.exists():
        if not record_path.is_file():
            raise RuntimeError(f"Refusing unmanaged existing source directory: {slug}")
        validate_source_tree(ROOT, slug, pinned, directory=destination)
        return pinned
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + slug + "-", dir=destination.parent))
    try:
        command = ["clone", "--depth", "1", "--filter=blob:none", "--no-checkout"]
        if branch:
            command += ["--branch", branch]
        command += ["https://github.com/" + owner + ".git", temporary]
        _git(*command)
        if pinned is not None:
            # A missing checkout is a restoration, never an implicit upgrade.
            _git("fetch", "--depth", "1", "origin", pinned["commit"], cwd=temporary)
            _git("reset", "--soft", pinned["commit"], cwd=temporary)
        commit = _git("rev-parse", "HEAD", cwd=temporary)
        actual_branch = _git("symbolic-ref", "--short", "HEAD", cwd=temporary)
        names = _git("ls-tree", "-r", "--name-only", "HEAD", cwd=temporary).splitlines()
        chosen = [name for name in names if _source_file(name, slug=slug)]
        _git("sparse-checkout", "init", "--no-cone", cwd=temporary)
        sparse = temporary / ".git" / "info" / "sparse-checkout"
        sparse.write_text("\n".join("/" + name for name in chosen) + "\n")
        _git("checkout", "--force", cwd=temporary)
        licenses = [
            name
            for name in chosen
            if Path(name).name.lower().startswith(("license", "licence", "copying", "notice"))
        ]
        record = {
            "repository": owner,
            "url": "https://github.com/" + owner,
            "branch": actual_branch,
            "commit": commit,
            "checkout": "upstream/" + slug,
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "code_only": True,
            "lfs_downloaded": False,
            "datasets_downloaded": False,
            "license_files": licenses,
            "source_files": len(chosen),
        }
        if source_exclusions(slug):
            record["excluded_files"] = source_exclusions(slug)
        validate_source_tree(ROOT, slug, record, directory=temporary)
        temporary.rename(destination)
        record_path.parent.mkdir(parents=True, exist_ok=True)
        record_path.write_text(json.dumps(record, indent=2) + "\n")
        return record
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def acquire_all(workers=4):
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for slug, record in zip(REPOSITORIES, executor.map(acquire_one, REPOSITORIES)):
            print(slug + " " + record["commit"], flush=True)


if __name__ == "__main__":
    acquire_all()
