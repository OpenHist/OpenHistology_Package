"""Offline HIPT model construction with explicit native checkpoint contracts."""

from pathlib import Path

import torch

from .loader import load_native


def module():
    return load_native(
        "hipt",
        "models.model_hierarchical_mil",
        roots=("2-Weakly-Supervised-Subtyping", "HIPT_4K"),
        replacements={
            "2-Weakly-Supervised-Subtyping/models/model_hierarchical_mil.py": [
                (
                    "sys.path.append('../HIPT_4K/')",
                    "# Imports handled by private repository loader",
                ),
                (
                    "        import pdb\n        pdb.set_trace()",
                    "        # Interactive debugger removed",
                ),
            ],
            "HIPT_4K/vision_transformer4k.py": [
                ('print("# of Patches:", num_patches)', "# Model construction is quiet"),
            ],
        },
    )


def load_weights(model, path, *, teacher=False):
    """Fail for absent/mismatched weights; never download or silently randomize."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Requested HIPT checkpoint does not exist: {path}")
    state = torch.load(path, map_location="cpu", weights_only=True)
    if teacher:
        state = state.get("teacher", state)
        state = {k.removeprefix("module.").removeprefix("backbone."): v for k, v in state.items()}
    elif isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
