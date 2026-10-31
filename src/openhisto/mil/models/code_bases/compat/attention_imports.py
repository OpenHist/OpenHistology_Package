"""Narrow import overlays for native attention/statistical MIL repositories.

Upstream files and license notices remain untouched. These overlays remove only
unused eager imports; no attention, masking, pooling, or EM math is replaced.
"""

from .loader import repository


def replacements(slug):
    if slug == "asmil":
        return {
            "architecture/mytransformer.py": [
                (
                    "from modules.emb_position import *",
                    "# Adapter: unused missing positional import omitted.",
                ),
            ]
        }
    if slug == "panther":
        # This initializer eagerly imports every unrelated supervised/survival
        # model. The requested PANTHER EM layers do not need those imports.
        path = repository(slug) / "src/mil_models/__init__.py"
        return {
            "src/mil_models/__init__.py": [
                (
                    path.read_text(encoding="utf-8"),
                    "# Adapter: import only the requested native model module.\n",
                ),
            ]
        }
    return {}
