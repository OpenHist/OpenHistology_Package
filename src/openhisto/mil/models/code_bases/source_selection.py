"""Explicit omissions from code-only snapshots of otherwise unchanged pins."""

SOURCE_EXCLUSIONS = {
    "cs_mil": {
        "CS-MIL_Docker/src/models/swav.py": "Unused, incomplete upstream SwAV reference with an empty forward body; "
        "not imported by the supported CS-MIL adapter.",
    },
}


def source_exclusions(slug):
    """Return a copy so source records cannot mutate the selection policy."""
    return dict(SOURCE_EXCLUSIONS.get(slug, {}))
