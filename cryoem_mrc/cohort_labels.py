"""Human-readable cohort structure labels for figures."""

from __future__ import annotations

import csv
from pathlib import Path

from .repo_paths import COHORT_MANIFEST
from .structure_validation import load_cohort_manifest_row


def load_display_name_map(manifest: Path | None = None) -> dict[str, str]:
    """``emdb_id`` → ``display_name`` from ``cohort/manifest.csv``."""
    path = manifest if manifest is not None else COHORT_MANIFEST
    out: dict[str, str] = {}
    if not path.is_file():
        return out
    with path.open(newline="") as f:
        for row in csv.DictReader(f):
            eid = str(row.get("emdb_id", "")).strip()
            if eid:
                out[eid] = str(row.get("display_name", "")).strip()
    return out


def short_display_name(name: str, *, max_len: int = 42) -> str:
    """
    Compact label for dense bar charts.

    Keeps text before the first ``(`` (conformation/state suffix) and truncates
    long names with an ellipsis.
    """
    text = name.split("(")[0].strip() or name.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"


def cohort_figure_label(
    emdb_id: str,
    *,
    manifest: Path | None = None,
    names: dict[str, str] | None = None,
    short: bool = True,
    max_len: int = 42,
) -> str:
    """
    Structure name for cohort figure axes (never bare ``EMD-`` unless manifest miss).

    ``short=True`` (default) strips parenthetical state tags for readability.
    """
    eid = str(emdb_id).strip()
    name = ""
    if names is not None:
        name = names.get(eid, "").strip()
    if not name:
        path = manifest if manifest is not None else COHORT_MANIFEST
        try:
            row = load_cohort_manifest_row(path, eid)
            name = str(row.get("display_name", "")).strip()
        except (KeyError, OSError, csv.Error):
            pass
    if not name:
        return f"EMD-{eid}"
    return short_display_name(name, max_len=max_len) if short else name


def cohort_figure_labels(
    emdb_ids: list[str] | tuple[str, ...],
    *,
    manifest: Path | None = None,
    names: dict[str, str] | None = None,
    short: bool = True,
    max_len: int = 42,
) -> list[str]:
    """Batch wrapper for :func:`cohort_figure_label`."""
    name_map = names if names is not None else load_display_name_map(manifest)
    return [
        cohort_figure_label(eid, manifest=manifest, names=name_map, short=short, max_len=max_len)
        for eid in emdb_ids
    ]
