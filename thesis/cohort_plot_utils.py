"""Shared cohort figure helpers (thesis-only; not part of cryoem-halfmap-qc)."""

from __future__ import annotations

from collections.abc import Sequence

from style.palette import PALETTES

WINDOWED_HALFMAP_CORRELATION_ONLY_SOURCE = "windowed_halfmap_correlation_only"


def infer_protein_class(display_name: str, flexibility_source: str = "") -> str:
    """
    Coarse architectural class for cohort stratification plots.

    Keyword rules on ``display_name``; falls back to ``flexibility_source`` when
    no structural keyword matches.
    """
    name = display_name.lower()
    if "apoferritin" in name:
        return "Rigid control"
    if any(k in name for k in ("ribosome", "70s")):
        return "Ribosome"
    if any(
        k in name
        for k in (
            "trpv",
            "beta2ar",
            "eag",
            "enac",
            "glp",
            "nmda",
            "receptor",
            "channel",
        )
    ):
        return "Ion channel / receptor"
    if any(
        k in name
        for k in (
            "mgt",
            "msba",
            "p97",
            "vcp",
            "atpase",
            "atp synthase",
            "v-atpase",
        )
    ):
        return "ATPase / transporter"
    if any(k in name for k in ("groel", "clpb")):
        return "Chaperone / AAA+"
    if "proteasome" in name:
        return "Proteasome"
    if any(k in name for k in ("spike", "hsv", " gB")):
        return "Viral glycoprotein"
    if any(k in name for k in ("beta-gal", "betagal", "lrrk", "complex iii", "spliceosome")):
        return "Large enzyme / assembly"
    if any(k in name for k in ("nucleosome", "ribozyme")):
        return "Nucleic acid"
    if flexibility_source in (
        WINDOWED_HALFMAP_CORRELATION_ONLY_SOURCE,
        "halfmap_cc_only",
    ):
        return "Correlation-only (no model)"
    if flexibility_source == "conformational_pair":
        return "Conformation pair"
    return "Other"


def cohort_class_colors(classes: Sequence[str]) -> dict[str, str]:
    palette = list(PALETTES["categorical"])
    return {cls: palette[i % len(palette)] for i, cls in enumerate(classes)}
