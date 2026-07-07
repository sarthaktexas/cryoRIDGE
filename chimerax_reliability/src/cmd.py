"""Run reliability computation and color a map surface."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from chimerax.core.commands import CmdDesc, EnumOf, FloatArg, IntArg, ModelsArg, SaveFileNameArg, Bounded
from chimerax.core.errors import UserError
from chimerax.map import Volume

from .compute import compute_reliability
from .volumes import (
    DEFAULT_SURFACE_TRANSPARENCY,
    OVERLAY_SCORE_NAME,
    OVERLAY_ZONE_NAME,
    add_overlay_volume,
    assert_same_grid,
    color_map_by_overlay,
    describe_map_work,
    read_maps_for_compute,
    remove_named_overlays,
)


def _pick_volumes(models, *, need: int, label: str) -> list[Volume]:
    volumes = [m for m in models if isinstance(m, Volume)]
    if len(volumes) < need:
        raise UserError(
            f"reliability {label}: select {need} density map(s); got {len(volumes)}."
        )
    return volumes[:need]


def run_reliability_coloring(
    session,
    reference: Volume,
    half1: Volume,
    half2: Volume,
    *,
    contour: float,
    mode: str = "score",
    window: int = 5,
    transparency: float = DEFAULT_SURFACE_TRANSPARENCY,
    save_path: Path | None = None,
    progress=None,
) -> None:
    def _status(msg: str) -> None:
        session.logger.status(msg)
        if progress is not None:
            progress(msg)

    assert_same_grid([reference, half1, half2])
    _status("Reading maps...")

    t_all = time.perf_counter()
    ref_data, h1, h2 = read_maps_for_compute(session, reference, half1, half2)

    _status("Computing reliability...")
    t0 = time.perf_counter()
    result = compute_reliability(
        ref_data,
        h1,
        h2,
        contour=contour,
        window=window,
    )
    elapsed = time.perf_counter() - t0

    session.logger.info(
        f"Reliability compute finished in {elapsed:.1f}s; "
        f"mask {result.mask_voxels:,} voxels"
        + (f"; {result.crop_log}" if result.crop_log else "")
    )
    inside = result.reliability_score[result.reliability_score > 0]
    if inside.size:
        session.logger.info(
            f"In-mask score min={float(inside.min()):.4f}, max={float(inside.max()):.4f}"
        )

    remove_named_overlays(session, (OVERLAY_SCORE_NAME, OVERLAY_ZONE_NAME))

    if mode == "zones":
        overlay_values = result.build_zone.astype(np.float32)
        overlay_name = OVERLAY_ZONE_NAME
    else:
        overlay_values = result.reliability_score
        overlay_name = OVERLAY_SCORE_NAME

    _status("Coloring surface...")
    overlay = add_overlay_volume(session, reference, overlay_values, name=overlay_name)
    if save_path is not None:
        from chimerax.core.commands import run

        run(session, f"save #{overlay.id_string} {save_path}")
    color_map_by_overlay(
        session,
        reference,
        overlay,
        mode=mode,
        contour=contour,
        hide_volumes_after=[half1, half2],
        transparency=transparency,
    )

    z = result.zone_counts
    total = time.perf_counter() - t_all
    session.logger.info(
        f"Zones omit/caution/build (in mask): "
        f"{z.get(0, 0):,} / {z.get(1, 0):,} / {z.get(2, 0):,}"
    )
    session.logger.info(
        f"Map reliability finished in {total:.1f}s "
        f"(compare zone counts to CLI: cryoem_mrc reliability_driver)"
    )
    _status(f"Done in {total:.1f}s")


def reliability(session, models, contour=None, mode="score", window=5, transparency=None, save_path=None):
    """Color a reference map by half-map reliability."""
    if len(models) < 3:
        raise UserError("Select three maps: reference, half-map 1, half-map 2.")
    reference, half1, half2 = _pick_volumes(models, need=3, label="color")
    if contour is None:
        from .volumes import default_contour_level

        contour = default_contour_level(reference)
        if contour is None:
            raise UserError("Specify contour (density threshold for the analysis mask).")
    if transparency is None:
        transparency = DEFAULT_SURFACE_TRANSPARENCY
    run_reliability_coloring(
        session,
        reference,
        half1,
        half2,
        contour=contour,
        mode=mode,
        window=window,
        transparency=transparency,
        save_path=Path(save_path) if save_path else None,
    )


reliability_desc = CmdDesc(
    synopsis="Color a map by half-map reliability or build zones",
    required=[("models", ModelsArg)],
    optional=[
        ("contour", FloatArg),
        ("mode", EnumOf(("score", "zones"))),
        ("window", IntArg),
        ("transparency", Bounded(FloatArg, min=0, max=95)),
        ("save_path", SaveFileNameArg),
    ],
)
