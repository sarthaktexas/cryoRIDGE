"""Read ChimeraX volumes and color surfaces by reliability overlays."""

from __future__ import annotations

import time

import numpy as np

from chimerax.core.errors import UserError
from chimerax.map import Volume, volume_from_grid_data
from chimerax.map_data import ArrayGridData


# Match thesis figure scripts (thesis/chimerax_figures.py): gray shell at 82% transparency.
DEFAULT_SURFACE_TRANSPARENCY = 82

OVERLAY_SCORE_NAME = "reliability score"
OVERLAY_ZONE_NAME = "build zones"


def describe_map_work(volumes: list[Volume]) -> str:
    """Grid size from ChimeraX metadata (no full map read)."""
    ref = volumes[0]
    nz, ny, nx = (int(s) for s in ref.data.size)
    n = nz * ny * nx
    if n >= 1_000_000:
        n_label = f"{n / 1e6:.1f}M"
    elif n >= 1_000:
        n_label = f"{n / 1e3:.0f}k"
    else:
        n_label = str(n)
    return f"Grid {nz}×{ny}×{nx} ({n_label} voxels per map)"


def _as_volume_list(result) -> list[Volume]:
    if result is None:
        return []
    if isinstance(result, Volume):
        return [result]
    return [v for v in result if isinstance(v, Volume)]


def grids_aligned(reference: Volume, other: Volume) -> bool:
    return reference.data.ijk_to_xyz_transform.same(other.data.ijk_to_xyz_transform)


def resample_onto_reference(session, source: Volume, reference: Volume) -> Volume:
    from chimerax.map_filter.vopcommand import volume_resample

    session.logger.info(
        f"Resampling #{source.id_string} onto #{reference.id_string} grid."
    )
    result = volume_resample(session, [source], on_grid=[reference], hide_maps=False)
    volumes = _as_volume_list(result)
    if not volumes:
        raise UserError(f"Failed to resample #{source.id_string}.")
    return volumes[0]


def read_volume_array(session, volume: Volume) -> np.ndarray:
    """Read the full map grid (Z, Y, X) from a ChimeraX Volume."""
    data = volume.data
    from chimerax.map_data import ProgressReporter

    ijk_size = tuple(data.size)
    progress = ProgressReporter(
        "reading %s" % data.name,
        ijk_size,
        data.value_type.itemsize,
        log=session.logger,
    )
    t0 = time.perf_counter()
    out = np.asarray(
        data.matrix((0, 0, 0), ijk_size, (1, 1, 1), progress),
        dtype=np.float32,
    )
    session.logger.info(
        f"Read #{volume.id_string} shape {out.shape} "
        f"({out.size:,} voxels) in {time.perf_counter() - t0:.1f}s"
    )

    if out.ndim != 3 or out.size < 1_000:
        raise UserError(
            f"Map #{volume.id_string} read as {out.shape} — expected a full 3D volume."
        )
    return out


def read_maps_for_compute(
    session,
    reference: Volume,
    half1: Volume,
    half2: Volume,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read reference + half-maps on the reference grid (resample in ChimeraX if needed)."""
    temp: list[Volume] = []
    try:
        h1_vol = half1
        h2_vol = half2
        if not grids_aligned(reference, half1):
            h1_vol = resample_onto_reference(session, half1, reference)
            temp.append(h1_vol)
        if not grids_aligned(reference, half2):
            h2_vol = resample_onto_reference(session, half2, reference)
            temp.append(h2_vol)

        ref = read_volume_array(session, reference)
        h1 = read_volume_array(session, h1_vol)
        h2 = read_volume_array(session, h2_vol)
        if ref.shape != h1.shape or ref.shape != h2.shape:
            raise UserError(
                f"After resampling, grid shapes still differ: ref {ref.shape}, "
                f"half1 {h1.shape}, half2 {h2.shape}."
            )
        return ref, h1, h2
    finally:
        if temp:
            session.models.close(temp)


def assert_same_grid(volumes: list[Volume]) -> None:
    ref = volumes[0]
    for vol in volumes[1:]:
        if tuple(vol.data.size) != tuple(ref.data.size):
            names = ", ".join(f"#{v.id_string}" for v in volumes)
            raise UserError(f"Selected maps must share grid size. Sizes for {names} differ.")


def default_contour_level(volume: Volume) -> float | None:
    ro = volume.rendering_options
    for attr in ("surface_levels", "levels"):
        levels = getattr(ro, attr, None)
        if levels:
            return float(levels[0])
    return None


def remove_named_overlays(session, names: tuple[str, ...]) -> None:
    for model in list(session.models.list(type=Volume)):
        if model.name in names:
            session.models.close([model])


def add_overlay_volume(
    session,
    reference: Volume,
    values: np.ndarray,
    *,
    name: str,
) -> Volume:
    grid = ArrayGridData(
        np.asarray(values, dtype=np.float32),
        origin=reference.data.origin,
        step=reference.data.step,
        cell_angles=reference.data.cell_angles,
        rotation=reference.data.rotation,
        name=name,
    )
    overlay = volume_from_grid_data(grid, session, show_dialog=False)
    overlay.position = reference.position
    return overlay


def hide_volumes(session, volumes: list[Volume]) -> None:
    """Hide maps and clear the selection so only the colored reference stays in view."""
    from chimerax.core.commands import run

    seen: set[int] = set()
    for vol in volumes:
        if vol is None:
            continue
        key = id(vol)
        if key in seen:
            continue
        seen.add(key)
        vol.display = False
    run(session, "select clear")


def close_volumes(session, volumes: list[Volume]) -> None:
    """Close temporary overlay maps (keeps half-maps hidden, not deleted)."""
    seen: set[int] = set()
    to_close: list[Volume] = []
    for vol in volumes:
        if vol is None:
            continue
        key = id(vol)
        if key in seen:
            continue
        seen.add(key)
        to_close.append(vol)
    if to_close:
        session.models.close(to_close)


def _opacity_byte(transparency_percent: float) -> int:
    return min(255, max(0, int(2.56 * (100 - transparency_percent))))


def _surfaces_with_vertices(reference: Volume, *, contour: float | None = None) -> list:
    reference.update_drawings()
    tol = 1e-5 * max(abs(contour or 0.0), 1.0)
    surfaces = []
    for surf in reference.surfaces:
        if surf.vertices is None or len(surf.vertices) == 0:
            continue
        if contour is not None and abs(surf.level - contour) > tol:
            continue
        surfaces.append(surf)
    if not surfaces:
        surfaces = [
            s
            for s in reference.surfaces
            if s.vertices is not None and len(s.vertices) > 0
        ]
    if not surfaces:
        raise UserError(
            f"No surface mesh on #{reference.id_string}. "
            "Check that the contour level produces a visible surface."
        )
    return surfaces


def color_surface_by_build_zones(
    session,
    reference: Volume,
    zone_map: Volume,
    *,
    contour: float,
    transparency: float = DEFAULT_SURFACE_TRANSPARENCY,
) -> None:
    """Color reference surfaces by discrete 0/1/2 zone labels (nearest-neighbor sampling)."""
    from .compute import BUILD_ZONE_RGBA

    opacity = _opacity_byte(transparency)
    outside_rgba = BUILD_ZONE_RGBA[0]
    zone_counts = {0: 0, 1: 0, 2: 0}
    outside_count = 0

    for surf in _surfaces_with_vertices(reference, contour=contour):
        values, outside = zone_map.interpolated_values(
            surf.vertices,
            surf.scene_position,
            out_of_bounds_list=True,
            method="nearest",
        )
        zones = np.clip(np.rint(values).astype(np.int32), 0, 2)
        unique, counts = np.unique(zones, return_counts=True)
        session.logger.info(
            "Zone map samples at surface: "
            + ", ".join(f"{int(u)}={int(c):,}" for u, c in zip(unique, counts))
        )
        rgba = np.empty((len(zones), 4), dtype=np.float32)
        for zone_id, color in BUILD_ZONE_RGBA.items():
            rgba[zones == zone_id] = color
        if outside:
            outside_idx = np.asarray(outside, dtype=np.int32)
            rgba[outside_idx] = outside_rgba
            outside_count += int(len(outside_idx))
        for zone_id in (0, 1, 2):
            zone_counts[zone_id] += int((zones == zone_id).sum())

        rgba8 = (255 * rgba).astype(np.uint8)
        rgba8[:, 3] = opacity
        surf.auto_recolor_vertices = None
        surf.vertex_colors = rgba8

    reference.redraw_needed()
    session.update_loop.update_graphics_now()
    session.logger.info(
        "Build-zone surface samples: "
        f"omit {zone_counts[0]:,}, caution {zone_counts[1]:,}, "
        f"build {zone_counts[2]:,}"
        + (f", outside {outside_count:,}" if outside_count else "")
    )


def color_map_by_overlay(
    session,
    reference: Volume,
    overlay: Volume,
    *,
    mode: str,
    contour: float,
    hide_volumes_after: list[Volume] | None = None,
    transparency: float = DEFAULT_SURFACE_TRANSPARENCY,
) -> None:
    from chimerax.core.commands import run

    from .compute import RELIABILITY_PALETTE

    ref_id = reference.id_string
    run(session, f"volume #{ref_id} step 1")
    run(
        session,
        f"volume #{ref_id} style surface level {contour:g} transparency {transparency:g}",
    )

    color_map = overlay
    temp: list[Volume] = []
    if not grids_aligned(reference, overlay):
        color_map = resample_onto_reference(session, overlay, reference)
        temp.append(color_map)

    if mode == "zones":
        # Discrete 0/1/2 labels must use nearest-neighbor sampling; color sample
        # interpolates linearly and rescales value-keyed palettes incorrectly.
        color_surface_by_build_zones(
            session, reference, color_map, contour=contour, transparency=transparency
        )
    else:
        map_id = color_map.id_string
        run(
            session,
            f"color sample #{ref_id} map #{map_id} palette {RELIABILITY_PALETTE} "
            f"range 0,1 transparency {transparency:g} key true",
        )

    hide_volumes(session, list(hide_volumes_after or []))
    close_volumes(session, [overlay, *temp])
