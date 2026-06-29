"""Per-residue aggregation of Å-valued local-resolution maps (BlocRes, etc.)."""

from __future__ import annotations

import logging
from pathlib import Path

import mrcfile
import numpy as np
import pandas as pd

from .map_grid import MapGrid, load_map_grid, resample_volume_onto_grid, verify_same_grid_as_reference
from .structure_validation import (
    build_ca_sphere_index_caches,
    iter_ca_residues,
)

logger = logging.getLogger(__name__)

MIN_SPHERE_VOXELS = 3


def _load_locres_volume(path: Path) -> tuple[np.ndarray, object]:
    """Load local-resolution data array and MRC header via ``mrcfile``."""
    with mrcfile.open(path, permissive=True) as mrc:
        data = np.asarray(mrc.data, dtype=np.float64)
        header = mrc.header
    if data.ndim == 2:
        data = data[np.newaxis, ...]
    return data, header


def aggregate_locres_to_ca(
    locres_mrc_path: str | Path,
    structure_path: str | Path,
    *,
    radius_angstrom: float = 2.0,
    mask_path: str | Path | None = None,
    reference_path: str | Path | None = None,
    positive_only: bool = True,
    exclude_at_or_above: float | None = None,
    value_column: str = "local_resolution_mean",
) -> pd.DataFrame:
    """
    Average a per-voxel scalar volume within a sphere around each Cα.

    Returns a DataFrame with columns ``chain``, ``seq_num``, ``<value_column>``,
    and ``n_voxels``. Residues with fewer than three in-sphere voxels are set to NaN.

    ``positive_only`` (default True) keeps the original local-resolution behaviour of
    discarding non-positive voxels (Å resolutions are strictly positive). Set it to
    ``False`` for signed maps such as FSC-Q, where values are differences in Å and
    legitimately negative; in that case only non-finite voxels are dropped.

    ``exclude_at_or_above`` drops voxels with value >= the cutoff (ResMap writes 100 Å
  for unresolved voxels).
    """
    locres_path = Path(locres_mrc_path)
    struct_path = Path(structure_path)
    locres_raw, _ = _load_locres_volume(locres_path)
    grid = load_map_grid(locres_path, dtype=np.float64)
    if grid.data.shape != locres_raw.shape:
        grid = type(grid)(
            data=locres_raw,
            voxel_size_zyx=grid.voxel_size_zyx,
            origin_zyx=grid.origin_zyx,
            shape_zyx=locres_raw.shape,
            mapc=grid.mapc,
            mapr=grid.mapr,
            maps=grid.maps,
            path=grid.path,
        )

    ref_g = load_map_grid(reference_path, dtype=np.float32) if reference_path is not None else None
    if ref_g is not None:
        grid = _align_locres_grid_to_reference(grid, ref_g, locres_path.name)

    mask_vol: np.ndarray | None = None
    if mask_path is not None:
        mask_g = load_map_grid(mask_path, dtype=np.float32)
        rep = grid.shape_zyx == mask_g.shape_zyx
        if not rep:
            raise ValueError(
                f"mask shape {mask_g.shape_zyx} != locres {grid.shape_zyx}"
            )
        mask_vol = np.asarray(mask_g.data, dtype=np.float32)
    elif ref_g is not None:
        mask_vol = (np.asarray(ref_g.data, dtype=np.float32) != 0).astype(np.float32)

    residues = iter_ca_residues(struct_path)
    caches = build_ca_sphere_index_caches(residues, grid, float(radius_angstrom))
    volume = np.asarray(grid.data, dtype=np.float64)

    records: list[dict[str, object]] = []
    for res, idx in zip(residues, caches):
        if idx.size == 0:
            logger.warning(
                "EMD Cα %s:%d: empty sphere index; local_resolution_mean=NaN",
                res.chain,
                res.seq_num,
            )
            records.append(
                {
                    "chain": res.chain,
                    "seq_num": res.seq_num,
                    value_column: float("nan"),
                    "n_voxels": 0,
                }
            )
            continue

        if mask_vol is not None:
            in_mask = mask_vol[idx[:, 0], idx[:, 1], idx[:, 2]] > 0
            idx = idx[in_mask]

        n_voxels = int(idx.shape[0])
        if n_voxels < MIN_SPHERE_VOXELS:
            logger.warning(
                "Cα %s:%d: only %d voxels in sphere (<%d); local_resolution_mean=NaN",
                res.chain,
                res.seq_num,
                n_voxels,
                MIN_SPHERE_VOXELS,
            )
            mean_val = float("nan")
        else:
            vals = volume[idx[:, 0], idx[:, 1], idx[:, 2]]
            keep = np.isfinite(vals)
            if positive_only:
                keep &= vals > 0
            if exclude_at_or_above is not None:
                keep &= vals < float(exclude_at_or_above)
            finite = vals[keep]
            if finite.size < MIN_SPHERE_VOXELS:
                logger.warning(
                    "Cα %s:%d: only %d usable voxels; %s=NaN",
                    res.chain,
                    res.seq_num,
                    finite.size,
                    value_column,
                )
                mean_val = float("nan")
                n_voxels = int(finite.size)
            else:
                mean_val = float(np.mean(finite))
                n_voxels = int(finite.size)

        records.append(
            {
                "chain": res.chain,
                "seq_num": res.seq_num,
                value_column: mean_val,
                "n_voxels": n_voxels,
            }
        )

    # Always return the documented schema with stable dtypes, even when ``records``
    # is empty (e.g. no residues, or every Cα falls outside the grid so the
    # sphere-cache zip is empty). A column-less or all-float64 empty frame makes
    # callers fail with a KeyError on column selection, or a dtype ValueError when
    # merging the object ``chain`` key (both observed on EMD-33736).
    out = pd.DataFrame.from_records(
        records,
        columns=["chain", "seq_num", value_column, "n_voxels"],
    )
    if out.empty:
        out = out.astype(
            {"chain": "object", "seq_num": "int64", value_column: "float64", "n_voxels": "int64"}
        )
    return out


def _align_locres_grid_to_reference(
    grid: MapGrid,
    ref_g: MapGrid,
    label: str,
    *,
    resample_cval: float = 0.0,
) -> MapGrid:
    """
    Bring a local-resolution volume onto ``ref_g``'s index grid for Cα sampling.

    Same-origin reheader when shape and voxel size match; otherwise trilinear
    resample in physical Å coordinates (as for half-map alignment before BlocRes).
    """
    align = verify_same_grid_as_reference(grid, ref_g)
    if align.ok:
        return grid
    if align.shape_match and align.voxel_match:
        logger.info(
            "Reheadering %s onto reference origin (%s)",
            label,
            ", ".join(align.messages),
        )
        return MapGrid(
            data=grid.data,
            voxel_size_zyx=ref_g.voxel_size_zyx,
            origin_zyx=ref_g.origin_zyx,
            shape_zyx=ref_g.shape_zyx,
            mapc=ref_g.mapc,
            mapr=ref_g.mapr,
            maps=ref_g.maps,
            path=grid.path,
            normalization=None,
        )
    logger.info(
        "Resampling %s onto reference grid (%s)",
        label,
        "; ".join(align.messages),
    )
    data = resample_volume_onto_grid(grid, ref_g, order=1, chunk_z=32, cval=resample_cval)
    return MapGrid(
        data=data.astype(grid.data.dtype, copy=False),
        voxel_size_zyx=ref_g.voxel_size_zyx,
        origin_zyx=ref_g.origin_zyx,
        shape_zyx=ref_g.shape_zyx,
        mapc=ref_g.mapc,
        mapr=ref_g.mapr,
        maps=ref_g.maps,
        path=grid.path,
        normalization=None,
    )


def align_locres_to_reference(
    reference: str | Path,
    locres_path: str | Path,
    out_path: str | Path,
    *,
    extra_label: str = "local resolution (aligned to reference)",
    resample_cval: float = 0.0,
) -> list[str]:
    """
    Write ``locres_path`` onto the deposited reference MRC header.

    When shape/voxel differ (common for ResMap on half-map grids), resamples in Å
    space first — same convention as half-map resampling before BlocRes.
    """
    from .io import save_volume_like_reference

    reference = Path(reference)
    locres_path = Path(locres_path)
    out_path = Path(out_path)
    ref_g = load_map_grid(reference, dtype=np.float32)
    loc_g = load_map_grid(locres_path, dtype=np.float64)
    notes: list[str] = []
    align = verify_same_grid_as_reference(loc_g, ref_g)
    if align.ok:
        data = np.asarray(loc_g.data, dtype=np.float32)
        notes.append("grid already matches reference")
    elif align.shape_match and align.voxel_match:
        data = np.asarray(loc_g.data, dtype=np.float32)
        notes.append("reheadered: " + "; ".join(align.messages))
    else:
        aligned = _align_locres_grid_to_reference(
            loc_g, ref_g, locres_path.name, resample_cval=resample_cval
        )
        data = np.asarray(aligned.data, dtype=np.float32)
        notes.append("resampled: " + "; ".join(align.messages))
    save_volume_like_reference(reference, data, out_path, extra_label=extra_label)
    return notes


def locres_blocres_path(emdb_id: str | int) -> Path:
    """Standard BlocRes output path for one cohort entry."""
    from .repo_paths import emd_output_dir

    return emd_output_dir(emdb_id) / "locres_blocres.mrc"


def locres_blocres_nomask_path(emdb_id: str | int) -> Path:
    """BlocRes output without ``-Mask`` (Bsoft default: all-but-zero voxels)."""
    from .repo_paths import emd_output_dir

    return emd_output_dir(emdb_id) / "locres_blocres_nomask.mrc"


def locres_resmap_raw_path(emdb_id: str | int) -> Path:
    """ResMap tool output before reference-grid alignment."""
    from .repo_paths import emd_output_dir

    return emd_output_dir(emdb_id) / "resmap" / "resmap.mrc"


def locres_resmap_path(emdb_id: str | int) -> Path:
    """ResMap local-resolution map used for Cα aggregation."""
    return locres_resmap_raw_path(emdb_id)


def locres_monores_path(emdb_id: str | int) -> Path:
    """MonoRes map aligned to the deposited reference grid (Chimera export)."""
    from .repo_paths import emd_output_dir

    return emd_output_dir(emdb_id) / "locres_monores.mrc"


def find_monores_chimera_mrc(emdb_id: str | int, *, data_root: Path | None = None) -> Path | None:
    """Return ``data/emd_<ID>*/monores/monoresResolutionChimera.mrc`` when present."""
    from .repo_paths import DATA_ROOT

    root = data_root or DATA_ROOT
    eid = str(emdb_id).strip()
    for folder in sorted(root.glob(f"emd_{eid}*")):
        candidate = folder / "monores" / "monoresResolutionChimera.mrc"
        if candidate.is_file():
            return candidate
    return None


def find_monores_refined_mask(emdb_id: str | int, *, data_root: Path | None = None) -> Path | None:
    """Return MonoRes ``refinedMask.mrc`` co-located with the Chimera export."""
    from .repo_paths import DATA_ROOT

    root = data_root or DATA_ROOT
    eid = str(emdb_id).strip()
    for folder in sorted(root.glob(f"emd_{eid}*")):
        candidate = folder / "monores" / "refinedMask.mrc"
        if candidate.is_file():
            return candidate
    return None


def ensure_locres_monores_aligned(
    emdb_id: str | int,
    *,
    reference: str | Path,
    force: bool = False,
) -> Path | None:
    """
    Align MonoRes Chimera export onto ``reference`` when raw data exist.

    Returns the aligned ``outputs/emd_<ID>/locres_monores.mrc``, or ``None`` if no
    MonoRes folder is on disk.
    """
    out = locres_monores_path(emdb_id)
    if out.is_file() and not force:
        return out
    raw = find_monores_chimera_mrc(emdb_id)
    if raw is None:
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    align_locres_to_reference(reference, raw, out)
    return out


# ResMap writes 100 Å for voxels that did not pass the FDR threshold.
RESMAP_UNRESOLVED_SENTINEL_A = 100.0
