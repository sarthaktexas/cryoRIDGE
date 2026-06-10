"""ChimeraX CLI scripts and publication-style protein figure panels.

Generates UCSF ChimeraX ``.cxc`` command scripts for 3D density surfaces colored by
map statistics or deposited-model domains, then composes thesis rows (3D + slices +
histograms) and triptychs. When ChimeraX is not installed, matplotlib fallbacks render
Cα-colored 3D previews so panels still build offline.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.gridspec import GridSpecFromSubplotSpec
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from style.nature import apply, label_panel, savefig as save_nature

from .analysis import build_contour_mask
from .half_map_repro import (
    WINDOWED_HALFMAP_CORRELATION_KEY,
    WINDOWED_HALFMAP_CORRELATION_LABEL,
    WINDOWED_HALFMAP_CORRELATION_MRC_NAME,
    resolve_windowed_halfmap_correlation_mrc,
)
from .conformation_pair import (
    DOMAIN_COLORS,
    UNASSIGNED_DOMAIN_COLOR,
    DomainRegion,
    domain_residue_color,
    get_domain_regions_for_emdb,
    region_matches_residue,
)
from .io import load_mrc, save_volume_like_reference
from .repo_paths import (
    COHORT_MANIFEST,
    analysis_dir,
    emd_output_dir,
    find_features_npz,
    lh_map_reliability_dir,
    locres_blocres_mrc,
)
from .structure_validation import iter_ca_residues, load_cohort_manifest_row
from .thesis_figures import (
    LOCRES_CBAR_LABEL,
    RELIABILITY_CMAP_CC,
    RELIABILITY_CMAP_LOCRES,
    _locres_robust_limits,
    _robust_limits,
    apply_contour_mask,
    SliceCrop,
    crop_slice_2d,
    extract_slice,
    mask_slice_values,
    pick_slice_index,
    plot_masked_slice,
    slice_crop_from_mask,
)

# ChimeraX domain render uses this shell color (``write_domain_surface_cxc``).
DENSITY_SHELL_GRAY = "#bbbbbb"

# Default thesis triptych: domain-annotated maps with deposited models.
DEFAULT_DOMAIN_TRIPTYCH_IDS: tuple[str, ...] = ("49450", "23129", "4941")

# Diverse statistic showcase (symmetric / membrane / transporter).
DEFAULT_STATISTIC_TRIPTYCH_IDS: tuple[str, ...] = ("11638", "49450", "23129")

# Default batch statistics (thesis key readouts).
DEFAULT_STATISTIC_KEYS: tuple[str, ...] = (
    "local_resolution",
    "reliability_score",
    "windowed_halfmap_correlation",
)

# Anchor MgtA conformation pair for thesis-only ChimeraX quadtychs.
MGTA_CONFORMATION_PAIR: tuple[str, str] = ("49450", "48923")

PIPELINE_STEPS: tuple[tuple[str, str], ...] = (
    ("1", "Deposited model\n+ map shell"),
    ("2", "BlocRes\nlocal resolution"),
    ("3", "LH map\nreliability score"),
    ("4", "Windowed\nhalf-map CC"),
)

# Keys exported as individual pipeline panels (step 1 = gray shell, not domain colors).
PIPELINE_PANEL_KEYS: tuple[str, ...] = ("map_shell", *DEFAULT_STATISTIC_KEYS)

ColorMode = Literal["domain", "statistic", "map_shell"]

# ChimeraX surface render knobs. Preview uses coarse step + local staging.
PREVIEW_RENDER = dict(step=4, width=640, height=640, supersample=1, timeout_s=1200)
PUBLICATION_RENDER = dict(step=1, width=900, height=900, supersample=3, timeout_s=1800)


@dataclass(frozen=True)
class StatisticSpec:
    """One map-based coloring mode for ChimeraX ``color sample`` and slice panels."""

    key: str
    label: str
    cbar_label: str
    matplotlib_cmap: str
    chimerax_palette: str
    vmin: float | None = None
    vmax: float | None = None
    robust: bool = True
    locres_limits: bool = False
    npz_key: str | None = None
    mrc_suffix: str | None = None
    halfmap_mrc_name: str | None = None


STATISTIC_SPECS: dict[str, StatisticSpec] = {
    "local_resolution": StatisticSpec(
        key="local_resolution",
        label="BlocRes local resolution",
        cbar_label=LOCRES_CBAR_LABEL,
        matplotlib_cmap=RELIABILITY_CMAP_LOCRES,
        chimerax_palette="buylrd",
        locres_limits=True,
        mrc_suffix="locres_blocres.mrc",
    ),
    "reliability_score": StatisticSpec(
        key="reliability_score",
        label="Reliability score",
        cbar_label="reliability score",
        matplotlib_cmap="viridis",
        chimerax_palette="viridis",
        vmin=0.0,
        vmax=1.0,
        robust=False,
        npz_key="reliability_score",
        mrc_suffix="reliability.mrc",
    ),
    WINDOWED_HALFMAP_CORRELATION_KEY: StatisticSpec(
        key=WINDOWED_HALFMAP_CORRELATION_KEY,
        label=WINDOWED_HALFMAP_CORRELATION_LABEL.title(),
        cbar_label=WINDOWED_HALFMAP_CORRELATION_LABEL,
        matplotlib_cmap=RELIABILITY_CMAP_CC,
        chimerax_palette="redblue",
        vmin=0.0,
        vmax=1.0,
        robust=False,
        halfmap_mrc_name=WINDOWED_HALFMAP_CORRELATION_MRC_NAME,
    ),
    "local_variance": StatisticSpec(
        key="local_variance",
        label="Local variance",
        cbar_label="local variance",
        matplotlib_cmap="magma",
        chimerax_palette="magma",
        npz_key="local_variance",
    ),
    "H_repro": StatisticSpec(
        key="H_repro",
        label="H_repro",
        cbar_label="H_repro (low = reliable)",
        matplotlib_cmap="viridis_r",
        chimerax_palette="viridis",
        npz_key="reliability_H_repro",
    ),
    "build_zone": StatisticSpec(
        key="build_zone",
        label="Build zones",
        cbar_label="zone (0=omit, 2=build)",
        matplotlib_cmap="RdYlGn",
        chimerax_palette="redblue",
        vmin=0.0,
        vmax=2.0,
        robust=False,
        npz_key="build_zone",
    ),
}


@dataclass
class ProteinFigureBundle:
    """Resolved inputs for one EMDB entry."""

    emdb_id: str
    display_name: str
    reference_mrc: Path
    structure_path: Path
    contour: float
    domain_regions: list[DomainRegion] = field(default_factory=list)
    mask: np.ndarray | None = None
    voxel_size_a: float = 1.0


def find_chimerax_executable(explicit: str | Path | None = None) -> Path | None:
    """Return ChimeraX binary if found (explicit path, PATH, or common macOS bundle)."""
    if explicit is not None:
        path = Path(explicit).expanduser()
        if path.is_file() and os.access(path, os.X_OK):
            return path
        raise FileNotFoundError(f"ChimeraX not executable: {path}")

    found = shutil.which("ChimeraX") or shutil.which("chimerax")
    if found:
        return Path(found)

    candidates: list[Path] = []
    for root in (Path("/Applications"), Path.home() / "Applications"):
        if not root.is_dir():
            continue
        for app in sorted(root.glob("ChimeraX*.app")) + sorted(root.glob("UCSF ChimeraX*.app")):
            exe = app / "Contents" / "bin" / "ChimeraX"
            if exe.is_file():
                candidates.append(exe)
    return candidates[0] if candidates else None


def _chimerax_chain_filter_lines(
    structure_path: Path,
    *,
    model_id: str = "#1",
    max_ca: int = 900,
) -> list[str]:
    """Hide extra chains when the deposited model is a large assembly."""
    residues = iter_ca_residues(structure_path)
    if len(residues) <= max_ca:
        return []
    from collections import Counter

    counts = Counter(str(r.auth_chain or r.chain).strip() for r in residues)
    keep = counts.most_common(1)[0][0]
    return [f"hide {model_id} & ~/{keep}"]


def _auth_chains_for_region(structure_path: Path, reg: DomainRegion) -> list[str]:
    import gemmi

    st = gemmi.read_structure(str(structure_path))
    chains: list[str] = []
    for model in st:
        for chain in model:
            name = chain.name
            if reg.chains is not None and name not in reg.chains:
                continue
            if reg.chain_prefixes is not None and not any(name.startswith(p) for p in reg.chain_prefixes):
                continue
            chains.append(name)
        break
    return sorted(set(chains))


def chimerax_domain_select_expr(reg: DomainRegion, structure_path: Path) -> str:
    """ChimeraX ``select`` expression for one domain band (auth seq_id)."""
    seq = f":{reg.seq_start}-{reg.seq_end}"
    if reg.chains is None and reg.chain_prefixes is None:
        return seq
    chains = _auth_chains_for_region(structure_path, reg)
    if not chains:
        return seq
    return " | ".join(f"/{chain}{seq}" for chain in chains)


def _quote(path: Path) -> str:
    return str(path.resolve()).replace("\\", "\\\\").replace('"', '\\"')


def _stage_map_for_chimerax(src: Path, cache_dir: Path) -> Path:
    """Copy a map to a local cache dir so ChimeraX reads from fast storage."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dst = cache_dir / src.name
    if not dst.is_file() or dst.stat().st_mtime < src.stat().st_mtime:
        print(f"[chimerax] staging {src.name} -> {cache_dir}", flush=True)
        shutil.copy2(src, dst)
    return dst


def write_statistic_surface_cxc(
    bundle: ProteinFigureBundle,
    *,
    statistic: StatisticSpec,
    statistic_mrc: Path,
    out_png: Path,
    out_script: Path,
    vmin: float,
    vmax: float,
    width: int = 900,
    height: int = 900,
    step: int = 1,
    supersample: int = 3,
) -> None:
    """Write ChimeraX script: density isosurface colored by a statistic map."""
    ref = _quote(bundle.reference_mrc)
    stat = _quote(statistic_mrc)
    png = _quote(out_png)
    lines = [
        f'open "{ref}" name density',
        f'open "{stat}" name metric',
        f"volume #1 style surface level {bundle.contour:g} step {step}",
        f"color sample #1 map #2 palette {statistic.chimerax_palette} range {vmin:g},{vmax:g}",
        "set bgcolor white",
        "lighting soft",
        "view orient",
        "turn y 135",
        "turn x 15",
        f'save "{png}" width {width} height {height} supersample {supersample}',
        "exit",
    ]
    out_script.parent.mkdir(parents=True, exist_ok=True)
    out_script.write_text("\n".join(lines) + "\n")


def write_domain_surface_cxc(
    bundle: ProteinFigureBundle,
    *,
    out_png: Path,
    out_script: Path,
    width: int = 900,
    height: int = 900,
    step: int = 1,
    supersample: int = 3,
    cartoon_only: bool = False,
    domain_colors: Mapping[str, str] | None = None,
) -> None:
    """Write ChimeraX script: domain-colored deposited model (optional density shell)."""
    model = _quote(bundle.structure_path)
    png = _quote(out_png)
    lines: list[str] = []
    if cartoon_only:
        lines.append(f'open "{model}" name model')
        lines.extend(["cartoon #1", "hide #1 atoms"])
        lines.extend(_chimerax_chain_filter_lines(bundle.structure_path, model_id="#1"))
        color_target = "#1"
    else:
        ref = _quote(bundle.reference_mrc)
        lines.extend(
            [
                f'open "{ref}" name density',
                f'open "{model}" name model',
                f"volume #1 style surface level {bundle.contour:g} step {step} color #bbbbbb transparency 82",
                "cartoon #2",
                "hide #2 atoms",
            ]
        )
        lines.extend(_chimerax_chain_filter_lines(bundle.structure_path, model_id="#2"))
        color_target = "#2"
    for reg in bundle.domain_regions:
        sel = chimerax_domain_select_expr(reg, bundle.structure_path)
        if domain_colors and reg.name in domain_colors:
            color = domain_colors[reg.name]
        else:
            color = reg.color if reg.color.startswith("#") else f"#{reg.color}"
        if not color.startswith("#"):
            color = f"#{color}"
        lines.append(f"select {sel}")
        lines.append(f"color {color} sel")
        lines.append("select clear")
    lines.extend(
        [
            "set bgcolor white",
            "lighting soft",
            "view orient",
            "turn y 135",
            "turn x 15",
            f'save "{png}" width {width} height {height} supersample {supersample}',
            "exit",
        ]
    )
    out_script.parent.mkdir(parents=True, exist_ok=True)
    out_script.write_text("\n".join(lines) + "\n")


def write_map_shell_surface_cxc(
    bundle: ProteinFigureBundle,
    *,
    out_png: Path,
    out_script: Path,
    width: int = 900,
    height: int = 900,
    step: int = 1,
    supersample: int = 3,
) -> None:
    """Write ChimeraX script: gray density shell + deposited cartoon (no domain coloring)."""
    ref = _quote(bundle.reference_mrc)
    model = _quote(bundle.structure_path)
    png = _quote(out_png)
    shell = DENSITY_SHELL_GRAY
    lines = [
        f'open "{ref}" name density',
        f'open "{model}" name model',
        f"volume #1 style surface level {bundle.contour:g} step {step} color {shell} transparency 82",
        "cartoon #2",
        f"color {shell} #2",
        "hide #2 atoms",
    ]
    lines.extend(_chimerax_chain_filter_lines(bundle.structure_path, model_id="#2"))
    lines.extend(
        [
            "set bgcolor white",
            "lighting soft",
            "view orient",
            "turn y 135",
            "turn x 15",
            f'save "{png}" width {width} height {height} supersample {supersample}',
            "exit",
        ]
    )
    out_script.parent.mkdir(parents=True, exist_ok=True)
    out_script.write_text("\n".join(lines) + "\n")


def run_chimerax_script(
    script_path: Path,
    *,
    executable: Path | None = None,
    timeout_s: int = 180,
) -> bool:
    """Run a ``.cxc`` script headless. Returns False when ChimeraX is missing or fails."""
    exe = executable or find_chimerax_executable()
    if exe is None:
        return False
    cmd = [str(exe), "--nogui", str(script_path)]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return proc.returncode == 0


def resolve_protein_bundle(
    emdb_id: str,
    *,
    manifest: Path = COHORT_MANIFEST,
) -> ProteinFigureBundle:
    """Load manifest row, mask, and domain registry entry for one map."""
    row = load_cohort_manifest_row(manifest, emdb_id)
    ref = Path(row["reference_mrc"])
    pdb_raw = str(row.get("flexibility_path_or_pdb", "")).strip()
    if not pdb_raw:
        raise FileNotFoundError(f"EMD-{emdb_id}: no deposited structure in manifest")
    structure = Path(pdb_raw)
    if not ref.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id}: missing reference map {ref}")
    if not structure.is_file():
        raise FileNotFoundError(f"EMD-{emdb_id}: missing structure {structure}")

    contour = float(row["contour"])
    ref_vol = load_mrc(ref, dtype=np.float32)
    mask = build_contour_mask(ref_vol, contour)

    voxel_size_a = 1.0
    try:
        import mrcfile

        with mrcfile.open(ref, header_only=True) as mrc:
            if mrc.voxel_size.x:
                voxel_size_a = float(mrc.voxel_size.x)
    except Exception:
        pass

    return ProteinFigureBundle(
        emdb_id=str(emdb_id).strip(),
        display_name=str(row.get("display_name", "")).strip(),
        reference_mrc=ref,
        structure_path=structure,
        contour=contour,
        domain_regions=get_domain_regions_for_emdb(emdb_id),
        mask=mask,
        voxel_size_a=voxel_size_a,
    )


def _resolve_statistic_mrc(
    bundle: ProteinFigureBundle,
    spec: StatisticSpec,
    *,
    work_dir: Path,
) -> Path | None:
    emd = bundle.emdb_id
    out_root = emd_output_dir(emd)

    if spec.mrc_suffix:
        if spec.mrc_suffix == "locres_blocres.mrc":
            path = locres_blocres_mrc(emd)
        else:
            path = lh_map_reliability_dir(emd) / f"emd_{emd}_{spec.mrc_suffix}"
        return path if path.is_file() else None

    if spec.halfmap_mrc_name:
        metrics_dir = analysis_dir(emd) / "halfmap_metrics"
        if spec.key == WINDOWED_HALFMAP_CORRELATION_KEY:
            resolved = resolve_windowed_halfmap_correlation_mrc(metrics_dir)
            return resolved
        path = metrics_dir / spec.halfmap_mrc_name
        return path if path.is_file() else None

    if spec.npz_key:
        rel_npz = lh_map_reliability_dir(emd) / "reliability.npz"
        if rel_npz.is_file() and spec.npz_key in ("reliability_score", "reliability_H_repro", "build_zone"):
            with np.load(rel_npz, allow_pickle=False) as d:
                if spec.npz_key not in d:
                    return None
                vol = np.asarray(d[spec.npz_key], dtype=np.float32)
            path = work_dir / f"emd_{emd}_{spec.key}.mrc"
            save_volume_like_reference(bundle.reference_mrc, vol, path)
            return path

        if spec.npz_key == "local_variance":
            data_dir = bundle.reference_mrc.parent
            feat = find_features_npz(data_dir, emd, bundle.contour)
            if feat is None or not feat.is_file():
                return None
            with np.load(feat, allow_pickle=False) as d:
                if "local_variance" not in d:
                    return None
                vol = np.asarray(d["local_variance"], dtype=np.float32)
            path = work_dir / f"emd_{emd}_local_variance.mrc"
            save_volume_like_reference(bundle.reference_mrc, vol, path)
            return path

    return None


def _value_limits(
    volume: np.ndarray,
    mask: np.ndarray,
    spec: StatisticSpec,
    *,
    slice_index: int,
    axis: int = 0,
) -> tuple[float, float]:
    if spec.vmin is not None and spec.vmax is not None:
        return spec.vmin, spec.vmax
    sl = extract_slice(volume, axis=axis, index=slice_index)
    msl = extract_slice(mask, axis=axis, index=slice_index)
    if spec.locres_limits:
        return _locres_robust_limits(sl, msl)
    if spec.robust:
        return _robust_limits(sl, mask_sl=msl)
    vals = volume[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return 0.0, 1.0
    return float(np.min(vals)), float(np.max(vals))


def render_matplotlib_surface_fallback(
    bundle: ProteinFigureBundle,
    *,
    mode: ColorMode,
    out_png: Path,
    statistic: StatisticSpec | None = None,
    statistic_volume: np.ndarray | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    dpi: int = 150,
    domain_color_override: Mapping[str, str] | None = None,
) -> None:
    """Cα 3D preview when ChimeraX is unavailable."""
    residues = iter_ca_residues(bundle.structure_path)
    if not residues:
        raise ValueError(f"EMD-{bundle.emdb_id}: no Cα residues in {bundle.structure_path}")

    coords = np.array([[r.x, r.y, r.z] for r in residues], dtype=np.float64)
    if mode == "domain":
        face_colors = []
        for r in residues:
            if domain_color_override:
                c = UNASSIGNED_DOMAIN_COLOR
                row = type(
                    "Row",
                    (),
                    {
                        "auth_seq_num": r.auth_seq_num,
                        "seq_num": r.seq_num,
                        "auth_chain": r.auth_chain,
                        "chain": r.chain,
                    },
                )()
                for reg in bundle.domain_regions:
                    if region_matches_residue(reg, row):
                        c = domain_color_override.get(
                            reg.name, DOMAIN_COLORS.get(reg.name, reg.color)
                        )
                        break
                face_colors.append(c)
            else:
                c = domain_residue_color(
                    int(r.auth_seq_num or r.seq_num),
                    bundle.domain_regions,
                    chain=r.auth_chain or r.chain,
                )
                face_colors.append(c or UNASSIGNED_DOMAIN_COLOR)
        colors_arr = np.array(face_colors)
    elif mode == "map_shell":
        colors_arr = DENSITY_SHELL_GRAY
    else:
        assert statistic is not None and statistic_volume is not None
        from .map_grid import load_map_grid
        from .structure_validation import sample_volume_at_ca

        grid = load_map_grid(bundle.reference_mrc, dtype=np.float32)
        sampled = sample_volume_at_ca(statistic_volume, grid, residues, window_radius=0.0)
        norm = colors.Normalize(vmin=vmin, vmax=vmax)
        cmap = plt.colormaps[statistic.matplotlib_cmap]
        colors_arr = cmap(norm(sampled))

    fig = plt.figure(figsize=(5.5, 5.5), facecolor="white")
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        c=colors_arr,
        s=8,
        depthshade=True,
        linewidths=0,
    )
    ax.set_axis_off()
    ax.view_init(elev=18, azim=135)
    fig.tight_layout(pad=0)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_statistic_histogram(
    volume: np.ndarray,
    mask: np.ndarray,
    *,
    spec: StatisticSpec,
    vmin: float,
    vmax: float,
    title: str,
    ax: plt.Axes,
) -> None:
    vals = np.asarray(volume, dtype=np.float64)[mask]
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        ax.text(0.5, 0.5, "no in-mask data", ha="center", va="center", transform=ax.transAxes)
        return
    bins = min(60, max(20, int(np.sqrt(vals.size))))
    ax.hist(vals, bins=bins, range=(vmin, vmax), color="#4E79A7", edgecolor="white", linewidth=0.3)
    apply(ax)
    ax.set_title(title, fontsize=7)
    ax.set_xlabel(spec.cbar_label)
    ax.set_ylabel("Counts")


def compose_protein_row(
    bundle: ProteinFigureBundle,
    *,
    mode: ColorMode,
    render_png: Path,
    out_path: Path,
    statistic: StatisticSpec | None = None,
    statistic_volume: np.ndarray | None = None,
    density_volume: np.ndarray | None = None,
    vmin: float = 0.0,
    vmax: float = 1.0,
    slice_index: int | None = None,
    dpi: int = 200,
    panel_label: str | None = None,
) -> plt.Figure:
    """
    One publication row: [3D render | statistic/density slices | histograms].

    Layout follows the MonoRes/ResMap overview style from the thesis reference figure.
    """
    mask = bundle.mask
    if mask is None:
        raise ValueError("bundle.mask is required")
    if density_volume is None:
        density_volume = load_mrc(bundle.reference_mrc, dtype=np.float32)

    axis = 0
    z = slice_index if slice_index is not None else pick_slice_index(mask, axis=axis)
    msl = extract_slice(mask, axis=axis, index=z)
    crop = slice_crop_from_mask(msl, pad_voxels=20)

    fig = plt.figure(figsize=(13.5, 3.8), facecolor="white")
    outer = fig.add_gridspec(1, 3, width_ratios=[1.35, 0.95, 0.75], wspace=0.12)

    ax3d = fig.add_subplot(outer[0, 0])
    ax3d.imshow(plt.imread(render_png))
    ax3d.set_axis_off()
    title = bundle.display_name or f"EMD-{bundle.emdb_id}"
    if mode == "domain":
        ax3d.set_title(f"{title}\n(domain coloring)", fontsize=8)
    else:
        assert statistic is not None
        ax3d.set_title(f"{title}\n({statistic.label})", fontsize=8)

    mid = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 1], hspace=0.08)
    ax_top = fig.add_subplot(mid[0, 0])
    ax_bot = fig.add_subplot(mid[1, 0])

    dens_sl = mask_slice_values(extract_slice(density_volume, axis=axis, index=z), msl)
    plot_masked_slice(
        ax_top,
        dens_sl,
        msl,
        cmap="gray",
        title=f"Density\nZ = {z}",
        crop_bbox=crop,
        cbar_label="density",
    )

    if mode == "domain" and statistic_volume is not None and statistic is not None:
        stat_sl = mask_slice_values(extract_slice(statistic_volume, axis=axis, index=z), msl)
        plot_masked_slice(
            ax_bot,
            stat_sl,
            msl,
            cmap=statistic.matplotlib_cmap,
            vmin=vmin,
            vmax=vmax,
            robust=False,
            title=statistic.label,
            crop_bbox=crop,
            cbar_label=statistic.cbar_label,
        )
    elif mode == "domain":
        ax_bot.text(0.5, 0.5, "domain legend in 3D panel", ha="center", va="center", transform=ax_bot.transAxes)
        ax_bot.set_axis_off()
    else:
        assert statistic is not None and statistic_volume is not None
        stat_sl = mask_slice_values(extract_slice(statistic_volume, axis=axis, index=z), msl)
        plot_masked_slice(
            ax_bot,
            stat_sl,
            msl,
            cmap=statistic.matplotlib_cmap,
            vmin=vmin,
            vmax=vmax,
            robust=False,
            title=statistic.label,
            crop_bbox=crop,
            cbar_label=statistic.cbar_label,
        )

    right = GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0, 2], hspace=0.08)
    ax_h1 = fig.add_subplot(right[0, 0])
    ax_h2 = fig.add_subplot(right[1, 0])

    if mode == "statistic" and statistic is not None and statistic_volume is not None:
        plot_statistic_histogram(
            statistic_volume,
            mask,
            spec=statistic,
            vmin=vmin,
            vmax=vmax,
            title=statistic.label,
            ax=ax_h1,
        )
        loc_spec = STATISTIC_SPECS["local_resolution"]
        loc_path = locres_blocres_mrc(bundle.emdb_id)
        if loc_path.is_file():
            loc = load_mrc(loc_path, dtype=np.float32)
            loc = apply_contour_mask(loc, mask)
            lo, hi = _value_limits(loc, mask, loc_spec, slice_index=z, axis=axis)
            plot_statistic_histogram(loc, mask, spec=loc_spec, vmin=lo, vmax=hi, title=loc_spec.label, ax=ax_h2)
        else:
            ax_h2.set_axis_off()
    elif mode == "domain":
        # Domain occupancy by residue count (proxy for fold composition).
        residues = iter_ca_residues(bundle.structure_path)
        names = [reg.name for reg in bundle.domain_regions]
        counts = {name: 0 for name in names}
        unassigned = 0
        color_to_name = {DOMAIN_COLORS.get(reg.name, reg.color): reg.name for reg in bundle.domain_regions}
        for r in residues:
            c = domain_residue_color(
                int(r.auth_seq_num or r.seq_num),
                bundle.domain_regions,
                chain=r.auth_chain or r.chain,
            )
            if c is None or c not in color_to_name:
                unassigned += 1
                continue
            counts[color_to_name[c]] += 1
        labels = [n for n, v in counts.items() if v > 0]
        values = [counts[n] for n in labels]
        if unassigned:
            labels.append("unassigned")
            values.append(unassigned)
        ax_h1.barh(labels, values, color=[DOMAIN_COLORS.get(l, UNASSIGNED_DOMAIN_COLOR) for l in labels])
        apply(ax_h1)
        ax_h1.set_title("Residues per domain")
        ax_h2.set_axis_off()
    else:
        ax_h1.set_axis_off()
        ax_h2.set_axis_off()

    if panel_label:
        label_panel(ax3d, panel_label)
    fig.tight_layout()
    save_nature(fig, out_path, dpi=dpi)
    return fig


def chimerax_renders_dir(emdb_id: str) -> Path:
    return emd_output_dir(emdb_id) / "chimerax_figures" / "renders"


def chimerax_render_png(emdb_id: str, key: str) -> Path:
    """``key`` is ``domain``, ``map_shell``, or a statistic key (e.g. ``local_resolution``)."""
    base = chimerax_renders_dir(emdb_id)
    if key == "domain":
        return base / "surface_domain.png"
    if key == "map_shell":
        return base / "surface_map_shell.png"
    return base / f"surface_{key}.png"


def render_chimerax_domain_colored_surface(
    emdb_id: str,
    *,
    domain_colors: Mapping[str, str],
    out_png: Path,
    chimerax_exe: Path | None = None,
    preview: bool = True,
    force: bool = False,
    dpi: int = 150,
) -> Path:
    """
    Render a domain-colored ChimeraX surface with custom per-domain hex colors.

    Used for conformation-pair coupling-block views (red/blue from panel a scale).
    Falls back to matplotlib Cα preview when ChimeraX is unavailable.
    """
    out_png = Path(out_png)
    meta_path = out_png.with_suffix(".json")
    color_sig = json.dumps(dict(sorted(domain_colors.items())), sort_keys=True)
    if out_png.is_file() and meta_path.is_file() and not force:
        try:
            meta = json.loads(meta_path.read_text())
            if meta.get("domain_colors_sig") == color_sig:
                return out_png
        except (json.JSONDecodeError, OSError):
            pass

    bundle = resolve_protein_bundle(emdb_id)
    out_dir = emd_output_dir(emdb_id) / "chimerax_figures"
    scripts_dir = out_dir / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script = scripts_dir / f"custom_{out_png.stem}.cxc"
    render_kw = PREVIEW_RENDER if preview else PUBLICATION_RENDER
    cache_dir = Path("/tmp") / f"chimerax_emd_{emdb_id}"
    ref_for_chimerax = (
        _stage_map_for_chimerax(bundle.reference_mrc, cache_dir) if preview else bundle.reference_mrc
    )
    bundle_for_chimerax = ProteinFigureBundle(
        emdb_id=bundle.emdb_id,
        display_name=bundle.display_name,
        reference_mrc=ref_for_chimerax,
        structure_path=bundle.structure_path,
        contour=bundle.contour,
        domain_regions=bundle.domain_regions,
        mask=bundle.mask,
        voxel_size_a=bundle.voxel_size_a,
    )
    write_domain_surface_cxc(
        bundle_for_chimerax,
        out_png=out_png,
        out_script=script,
        width=render_kw["width"],
        height=render_kw["height"],
        step=render_kw["step"],
        supersample=render_kw["supersample"],
        cartoon_only=False,
        domain_colors=domain_colors,
    )
    exe = chimerax_exe or find_chimerax_executable()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if exe is not None:
        ok = run_chimerax_script(script, executable=exe, timeout_s=render_kw["timeout_s"])
        if ok and out_png.is_file():
            meta_path.write_text(
                json.dumps({"domain_colors_sig": color_sig, "emdb_id": emdb_id}, indent=2) + "\n"
            )
            return out_png
    render_matplotlib_surface_fallback(
        bundle,
        mode="domain",
        out_png=out_png,
        dpi=dpi,
        domain_color_override=domain_colors,
    )
    meta_path.write_text(json.dumps({"domain_colors_sig": color_sig, "emdb_id": emdb_id}, indent=2) + "\n")
    return out_png


def _framed_image_ax(ax: plt.Axes, png: Path, *, facecolor: str = "#f4f4f5") -> None:
    """Draw a ChimeraX render inside a subtle rounded frame."""
    ax.set_facecolor(facecolor)
    img = plt.imread(png)
    ax.imshow(img, aspect="equal")
    ax.set_axis_off()
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    pad = 0.04 * max(x1 - x0, y1 - y0)
    ax.add_patch(
        FancyBboxPatch(
            (x0 + pad, y0 + pad),
            (x1 - x0) - 2 * pad,
            (y1 - y0) - 2 * pad,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=0.6,
            edgecolor="#cccccc",
            facecolor="none",
            transform=ax.transData,
            clip_on=False,
        )
    )


def _arrow_in_ax(
    ax: plt.Axes,
    *,
    label: str | None = None,
    step: str | None = None,
) -> None:
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.annotate(
        "",
        xy=(0.82, 0.5),
        xytext=(0.18, 0.5),
        xycoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color="#4E79A7", lw=1.4, mutation_scale=12),
    )
    if step:
        ax.text(
            0.5,
            0.72,
            step,
            ha="center",
            va="center",
            fontsize=8,
            fontweight="bold",
            color="white",
            bbox=dict(boxstyle="circle,pad=0.25", facecolor="#4E79A7", edgecolor="none"),
            transform=ax.transAxes,
        )
    if label:
        ax.text(
            0.5,
            0.22,
            label,
            ha="center",
            va="top",
            fontsize=5.5,
            color="#555555",
            transform=ax.transAxes,
            linespacing=1.15,
        )


def _compact_slice_panel(
    ax: plt.Axes,
    sl: np.ndarray,
    mask_sl: np.ndarray,
    *,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
    cbar_label: str | None = None,
    crop_bbox: SliceCrop | None = None,
) -> None:
    """Small slice panel with a thin horizontal colorbar below the image."""
    im = plot_masked_slice(
        ax,
        sl,
        mask_sl,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        robust=vmin is None and vmax is None,
        title=None,
        crop_bbox=crop_bbox,
        cbar_label=None,
    )
    ax.set_aspect("equal")
    if cbar_label:
        cb = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.09, pad=0.06, aspect=28)
        cb.set_label(cbar_label, fontsize=5.5)
        cb.ax.tick_params(labelsize=5, length=2)


def _plot_density_cage_slice(
    ax: plt.Axes,
    mask_sl: np.ndarray,
    *,
    crop_bbox: SliceCrop | None = None,
    shell_color: str = DENSITY_SHELL_GRAY,
) -> None:
    """Map contour cross-section in ChimeraX shell gray (matches 3D density cage)."""
    m = np.asarray(mask_sl, dtype=bool)
    if crop_bbox is not None:
        m = crop_slice_2d(m, crop_bbox)
    h, w = m.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)
    rgba[..., :3] = (0.12, 0.12, 0.14)
    rgba[..., 3] = 1.0
    shell_rgb = np.array(colors.to_rgb(shell_color), dtype=np.float32)
    rgba[m] = np.concatenate([shell_rgb, [1.0]])
    ax.imshow(rgba, origin="lower", aspect="equal")
    ax.contour(m.astype(np.float64), levels=[0.5], colors="#777777", linewidths=0.55, origin="lower")
    apply(ax)
    ax.set_xticks([])
    ax.set_yticks([])


def _domain_legend_in_ax(ax: plt.Axes, regions: Sequence[DomainRegion]) -> None:
    """Horizontal domain-color key between 3D render and slice (domain column only)."""
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if not regions:
        return
    n = len(regions)
    total_w = 0.14 * n
    x = 0.5 - total_w / 2
    for reg in regions:
        color = DOMAIN_COLORS.get(reg.name, reg.color)
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.55),
                0.018,
                0.3,
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor="#999999",
                linewidth=0.4,
                transform=ax.transAxes,
                clip_on=False,
            )
        )
        ax.text(x + 0.022, 0.7, reg.name, ha="left", va="center", fontsize=5.5, transform=ax.transAxes)
        x += 0.14


def _load_available_statistics(
    bundle: ProteinFigureBundle,
    *,
    statistic_keys: Sequence[str] = DEFAULT_STATISTIC_KEYS,
    work_dir: Path | None = None,
) -> list[tuple[StatisticSpec, np.ndarray, float, float]]:
    mask = bundle.mask
    if mask is None:
        raise ValueError("bundle.mask is required")
    work = work_dir or (emd_output_dir(bundle.emdb_id) / "chimerax_figures" / "work")
    work.mkdir(parents=True, exist_ok=True)
    z = pick_slice_index(mask, axis=0)
    out: list[tuple[StatisticSpec, np.ndarray, float, float]] = []
    for key in statistic_keys:
        spec = STATISTIC_SPECS.get(key)
        if spec is None:
            continue
        mrc = _resolve_statistic_mrc(bundle, spec, work_dir=work)
        if mrc is None:
            continue
        vol = load_mrc(mrc, dtype=np.float32)
        vol = apply_contour_mask(vol, mask, outside=np.nan)
        vmin, vmax = _value_limits(vol, mask, spec, slice_index=z, axis=0)
        out.append((spec, vol, vmin, vmax))
    return out


def _draw_domain_legend(fig: plt.Figure, regions: Sequence[DomainRegion], *, y: float = 0.03) -> None:
    if not regions:
        return
    n = len(regions)
    x0 = 0.5 - 0.09 * n
    for i, reg in enumerate(regions):
        color = DOMAIN_COLORS.get(reg.name, reg.color)
        x = x0 + 0.18 * i
        fig.patches.append(
            FancyBboxPatch(
                (x, y),
                0.02,
                0.02,
                boxstyle="square,pad=0",
                facecolor=color,
                edgecolor="#888888",
                linewidth=0.4,
                transform=fig.transFigure,
                clip_on=False,
            )
        )
        fig.text(x + 0.025, y + 0.01, reg.name, ha="left", va="center", fontsize=6, transform=fig.transFigure)


def _save_map_shell_slice_png(
    mask_sl: np.ndarray,
    *,
    out_path: Path,
    crop_bbox: SliceCrop | None = None,
    dpi: int = 200,
) -> Path:
    fig, ax = plt.subplots(figsize=(3.2, 3.2), facecolor="white")
    _plot_density_cage_slice(ax, mask_sl, crop_bbox=crop_bbox)
    save_nature(fig, out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_statistic_slice_png(
    stat_sl: np.ndarray,
    mask_sl: np.ndarray,
    *,
    spec: StatisticSpec,
    vmin: float,
    vmax: float,
    out_path: Path,
    crop_bbox: SliceCrop | None = None,
    dpi: int = 200,
) -> Path:
    fig, ax = plt.subplots(figsize=(3.4, 3.6), facecolor="white")
    _compact_slice_panel(
        ax,
        stat_sl,
        mask_sl,
        cmap=spec.matplotlib_cmap,
        vmin=vmin,
        vmax=vmax,
        crop_bbox=crop_bbox,
        cbar_label=spec.cbar_label,
    )
    save_nature(fig, out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def export_pipeline_panel_assets(
    emdb_id: str,
    *,
    out_dir: Path,
    statistic_keys: Sequence[str] = DEFAULT_STATISTIC_KEYS,
    dpi: int = 200,
) -> dict[str, Path]:
    """
    Export each pipeline step as separate PNGs for manual assembly.

    Writes ``{out_dir}/{emdb_id}/`` with paired 3D ChimeraX renders and map slices::

        map_shell_3d.png / map_shell_slice.png
        local_resolution_3d.png / local_resolution_slice.png
        ...

    Step 1 uses the gray map shell (not domain coloring). A ``manifest.json`` records
    slice index and value limits.
    """
    bundle = resolve_protein_bundle(emdb_id)
    mask = bundle.mask
    assert mask is not None

    panel_dir = out_dir / emdb_id
    panel_dir.mkdir(parents=True, exist_ok=True)

    axis = 0
    z = pick_slice_index(mask, axis=axis)
    msl = extract_slice(mask, axis=axis, index=z)
    crop = slice_crop_from_mask(msl, pad_voxels=20)

    outputs: dict[str, Path] = {}
    meta_panels: list[dict[str, object]] = []

    shell_png = chimerax_render_png(emdb_id, "map_shell")
    if not shell_png.is_file():
        shell_png.parent.mkdir(parents=True, exist_ok=True)
        render_matplotlib_surface_fallback(bundle, mode="map_shell", out_png=shell_png, dpi=dpi)
    shell_3d = panel_dir / "map_shell_3d.png"
    shutil.copy2(shell_png, shell_3d)
    outputs["map_shell_3d"] = shell_3d
    shell_slice = _save_map_shell_slice_png(msl, out_path=panel_dir / "map_shell_slice.png", crop_bbox=crop, dpi=dpi)
    outputs["map_shell_slice"] = shell_slice
    meta_panels.append(
        {
            "key": "map_shell",
            "label": PIPELINE_STEPS[0][1].replace("\n", " "),
            "render_3d": shell_3d.name,
            "slice": shell_slice.name,
            "slice_z": z,
        }
    )

    stats = _load_available_statistics(bundle, statistic_keys=statistic_keys)
    for i, (spec, vol, vmin, vmax) in enumerate(stats, start=2):
        render_src = chimerax_render_png(emdb_id, spec.key)
        if not render_src.is_file():
            continue
        render_dst = panel_dir / f"{spec.key}_3d.png"
        shutil.copy2(render_src, render_dst)
        outputs[f"{spec.key}_3d"] = render_dst
        stat_sl = mask_slice_values(extract_slice(vol, axis=axis, index=z), msl)
        slice_dst = _save_statistic_slice_png(
            stat_sl,
            msl,
            spec=spec,
            vmin=vmin,
            vmax=vmax,
            out_path=panel_dir / f"{spec.key}_slice.png",
            crop_bbox=crop,
            dpi=dpi,
        )
        outputs[f"{spec.key}_slice"] = slice_dst
        step_label = PIPELINE_STEPS[i - 1][1].replace("\n", " ") if i - 1 < len(PIPELINE_STEPS) else spec.label
        meta_panels.append(
            {
                "key": spec.key,
                "label": step_label,
                "render_3d": render_dst.name,
                "slice": slice_dst.name,
                "slice_z": z,
                "vmin": vmin,
                "vmax": vmax,
                "cbar_label": spec.cbar_label,
            }
        )

    if len(meta_panels) < 2:
        raise FileNotFoundError(f"EMD-{emdb_id}: need map_shell + ≥1 statistic render")

    manifest = {
        "emdb_id": emdb_id,
        "display_name": bundle.display_name,
        "slice_axis": axis,
        "slice_index": z,
        "panels": meta_panels,
    }
    manifest_path = panel_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    outputs["manifest"] = manifest_path
    return outputs


def compose_map_pipeline_schematic(
    emdb_id: str,
    *,
    out_path: Path,
    statistic_keys: Sequence[str] = DEFAULT_STATISTIC_KEYS,
    dpi: int = 200,
    title: str | None = None,
) -> plt.Figure:
    """
    Thesis pipeline figure: ChimeraX 3D renders, optional domain legend band, then
    compact map slices — with arrow gutters between steps.
    """
    bundle = resolve_protein_bundle(emdb_id)
    mask = bundle.mask
    assert mask is not None

    shell_png = chimerax_render_png(emdb_id, "map_shell")
    if not shell_png.is_file():
        raise FileNotFoundError(f"missing map_shell render: {shell_png}")

    stats = _load_available_statistics(bundle, statistic_keys=statistic_keys)
    columns: list[tuple[str, Path, StatisticSpec | None, np.ndarray | None, float, float]] = [
        ("map_shell", shell_png, None, None, 0.0, 1.0),
    ]
    for spec, vol, vmin, vmax in stats:
        png = chimerax_render_png(emdb_id, spec.key)
        if png.is_file():
            columns.append((spec.key, png, spec, vol, vmin, vmax))

    if len(columns) < 2:
        raise FileNotFoundError(f"EMD-{emdb_id}: need map_shell + ≥1 statistic render")

    axis = 0
    z = pick_slice_index(mask, axis=axis)
    msl = extract_slice(mask, axis=axis, index=z)
    crop = slice_crop_from_mask(msl, pad_voxels=20)

    n_steps = len(columns)
    width_ratios: list[float] = []
    for j in range(n_steps):
        width_ratios.append(4.0)
        if j < n_steps - 1:
            width_ratios.append(0.45)

    ncols = len(width_ratios)
    fig = plt.figure(figsize=(2.05 * n_steps + 0.6, 5.6), facecolor="white")
    # header | 3D | legend band (domain col only) | compact slice
    gs = fig.add_gridspec(
        4,
        ncols,
        height_ratios=[0.08, 1.45, 0.22, 0.72],
        width_ratios=width_ratios,
        hspace=0.42,
        wspace=0.05,
        left=0.03,
        right=0.995,
        top=0.9,
        bottom=0.05,
    )

    display = bundle.display_name or f"EMD-{emdb_id}"
    fig.suptitle(
        title or f"{display} · map readout pipeline",
        fontsize=10,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.925,
        f"EMD-{emdb_id} · same central slice (Z = {z}) in all columns",
        ha="center",
        fontsize=6.5,
        color="#555555",
    )

    letters = "abcdefgh"
    panel_col = 0
    for j, (key, png, spec, vol, vmin, vmax) in enumerate(columns):
        step_num, step_label = PIPELINE_STEPS[j] if j < len(PIPELINE_STEPS) else (str(j + 1), key)

        ax_head = fig.add_subplot(gs[0, panel_col])
        ax_head.set_axis_off()
        ax_head.text(
            0.5,
            0.35,
            f"{step_num}. {step_label.replace(chr(10), ' ')}",
            ha="center",
            va="center",
            fontsize=6.5,
            fontweight="bold",
            color="#333333",
            transform=ax_head.transAxes,
        )

        ax_top = fig.add_subplot(gs[1, panel_col])
        _framed_image_ax(ax_top, png)
        label_panel(ax_top, letters[j])

        ax_leg = fig.add_subplot(gs[2, panel_col])
        ax_leg.set_axis_off()

        ax_bot = fig.add_subplot(gs[3, panel_col])
        if key == "map_shell":
            _plot_density_cage_slice(ax_bot, msl, crop_bbox=crop)
        else:
            assert spec is not None and vol is not None
            stat_sl = mask_slice_values(extract_slice(vol, axis=axis, index=z), msl)
            _compact_slice_panel(
                ax_bot,
                stat_sl,
                msl,
                cmap=spec.matplotlib_cmap,
                vmin=vmin,
                vmax=vmax,
                crop_bbox=crop,
                cbar_label=spec.cbar_label,
            )

        if j < n_steps - 1:
            ax_arr_top = fig.add_subplot(gs[1, panel_col + 1])
            ax_arr_bot = fig.add_subplot(gs[3, panel_col + 1])
            _arrow_in_ax(ax_arr_top, step=str(j + 2))
            _arrow_in_ax(ax_arr_bot)

        panel_col += 2

    save_nature(fig, out_path, dpi=dpi)
    return fig


def compose_conformation_pair_quadtych(
    emdb_a: str,
    emdb_b: str,
    *,
    out_path: Path,
    change_statistic: str = "reliability_score",
    dpi: int = 200,
    manifest: Path = COHORT_MANIFEST,
) -> plt.Figure:
    """
    Thesis-only quadtych (2×2 + arrow gutters): domain row and statistic row for
    two conformations, with column headers and a shared domain legend.
    """
    row_a = load_cohort_manifest_row(manifest, emdb_a)
    row_b = load_cohort_manifest_row(manifest, emdb_b)
    name_a = str(row_a.get("display_name", f"EMD-{emdb_a}")).strip()
    name_b = str(row_b.get("display_name", f"EMD-{emdb_b}")).strip()

    domain_a = chimerax_render_png(emdb_a, "domain")
    domain_b = chimerax_render_png(emdb_b, "domain")
    stat_a = chimerax_render_png(emdb_a, change_statistic)
    stat_b = chimerax_render_png(emdb_b, change_statistic)
    for path in (domain_a, domain_b, stat_a, stat_b):
        if not path.is_file():
            raise FileNotFoundError(f"missing render for pair figure: {path}")

    stat_spec = STATISTIC_SPECS.get(change_statistic)
    stat_title = stat_spec.label if stat_spec else change_statistic
    regions = get_domain_regions_for_emdb(emdb_a) or get_domain_regions_for_emdb(emdb_b)

    fig = plt.figure(figsize=(12.0, 7.8), facecolor="white")
    gs = fig.add_gridspec(
        4,
        5,
        height_ratios=[0.07, 1.0, 1.0, 0.08],
        width_ratios=[0.45, 4.0, 0.55, 4.0, 0.45],
        hspace=0.28,
        wspace=0.04,
        left=0.05,
        right=0.95,
        top=0.9,
        bottom=0.08,
    )

    fig.suptitle(
        "MgtA conformation pair · ChimeraX surfaces",
        fontsize=11,
        fontweight="bold",
        y=0.97,
    )
    fig.text(
        0.5,
        0.925,
        f"EMD-{emdb_a} ({name_a})  to  EMD-{emdb_b} ({name_b})",
        ha="center",
        fontsize=7.5,
        color="#444444",
    )

    # Column headers
    for col, (emd, name) in enumerate(((emdb_a, name_a), (emdb_b, name_b))):
        ax_h = fig.add_subplot(gs[0, 1 + 2 * col])
        ax_h.set_axis_off()
        ax_h.text(0.5, 0.5, f"EMD-{emd}\n{name}", ha="center", va="center", fontsize=7.5, fontweight="bold")

    row_meta = (
        ("Fold domains\n(deposited model)", domain_a, domain_b, "State change"),
        (f"{stat_title}\n(map surface)", stat_a, stat_b, f"Δ {stat_title.lower()}"),
    )
    letters = ("a", "b", "c", "d")
    letter_i = 0

    for row_i, (row_label, png_left, png_right, arrow_label) in enumerate(row_meta):
        ax_lbl = fig.add_subplot(gs[1 + row_i, 0])
        ax_lbl.set_axis_off()
        ax_lbl.text(
            0.85,
            0.5,
            row_label,
            ha="right",
            va="center",
            fontsize=6.5,
            color="#333333",
            rotation=90,
            linespacing=1.2,
        )

        ax_left = fig.add_subplot(gs[1 + row_i, 1])
        _framed_image_ax(ax_left, png_left)
        label_panel(ax_left, letters[letter_i])
        letter_i += 1

        ax_arrow = fig.add_subplot(gs[1 + row_i, 2])
        _arrow_in_ax(ax_arrow, label=arrow_label)

        ax_right = fig.add_subplot(gs[1 + row_i, 3])
        _framed_image_ax(ax_right, png_right)
        label_panel(ax_right, letters[letter_i])
        letter_i += 1

    _draw_domain_legend(fig, regions, y=0.025)
    save_nature(fig, out_path, dpi=dpi)
    return fig


def compose_triptych(
    row_paths: Sequence[Path],
    *,
    out_path: Path,
    title: str | None = None,
    panel_labels: Sequence[str] = ("A", "B", "C"),
    dpi: int = 200,
) -> plt.Figure:
    """Stack pre-rendered row PNGs into a three-panel figure."""
    paths = [Path(p) for p in row_paths if Path(p).is_file()]
    if not paths:
        raise FileNotFoundError("no row images available for triptych")

    n = len(paths)
    fig = plt.figure(figsize=(13.5, 3.8 * n), facecolor="white")
    for i, path in enumerate(paths):
        ax = fig.add_subplot(n, 1, i + 1)
        ax.imshow(plt.imread(path))
        ax.set_axis_off()
        if i < len(panel_labels):
            label_panel(ax, panel_labels[i])
    if title:
        fig.suptitle(title, fontsize=9, y=0.995)
    fig.tight_layout()
    save_nature(fig, out_path, dpi=dpi)
    return fig


def generate_protein_figures(
    emdb_id: str,
    *,
    out_dir: Path,
    modes: Sequence[str] = ("domain", "statistics"),
    statistics: Sequence[str] | None = None,
    chimerax_exe: Path | None = None,
    dry_run: bool = False,
    dpi: int = 200,
    manifest: Path = COHORT_MANIFEST,
    preview: bool = False,
) -> dict[str, Path]:
    """
    Generate ChimeraX scripts, renders, per-map rows, and return key output paths.

    ``modes`` may include ``domain`` and/or ``statistics``. Statistic keys default to
    all entries in ``STATISTIC_SPECS`` that resolve for this map.
    """
    bundle = resolve_protein_bundle(emdb_id, manifest=manifest)
    out_dir.mkdir(parents=True, exist_ok=True)
    scripts_dir = out_dir / "scripts"
    renders_dir = out_dir / "renders"
    rows_dir = out_dir / "rows"
    work_dir = out_dir / "work"
    cache_dir = Path("/tmp") / f"chimerax_emd_{emdb_id}"
    for d in (scripts_dir, renders_dir, rows_dir, work_dir):
        d.mkdir(parents=True, exist_ok=True)

    density = load_mrc(bundle.reference_mrc, dtype=np.float32)
    mask = bundle.mask
    assert mask is not None
    z = pick_slice_index(mask, axis=0)
    outputs: dict[str, Path] = {}
    exe = chimerax_exe or find_chimerax_executable()
    render_kw = PREVIEW_RENDER if preview else PUBLICATION_RENDER
    ref_for_chimerax = (
        _stage_map_for_chimerax(bundle.reference_mrc, cache_dir) if preview else bundle.reference_mrc
    )
    bundle_for_chimerax = ProteinFigureBundle(
        emdb_id=bundle.emdb_id,
        display_name=bundle.display_name,
        reference_mrc=ref_for_chimerax,
        structure_path=bundle.structure_path,
        contour=bundle.contour,
        domain_regions=bundle.domain_regions,
        mask=bundle.mask,
        voxel_size_a=bundle.voxel_size_a,
    )

    stat_keys = list(statistics) if statistics is not None else list(DEFAULT_STATISTIC_KEYS)
    available_stats: list[tuple[StatisticSpec, Path, np.ndarray, float, float]] = []
    for key in stat_keys:
        spec = STATISTIC_SPECS.get(key)
        if spec is None:
            continue
        mrc = _resolve_statistic_mrc(bundle, spec, work_dir=work_dir)
        if mrc is None:
            continue
        mrc_chimerax = _stage_map_for_chimerax(mrc, cache_dir) if preview else mrc
        vol = load_mrc(mrc, dtype=np.float32)
        vol = apply_contour_mask(vol, mask, outside=np.nan)
        vmin, vmax = _value_limits(vol, mask, spec, slice_index=z, axis=0)
        available_stats.append((spec, mrc_chimerax, vol, vmin, vmax))

    if "map_shell" in modes:
        script = scripts_dir / "surface_map_shell.cxc"
        render_png = renders_dir / "surface_map_shell.png"
        write_map_shell_surface_cxc(
            bundle_for_chimerax,
            out_png=render_png,
            out_script=script,
            width=render_kw["width"],
            height=render_kw["height"],
            step=render_kw["step"],
            supersample=render_kw["supersample"],
        )
        if not dry_run:
            print(
                f"[chimerax] EMD-{emdb_id}: rendering map_shell "
                f"(step={render_kw['step']}, {render_kw['width']}px)...",
                flush=True,
            )
            ok = run_chimerax_script(script, executable=exe, timeout_s=render_kw["timeout_s"])
            if not ok or not render_png.is_file():
                render_matplotlib_surface_fallback(
                    bundle, mode="map_shell", out_png=render_png, dpi=dpi
                )
        outputs["map_shell_render"] = render_png

    if "domain" in modes:
        if not bundle.domain_regions:
            pass
        else:
            script = scripts_dir / "surface_domain.cxc"
            render_png = renders_dir / "surface_domain.png"
            write_domain_surface_cxc(
                bundle_for_chimerax,
                out_png=render_png,
                out_script=script,
                width=render_kw["width"],
                height=render_kw["height"],
                step=render_kw["step"],
                supersample=render_kw["supersample"],
                cartoon_only=False,
            )
            if not dry_run:
                mode_label = "cartoon-only" if preview else "density+cartoon"
                print(
                    f"[chimerax] EMD-{emdb_id}: rendering domain ({mode_label}, "
                    f"{render_kw['width']}px)...",
                    flush=True,
                )
                ok = run_chimerax_script(
                    script, executable=exe, timeout_s=render_kw["timeout_s"]
                )
                if not ok or not render_png.is_file():
                    render_matplotlib_surface_fallback(
                        bundle, mode="domain", out_png=render_png, dpi=dpi
                    )
            # Pair domain row with local resolution slices when available.
            loc_entry = next((t for t in available_stats if t[0].key == "local_resolution"), None)
            loc_spec, loc_vol, loc_vmin, loc_vmax = None, None, 0.0, 1.0
            if loc_entry is not None:
                loc_spec, _, loc_vol, loc_vmin, loc_vmax = loc_entry
            row = rows_dir / "row_domain.png"
            compose_protein_row(
                bundle,
                mode="domain",
                render_png=render_png,
                out_path=row,
                statistic=loc_spec,
                statistic_volume=loc_vol,
                density_volume=density,
                vmin=loc_vmin,
                vmax=loc_vmax,
                slice_index=z,
                dpi=dpi,
            )
            outputs["domain_row"] = row

    if "statistics" in modes:
        for spec, mrc, vol, vmin, vmax in available_stats:
            script = scripts_dir / f"surface_{spec.key}.cxc"
            render_png = renders_dir / f"surface_{spec.key}.png"
            write_statistic_surface_cxc(
                bundle_for_chimerax,
                statistic=spec,
                statistic_mrc=mrc,
                out_png=render_png,
                out_script=script,
                vmin=vmin,
                vmax=vmax,
                width=render_kw["width"],
                height=render_kw["height"],
                step=render_kw["step"],
                supersample=render_kw["supersample"],
            )
            if not dry_run:
                print(
                    f"[chimerax] EMD-{emdb_id}: rendering {spec.key} "
                    f"(step={render_kw['step']}, {render_kw['width']}px)...",
                    flush=True,
                )
                ok = run_chimerax_script(
                    script, executable=exe, timeout_s=render_kw["timeout_s"]
                )
                if not ok or not render_png.is_file():
                    render_matplotlib_surface_fallback(
                        bundle,
                        mode="statistic",
                        out_png=render_png,
                        statistic=spec,
                        statistic_volume=vol,
                        vmin=vmin,
                        vmax=vmax,
                        dpi=dpi,
                    )
            row = rows_dir / f"row_{spec.key}.png"
            compose_protein_row(
                bundle,
                mode="statistic",
                render_png=render_png,
                out_path=row,
                statistic=spec,
                statistic_volume=vol,
                density_volume=density,
                vmin=vmin,
                vmax=vmax,
                slice_index=z,
                dpi=dpi,
            )
            outputs[f"stat_{spec.key}"] = row

    meta = {
        "emdb_id": bundle.emdb_id,
        "display_name": bundle.display_name,
        "contour": bundle.contour,
        "domain_regions": [reg.name for reg in bundle.domain_regions],
        "statistics": [spec.key for spec, *_ in available_stats],
        "chimerax": str(exe) if exe else None,
        "preview": preview,
        "render": render_kw,
        "outputs": {k: str(v) for k, v in outputs.items()},
    }
    (out_dir / "manifest.json").write_text(json.dumps(meta, indent=2) + "\n")
    return outputs


__all__ = [
    "DEFAULT_DOMAIN_TRIPTYCH_IDS",
    "DEFAULT_STATISTIC_KEYS",
    "DEFAULT_STATISTIC_TRIPTYCH_IDS",
    "MGTA_CONFORMATION_PAIR",
    "PIPELINE_PANEL_KEYS",
    "PIPELINE_STEPS",
    "ProteinFigureBundle",
    "STATISTIC_SPECS",
    "StatisticSpec",
    "chimerax_domain_select_expr",
    "chimerax_render_png",
    "render_chimerax_domain_colored_surface",
    "chimerax_renders_dir",
    "compose_conformation_pair_quadtych",
    "compose_map_pipeline_schematic",
    "compose_protein_row",
    "export_pipeline_panel_assets",
    "compose_triptych",
    "find_chimerax_executable",
    "generate_protein_figures",
    "resolve_protein_bundle",
    "run_chimerax_script",
    "write_domain_surface_cxc",
    "write_map_shell_surface_cxc",
    "write_statistic_surface_cxc",
]
