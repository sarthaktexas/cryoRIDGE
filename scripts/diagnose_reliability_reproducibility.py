"""Diagnose why cryoridge reliability outputs differ between runs or machines.

Typical root causes this script checks:
1. Algorithm change (pre-v0.7 ranked H_repro; v0.7+ ranks smoothness / V)
2. Different feature input map (avg half vs deposited primary)
3. Different density-source flag (avg_half vs primary)
4. Input map byte differences
5. Discrete zone flips from tiny score drift

Example::

    python scripts/diagnose_reliability_reproducibility.py \\
      --reference emd_49450.map --half1 emd_49450_half_map_1.map --half2 emd_49450_half_map_2.map \\
      --contour 0.116 \\
      --candidate cryoridge_out/emd_49450_half_map_1_reliability.mrc \\
      --reference-output "/Volumes/Undergrad Thesis/thesis-data/outputs/emd_49450/halfmap-qc/reliability/emd_49450_reliability.mrc" \\
      --reference-npz "/Volumes/Undergrad Thesis/thesis-data/outputs/emd_49450/halfmap-qc/reliability/reliability.npz"
"""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import platform
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from cryoem_mrc.analysis import build_contour_mask
from cryoem_mrc.density_source import zscore_halfmap_average
from cryoem_mrc.io import load_mrc
from cryoem_mrc.map_grid import load_full_and_half_maps
from cryoem_mrc.mask_bbox import bbox_from_mask, crop_array, embed_array, pad_voxels_for_filters
from cryoem_mrc.mrc_compare import compare_mrc_files
from cryoem_mrc.reliability import percentile_rank_in_mask, windowed_smoothness


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference", required=True, type=Path)
    p.add_argument("--half1", required=True, type=Path)
    p.add_argument("--half2", required=True, type=Path)
    p.add_argument("--contour", required=True, type=float)
    p.add_argument("--window", type=int, default=5)
    p.add_argument("--candidate", type=Path, help="New/local reliability_score MRC")
    p.add_argument("--reference-output", type=Path, help="Older Arc reliability_score MRC")
    p.add_argument("--reference-npz", type=Path, help="Older reliability.npz bundle (optional)")
    p.add_argument("--features", type=Path, help="Local features.npz to compare input map scale")
    p.add_argument("--arc-features", type=Path, help="Arc-era features.npz for comparison")
    return p.parse_args(argv)


def _env_lines() -> list[str]:
    lines = [
        f"python: {sys.version.split()[0]} ({platform.platform()})",
        f"numpy: {np.__version__}",
    ]
    for pkg in ("scipy", "mrcfile", "cryoridge"):
        try:
            lines.append(f"{pkg}: {importlib.metadata.version(pkg)}")
        except importlib.metadata.PackageNotFoundError:
            pass
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        if var in os.environ:
            lines.append(f"{var}={os.environ[var]}")
    return lines


def _recompute_current(
    reference: Path,
    half1: Path,
    half2: Path,
    *,
    contour: float,
    window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    ref = load_mrc(reference, dtype=np.float32)
    mask = build_contour_mask(ref, contour)
    bundle = load_full_and_half_maps(
        reference, half1, half2, reference="full", dtype=np.float32, resample_if_needed=True
    )
    rho = zscore_halfmap_average(bundle.half1.data, bundle.half2.data)
    pad = pad_voxels_for_filters(window=window)
    bbox = bbox_from_mask(mask, pad=pad)
    smooth = windowed_smoothness(crop_array(rho, bbox), window=window)
    score = percentile_rank_in_mask(smooth, crop_array(mask, bbox))
    full_score = embed_array(ref.shape, bbox, score, dtype=np.float32)
    full_smooth = embed_array(ref.shape, bbox, smooth.astype(np.float32), dtype=np.float32)
    return ref, mask, full_score, full_smooth


def _masked_stats(a: np.ndarray, b: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))[mask]
    sp, _ = spearmanr(a[mask], b[mask])
    return float(d.max()), float(d.mean()), float(sp)


def _print_ranking_diagnosis(npz_path: Path, mask: np.ndarray) -> None:
    data = np.load(npz_path, allow_pickle=False)
    if "reliability_score" not in data.files:
        print(f"[ranking] {npz_path} has no reliability_score", flush=True)
        return
    score = data["reliability_score"]
    print(f"[ranking] bundle: {npz_path}", flush=True)
    for key in ("reliability_smoothness", "reliability_H_repro", "reliability_fluctuation"):
        if key not in data.files:
            continue
        ranked = percentile_rank_in_mask(data[key], mask)
        mx, mn, sp = _masked_stats(ranked, score, mask)
        tag = "MATCHES" if mx == 0.0 else "differs"
        print(
            f"  rank({key}) vs reliability_score: max|Δ|={mx:.6g} "
            f"mean|Δ|={mn:.6g} spearman={sp:.8f} -> {tag}",
            flush=True,
        )


def _feature_scale(path: Path, label: str) -> None:
    if not path.is_file():
        print(f"[features] {label}: missing {path}", flush=True)
        return
    with np.load(path, allow_pickle=False) as d:
        if "density_raw" not in d.files:
            print(f"[features] {label}: no density_raw in {path.name}", flush=True)
            return
        raw = d["density_raw"]
        print(
            f"[features] {label}: density_raw min/max = "
            f"{float(raw.min()):.4g} / {float(raw.max()):.4g}",
            flush=True,
        )


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    print("=== environment ===", flush=True)
    for line in _env_lines():
        print(line, flush=True)

    print("\n=== grid alignment ===", flush=True)
    bundle = load_full_and_half_maps(
        args.reference,
        args.half1,
        args.half2,
        reference="full",
        dtype=np.float32,
        resample_if_needed=True,
    )
    for name, rep in bundle.reports.items():
        status = "ok" if rep.ok else "RESAMPLED"
        print(f"  {name}: {status}", flush=True)
        for msg in rep.messages:
            print(f"    {msg}", flush=True)

    ref, mask, current_score, current_smooth = _recompute_current(
        args.reference,
        args.half1,
        args.half2,
        contour=args.contour,
        window=args.window,
    )
    print(f"\n=== current code (v0.7+ smoothness ranking) ===", flush=True)
    print(f"  in-mask voxels: {int(mask.sum()):,}", flush=True)
    print(
        f"  smoothness in mask: min/max = "
        f"{float(current_smooth[mask].min()):.4g} / {float(current_smooth[mask].max()):.4g}",
        flush=True,
    )

    if args.candidate and args.candidate.is_file():
        cand = load_mrc(args.candidate, dtype=np.float32)
        mx, mn, sp = _masked_stats(cand, current_score, mask)
        print(f"\n=== candidate MRC vs current recompute ===", flush=True)
        print(f"  {args.candidate}", flush=True)
        print(f"  max|Δ|={mx:.6g} mean|Δ|={mn:.6g} spearman={sp:.8f}", flush=True)
        if mx == 0.0:
            print("  -> candidate matches current cryoridge definition", flush=True)

    if args.reference_output and args.reference_output.is_file():
        arc = load_mrc(args.reference_output, dtype=np.float32)
        mx, mn, sp = _masked_stats(current_score, arc, mask)
        print(f"\n=== current recompute vs reference MRC ===", flush=True)
        print(f"  {args.reference_output}", flush=True)
        print(f"  max|Δ|={mx:.6g} mean|Δ|={mn:.6g} spearman={sp:.8f}", flush=True)
        mx_s, _, _ = _masked_stats(current_smooth, arc, mask)
        if mx_s > 0 and mx > 0:
            print("  note: score differs but check smoothness ranking diagnosis below", flush=True)

    if args.reference_npz and args.reference_npz.is_file():
        print("\n=== ranking definition diagnosis ===", flush=True)
        _print_ranking_diagnosis(args.reference_npz, mask)
        with np.load(args.reference_npz, allow_pickle=False) as d:
            if "reliability_smoothness" in d.files:
                arc_smooth = d["reliability_smoothness"]
                mx, mn, sp = _masked_stats(current_smooth, arc_smooth, mask)
                print(
                    f"  current smoothness vs arc smoothness: max|Δ|={mx:.6g} "
                    f"mean|Δ|={mn:.6g} spearman={sp:.8f}",
                    flush=True,
                )
                if mx == 0.0:
                    print(
                        "  -> smoothness (V) is identical; score gap is from ranking target "
                        "(H_repro on Arc vs smoothness now)",
                        flush=True,
                    )

    if args.features or args.arc_features:
        print("\n=== feature input map scale ===", flush=True)
        if args.features:
            _feature_scale(args.features, "local")
        if args.arc_features:
            _feature_scale(args.arc_features, "arc")
        print(
            "  avg-half features usually have density_raw max ~0.3–0.8; "
            "deposited primary is often >1.0",
            flush=True,
        )

    if args.candidate and args.reference_output:
        print("\n=== raw MRC compare ===", flush=True)
        report = compare_mrc_files(args.candidate, args.reference_output, rtol=0.0, atol=0.0)
        for line in report.summary_lines():
            print(f"  {line}", flush=True)

    print("\n=== interpretation ===", flush=True)
    print(
        "  pre-v0.7.0: reliability_score = percentile_rank(H_repro)\n"
        "  v0.7.0+:    reliability_score = percentile_rank(smoothness / V)\n"
        "  If smoothness matches but score differs, you are seeing the v0.7 definition change.",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
