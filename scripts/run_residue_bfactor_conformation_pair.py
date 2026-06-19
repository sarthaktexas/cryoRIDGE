"""Compare Cα RMSD vs Δreliability across two conformations (matched deposited models).

Each state uses its own deposited map, contour, and reliability table. Per-residue Cα RMSD
is computed after Kabsch alignment of state B onto state A. Residues are matched by mmCIF
(label_asym_id, label_seq_id, insertion).

Example::

    python scripts/run_residue_bfactor_conformation_pair.py --emd-a 23129 --emd-b 23130
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cryoem_mrc.conformation_pair import (
    compute_conformation_pair_coverage,
    compute_conformation_pair_stats,
    compute_per_residue_ca_rmsd,
    get_domain_assignments,
    get_domain_regions_for_pair,
    kabsch_align_coords,
    write_conformation_pair_md,
    write_conformation_pairs_summary,
)
from cryoem_mrc.repo_paths import COHORT_MANIFEST, conformation_pairs_dir
from cryoem_mrc.structure_validation import (
    default_reliability_out_dir,
    iter_ca_residues,
    load_cohort_manifest_row,
    match_residue_rows_by_key,
    read_residue_validation_csv,
    run_emdb_bfactor_validation,
)
from cryoem_mrc.thesis_figures import (
    DEFAULT_COUPLING_LAYOUT_THRESHOLD,
    compute_conformation_coupling,
    compute_coupling_layout_scores,
    compute_domain_coupling_block_colors,
    plot_conformation_pair_delta_reliability_supplement,
    plot_conformation_pair_domain_coupling_supplement,
    plot_conformation_pair_summary_triptych,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--emd-a", type=str, required=True)
    p.add_argument("--emd-b", type=str, required=True)
    p.add_argument("--manifest", type=Path, default=COHORT_MANIFEST)
    p.add_argument("--out-dir", type=Path, default=conformation_pairs_dir())
    p.add_argument("--window-radius", type=int, default=0)
    p.add_argument("--dpi", type=int, default=150)
    p.add_argument(
        "--layout",
        choices=("auto", "block", "domain"),
        default="auto",
        help="Main figure layout: auto from coupling layout score, or force block/domain",
    )
    p.add_argument(
        "--cluster-threshold",
        type=float,
        default=None,
        help="Block vs domain threshold (default: DEFAULT_COUPLING_LAYOUT_THRESHOLD)",
    )
    return p.parse_args(argv)


def _coverage_note(coverage) -> str:
    if coverage.coverage_flag:
        return (
            f"coverage A {coverage.missing_pct_a:.0f}% / B {coverage.missing_pct_b:.0f}% missing"
        )
    return (
        f"coverage OK (A {100 * coverage.frac_analysis_of_a:.0f}%, "
        f"B {100 * coverage.frac_analysis_of_b:.0f}% of deposited Cα)"
    )


def _ensure_residue_validation(emd_id: str, *, manifest: Path, window_radius: int) -> int:
    rel_dir = default_reliability_out_dir(emd_id)
    residue_csv = rel_dir / "residue_validation.csv"
    try:
        if not residue_csv.is_file():
            code, _, _, _ = run_emdb_bfactor_validation(
                emd_id,
                manifest=manifest,
                window_radius=window_radius,
                require_b_factor_source=False,
            )
            if code != 0:
                return code
    except (FileNotFoundError, ValueError) as e:
        print(f"[conformation_pair] ERROR: {e}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    pair_name = f"emd_{args.emd_a}_vs_{args.emd_b}"
    out_dir = args.out_dir / pair_name
    out_dir.mkdir(parents=True, exist_ok=True)

    row_a = load_cohort_manifest_row(args.manifest, args.emd_a)
    row_b = load_cohort_manifest_row(args.manifest, args.emd_b)
    pdb_a = Path(row_a["flexibility_path_or_pdb"])
    pdb_b = Path(row_b["flexibility_path_or_pdb"])

    for emd_id in (args.emd_a, args.emd_b):
        if _ensure_residue_validation(emd_id, manifest=args.manifest, window_radius=args.window_radius) != 0:
            return 2

    try:
        rows_a = read_residue_validation_csv(default_reliability_out_dir(args.emd_a) / "residue_validation.csv")
        rows_b = read_residue_validation_csv(default_reliability_out_dir(args.emd_b) / "residue_validation.csv")
    except FileNotFoundError as e:
        print(f"[conformation_pair] ERROR: {e}", file=sys.stderr)
        return 2

    pairs = match_residue_rows_by_key(rows_a, rows_b)
    pair_stats = compute_conformation_pair_stats(
        pairs, emdb_a=args.emd_a, emdb_b=args.emd_b, in_mask_both=True
    )
    coverage = compute_conformation_pair_coverage(
        pairs,
        emdb_a=args.emd_a,
        emdb_b=args.emd_b,
        n_ca_total_a=len(iter_ca_residues(pdb_a)),
        n_ca_total_b=len(iter_ca_residues(pdb_b)),
    )
    write_conformation_pair_md(out_dir / "CONFORMATION_PAIR.md", pair_stats, coverage=coverage)

    use, _rmsd = compute_per_residue_ca_rmsd(pairs, in_mask_both=True)
    layout_scores = {
        "diagonal_coupling_score": float("nan"),
        "domain_coupling_score": float("nan"),
        "coupling_layout_score": float("nan"),
        "hierarchical_cluster_score": float("nan"),
    }
    figure_layout = "block"
    recommended_layout = "domain"
    cluster_threshold = (
        args.cluster_threshold
        if args.cluster_threshold is not None
        else DEFAULT_COUPLING_LAYOUT_THRESHOLD
    )

    if len(use) >= 10:
        rho = pair_stats.spearman_rmsd_vs_delta_reliability
        cov_note = _coverage_note(coverage)

        coupling_data = compute_conformation_coupling(pairs, in_mask_both=True)
        if coupling_data is not None:
            layout_scores = compute_coupling_layout_scores(
                coupling_data["interior_corr"],
                emdb_a=args.emd_a,
                emdb_b=args.emd_b,
                interior_use=coupling_data["interior_use"],
            )

        if coupling_data is not None:
            use_full = coupling_data["use"]
            coords_a = np.array([[a.x, a.y, a.z] for a, _ in use_full], dtype=np.float64)
            coords_b = np.array([[b.x, b.y, b.z] for _, b in use_full], dtype=np.float64)
            coords_b_aligned, _ = kabsch_align_coords(coords_b, coords_a)

            has_domains = bool(get_domain_regions_for_pair(args.emd_a, args.emd_b))
            summary_name = (
                "conformation_pair_summary.png"
                if has_domains
                else "conformation_pair_summary_triptych.png"
            )

            chimerax_domain_png = None
            chimerax_coupling_png = None
            if has_domains:
                from cryoem_mrc.chimerax_figures import (
                    ensure_chimerax_domain_render,
                    render_chimerax_domain_colored_surface,
                )

                chimerax_domain_png = ensure_chimerax_domain_render(args.emd_a, preview=True)
                regions = get_domain_regions_for_pair(args.emd_a, args.emd_b)
                domain_order = [reg.name for reg in regions]
                use_int = coupling_data["interior_use"]
                assignments = get_domain_assignments(use_int, regions)
                block_hex, _ = compute_domain_coupling_block_colors(
                    coupling_data["interior_corr"], assignments, domain_order
                )
                chimerax_coupling_png = render_chimerax_domain_colored_surface(
                    args.emd_a,
                    domain_colors=block_hex,
                    out_png=out_dir / f"chimerax_emd_{args.emd_a}_domain_coupling.png",
                    preview=True,
                )

            triptych, recommended_layout = plot_conformation_pair_summary_triptych(
                pairs,
                emdb_a=args.emd_a,
                emdb_b=args.emd_b,
                in_mask_both=True,
                spearman_rho=rho,
                coverage_note=cov_note,
                coords_b_aligned=coords_b_aligned,
                cluster_separation_threshold=cluster_threshold,
                layout=args.layout,
                manifest=args.manifest,
                include_structure_panel=has_domains,
                chimerax_domain_png=chimerax_domain_png,
                chimerax_coupling_png=chimerax_coupling_png,
                save_path=out_dir / summary_name,
                dpi=args.dpi,
            )
            if triptych is not None:
                plt.close(triptych)

            if has_domains:
                delta_supp = plot_conformation_pair_delta_reliability_supplement(
                    pairs,
                    emdb_a=args.emd_a,
                    emdb_b=args.emd_b,
                    in_mask_both=True,
                    coverage_note=cov_note,
                    manifest=args.manifest,
                    save_path=out_dir / "conformation_pair_delta_reliability_supplement.png",
                    dpi=args.dpi,
                )
                if delta_supp is not None:
                    plt.close(delta_supp)

            supplement = plot_conformation_pair_domain_coupling_supplement(
                pairs,
                emdb_a=args.emd_a,
                emdb_b=args.emd_b,
                in_mask_both=True,
                coverage_note=cov_note,
                manifest=args.manifest,
                save_path=out_dir / "conformation_pair_domain_coupling_supplement.png",
                dpi=args.dpi,
            )
            if supplement is not None:
                plt.close(supplement)

    (out_dir / "conformation_pair_stats.json").write_text(
        json.dumps(
            {
                "emdb_a": pair_stats.emdb_a,
                "emdb_b": pair_stats.emdb_b,
                "n_matched": pair_stats.n_matched,
                "n_matched_in_mask_both": pair_stats.n_matched_in_mask_both,
                "median_ca_rmsd_a": pair_stats.median_ca_rmsd_a,
                "spearman_rmsd_vs_delta_reliability": pair_stats.spearman_rmsd_vs_delta_reliability,
                "diagonal_coupling_score": layout_scores["diagonal_coupling_score"],
                "domain_coupling_score": layout_scores["domain_coupling_score"],
                "coupling_layout_score": layout_scores["coupling_layout_score"],
                "hierarchical_cluster_score": layout_scores["hierarchical_cluster_score"],
                "cluster_separation_score": layout_scores["coupling_layout_score"],
                "coupling_layout_threshold": cluster_threshold,
                "cluster_separation_threshold": cluster_threshold,
                "figure_layout": figure_layout,
                "recommended_layout": recommended_layout,
                "n_ca_total_a": coverage.n_ca_total_a,
                "n_ca_total_b": coverage.n_ca_total_b,
                "frac_analysis_of_a": coverage.frac_analysis_of_a,
                "frac_analysis_of_b": coverage.frac_analysis_of_b,
                "missing_pct_a": coverage.missing_pct_a,
                "missing_pct_b": coverage.missing_pct_b,
                "coverage_flag": coverage.coverage_flag,
            },
            indent=2,
        )
        + "\n"
    )

    flag = " [COVERAGE FLAG]" if coverage.coverage_flag else ""
    layout_txt = ""
    if len(use) >= 10:
        layout = layout_scores["coupling_layout_score"]
        diag = layout_scores["diagonal_coupling_score"]
        domain = layout_scores["domain_coupling_score"]
        hier = layout_scores["hierarchical_cluster_score"]
        layout_txt = (
            f" main=cluster_matrix recommended={recommended_layout} "
            f"layout={layout:+.3f} diag={diag:+.3f} domain={domain:+.3f} hier={hier:+.3f}"
        )
    print(
        f"[conformation_pair] matched={pair_stats.n_matched} in-mask={pair_stats.n_matched_in_mask_both} "
        f"ρ(RMSD,Δrel)={pair_stats.spearman_rmsd_vs_delta_reliability:+.3f} "
        f"median RMSD={pair_stats.median_ca_rmsd_a:.2f} Å "
        f"missing A={coverage.missing_pct_a:.1f}% B={coverage.missing_pct_b:.1f}%{layout_txt}{flag}",
        flush=True,
    )
    csv_path, md_path = write_conformation_pairs_summary(args.out_dir, manifest=args.manifest)
    if csv_path is not None and md_path is not None:
        print(f"[conformation_pair] updated {csv_path.name} and {md_path.name}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
