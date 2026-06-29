"""Unit tests for semi-prospective LOMO placement validation."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cryoem_mrc.placement_utility import (
    cohort_representative_roc,
    evaluate_locres_method_lomo_fold,
    evaluate_map_predictor,
    pooled_roc_curve,
    run_locres_method_lomo_validation,
    run_lomo_placement_validation,
    single_map_roc_curve,
)


def _frame(emdb_id: str, seed: int, n: int = 80) -> tuple[str, pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    rel = rng.uniform(0, 1, size=n)
    q = 0.25 + 0.55 * rel + rng.normal(0, 0.08, size=n)
    cc = 0.15 + 0.65 * rel + rng.normal(0, 0.07, size=n)
    loc = rng.uniform(2.5, 5.0, size=n)
    loc_res = loc + rng.normal(0, 0.2, size=n)
    loc_mono = loc + rng.normal(0, 0.15, size=n)
    v = 0.2 + 0.6 * rel + rng.normal(0, 0.05, size=n)
    var = rng.uniform(0.5, 4.0, size=n)
    zone = np.digitize(rel, [1 / 3, 2 / 3]) - 1
    df = pd.DataFrame(
        {
            "reliability_score": rel,
            "q_score": q,
            "windowed_halfmap_correlation": cc,
            "local_resolution": loc,
            "local_resolution_resmap": loc_res,
            "local_resolution_monores": loc_mono,
            "v_metric": v,
            "local_variance": var,
            "build_zone": zone,
            "in_contour_mask": True,
        }
    )
    return emdb_id, df, 3.0


class TestLomoPlacement(unittest.TestCase):
    def test_lomo_runs_with_three_maps(self) -> None:
        frames = [_frame("a", 0), _frame("b", 1), _frame("c", 2)]
        summary = run_lomo_placement_validation(frames, q_threshold=0.5)
        self.assertEqual(len({r.held_out_emdb_id for r in summary.fold_rows}), 3)
        rel = summary.predictor_medians["reliability_below_0_33"]
        self.assertGreater(rel["median_auc"], 0.55)

    def test_pooled_roc_auc_high_when_correlated(self) -> None:
        _eid, df, _ = _frame("x", 3, n=120)
        per_map = [("x", df)]
        curve = pooled_roc_curve(per_map, "reliability_below_0_33", q_threshold=0.5)
        self.assertGreater(curve.auc, 0.7)

    def test_representative_roc_median_matches_per_map(self) -> None:
        frames = [_frame("a", 10, n=120), _frame("b", 11, n=120), _frame("c", 12, n=120)]
        per_map = [(eid, df) for eid, df, _ in frames]
        eligible = frozenset({"a", "b", "c"})
        summary = cohort_representative_roc(
            per_map,
            "reliability_below_0_33",
            q_threshold=0.5,
            eligible_emdb_ids=eligible,
        )
        self.assertEqual(summary.n_maps, 3)
        aucs = [a for _, a in summary.per_map_aucs]
        self.assertAlmostEqual(summary.median_auc, float(np.median(aucs)))
        self.assertIn(summary.representative_emdb_id, {"a", "b", "c"})
        self.assertAlmostEqual(
            summary.representative_auc,
            dict(summary.per_map_aucs)[summary.representative_emdb_id],
        )
        single = single_map_roc_curve(
            dict(per_map)[summary.representative_emdb_id],
            "reliability_below_0_33",
            q_threshold=0.5,
        )
        self.assertEqual(summary.fpr, single.fpr)
        self.assertEqual(summary.tpr, single.tpr)

    def test_train_derived_locres_differs_from_in_map(self) -> None:
        frames = [_frame("a", 4), _frame("b", 5), _frame("c", 6)]
        test_df = frames[0][1]
        train_dfs = [frames[1][1], frames[2][1]]
        from cryoem_mrc.placement_utility import _train_medians

        loc_med, _ = _train_medians(train_dfs)
        ba_train, _, _, _, _, _ = evaluate_map_predictor(
            test_df,
            "locres_worse_than_median",
            q_threshold=0.5,
            train_locres_median=loc_med,
        )
        ba_inmap, _, _, _, _, _ = evaluate_map_predictor(
            test_df,
            "locres_worse_than_median",
            q_threshold=0.5,
        )
        self.assertTrue(np.isfinite(ba_train))
        self.assertTrue(np.isfinite(ba_inmap))


class TestLocresMethodLomo(unittest.TestCase):
    def test_parallel_lomo_runs_with_three_maps(self) -> None:
        frames = [_frame("a", 0), _frame("b", 1), _frame("c", 2)]
        summary = run_locres_method_lomo_validation(frames, q_threshold=0.5)
        self.assertEqual(len({r.held_out_emdb_id for r in summary.fold_rows}), 3)
        omit_meds = summary.predictor_medians["omit_zone"]
        self.assertGreater(omit_meds["median_auc"], 0.25)

    def test_global_threshold_differs_from_inmap_median(self) -> None:
        frames = [_frame("a", 4), _frame("b", 5), _frame("c", 6)]
        test_df = frames[0][1]
        gres = 2.5
        in_med = float(np.nanmedian(test_df["local_resolution"]))
        ba_inmap, _, _, _, _, _, _ = evaluate_locres_method_lomo_fold(
            test_df,
            "blocres_locres_inmap_median",
            q_threshold=0.5,
            global_resolution_a=gres,
        )
        ba_global, _, _, _, _, _, _ = evaluate_locres_method_lomo_fold(
            test_df,
            "blocres_locres_vs_global",
            q_threshold=0.5,
            global_resolution_a=gres,
        )
        self.assertTrue(np.isfinite(ba_inmap))
        self.assertTrue(np.isfinite(ba_global))
        self.assertNotEqual(in_med, gres)


if __name__ == "__main__":
    unittest.main()
