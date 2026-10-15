"""Comparisons over seeds, sweeps and the benchmark, on short sessions."""

from __future__ import annotations

import numpy as np
import pytest

from lobmm.analytics import ABLATIONS, DEFAULT_STRATEGIES, aggregate, bench, compare, run_many, sweep
from lobmm.config import SimConfig

SHORT = SimConfig().with_(**{"market.horizon": 90.0})


def test_run_many_gives_one_row_per_seed_with_overrides_applied():
    df = run_many(SHORT, seeds=[0, 1, 2], variant="x", **{"maker.strategy": "naive"})
    assert len(df) == 3 and df["seed"].tolist() == [0, 1, 2]
    assert (df["strategy"] == "naive").all() and (df["variant"] == "x").all()


def test_compare_and_aggregate():
    per_seed, agg = compare(SHORT, DEFAULT_STRATEGIES, seeds=[0, 1, 2, 3])
    assert set(per_seed["variant"]) == set(DEFAULT_STRATEGIES)
    assert list(agg.index) == list(DEFAULT_STRATEGIES)
    for name, g in per_seed.groupby("variant"):
        assert agg.loc[name, "pnl_mean"] == pytest.approx(g["pnl"].mean())
        assert agg.loc[name, "pnl_std"] == pytest.approx(g["pnl"].std(ddof=1))
        assert agg.loc[name, "seeds"] == 4
    assert "markout_10" in agg.columns and "adverse_10" in agg.columns and "pnl_per_inv_risk" in agg.columns
    assert aggregate(per_seed).equals(agg)


def test_sweep_adds_the_swept_value():
    per_seed, agg = sweep(SHORT, "maker.gamma", [0.05, 0.5], seeds=[0, 1])
    assert agg["value"].tolist() == [0.05, 0.5]
    assert list(agg.index) == ["gamma=0.05", "gamma=0.5"]
    assert len(per_seed) == 4


def test_ablation_variants_are_valid_overrides():
    for overrides in ABLATIONS.values():
        SHORT.with_(**overrides)


def test_bench_reports_speed():
    b = bench(SHORT, horizon=30.0)
    assert list(b.index) == ["none", "as"]
    assert (b["events_per_sec"] > 1000).all() and np.isfinite(b["runtime_s"]).all()
