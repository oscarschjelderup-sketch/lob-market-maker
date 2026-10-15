"""Whole sessions: reproducible, the book stays consistent through every event, the maker obeys its limits."""

from __future__ import annotations

import numpy as np
import pytest

from lobmm.book import BUY, SELL
from lobmm.config import SimConfig
from lobmm.sim import Simulator, run_session

SHORT = SimConfig().with_(**{"market.horizon": 120.0})


class CheckedSimulator(Simulator):
    """Checks the invariants of the book after every single event."""

    def _record_mid(self, t: float) -> None:
        self.book.check_invariants()
        super()._record_mid(t)


@pytest.mark.parametrize("strategy", ["none", "naive", "as", "as_adverse", "probe"])
def test_book_invariants_hold_through_a_session(strategy):
    res = CheckedSimulator(SHORT.with_(**{"maker.strategy": strategy, "sample_interval": 10.0}), seed=1).run()
    assert res.n_events > 500
    assert len(res.mid_t) == len(res.mid) and np.all(np.diff(res.mid_t) >= 0)
    if strategy == "none":
        assert res.ledger.n_fills == 0 and res.series.empty
    else:
        assert res.ledger.n_fills > 0


def test_same_seed_same_session_and_different_seeds_differ():
    a, b, c = (run_session(SHORT, seed=s) for s in (4, 4, 5))
    assert a.n_events == b.n_events and a.ledger.n_fills == b.ledger.n_fills
    assert a.ledger.equity(a.end_mid) == b.ledger.equity(b.end_mid)
    assert np.array_equal(a.mid, b.mid) and a.fills.equals(b.fills)
    assert not np.array_equal(a.mid, c.mid)


def test_maker_quotes_never_cross_and_sit_on_or_behind_the_touch():
    res = run_session(SHORT, seed=2)
    s = res.series.dropna(subset=["bid", "ask"])
    assert len(s) > 50
    assert (s["bid"] < s["ask"]).all()
    assert (s["bid"] <= s["best_bid"]).all() and (s["ask"] >= s["best_ask"]).all()


def test_inventory_stays_within_the_limit_plus_one_fill():
    cfg = SHORT.with_(**{"maker.strategy": "naive", "maker.half_spread": 1, "maker.max_inventory": 10, "maker.size": 5})
    res = run_session(cfg, seed=0)
    assert res.series["inventory"].abs().max() <= 10 + 5
    assert res.series["inventory"].abs().max() >= 10, "the limit should actually bind in a 1-tick naive session"


def test_fills_are_marked_against_the_mid_before_the_trade():
    res = run_session(SHORT, seed=0)
    f = res.fills
    assert len(f) > 0
    buys, sells = f[f["side"] == BUY], f[f["side"] == SELL]
    assert (buys["price"] <= buys["mid"]).all() and (sells["price"] >= sells["mid"]).all(), "a post-only quote is never through the mid"
    assert res.attribution()["total"] == pytest.approx(res.ledger.equity(res.end_mid))


def test_kill_switch_stops_the_quoting():
    cfg = SHORT.with_(**{"maker.kill_drawdown": 0.001})
    res = run_session(cfg, seed=0)
    assert res.killed_at is not None
    assert (res.fills["ts"] <= res.killed_at).all()
    after = res.series[res.series["t"] > res.killed_at]
    assert after.empty, "no more wakes once killed"
    assert res.summary()["killed_at"] == res.killed_at


def test_latency_delays_the_orders_and_still_fills():
    slow = run_session(SHORT, seed=0, **{"maker.latency": 0.3})
    fast = run_session(SHORT, seed=0)
    assert slow.n_events > fast.n_events, "decide and order events are added to the queue"
    assert slow.ledger.n_fills > 0
    assert slow.series.dropna(subset=["bid", "ask"]).pipe(lambda s: (s["bid"] < s["ask"]).all())


def test_mid_tracks_the_fair_value():
    res = run_session(SimConfig().with_(**{"market.horizon": 600.0, "maker.strategy": "none"}), seed=0)
    idx = np.searchsorted(res.mid_t, res.fair_t, side="right") - 1
    err = res.mid[np.clip(idx, 0, len(res.mid) - 1)] - res.fair
    assert abs(err.mean()) < 2.0 and err.std() < 5.0, "the informed flow should keep the book near the fair value"


def test_drift_after_a_fill_is_about_zero_without_informed_traders_and_negative_with_them():
    """The markout against the pre-trade mid is negative in both worlds, because the order that fills the maker
    moves the touch (mechanical impact). Once that is taken out, the drift of the mid over the next minute is
    about zero when nobody knows anything and negative when informed traders are still catching up with news.
    Fixed quotes (the naive maker) so the two worlds are compared at the same distance from the mid."""
    cfg = SimConfig().with_(**{"market.horizon": 1800.0, "maker.strategy": "naive"})
    quiet = [run_session(cfg, seed=s, **{"flow.informed_rate": 0.0}).summary() for s in range(8)]
    informed = [run_session(cfg, seed=s).summary() for s in range(8)]
    assert np.mean([r["impact"] for r in quiet]) < -0.005 and np.mean([r["impact"] for r in informed]) < -0.005
    assert np.mean([r["markout_10"] for r in quiet]) < 0
    quiet_drift = np.mean([r["drift_60"] for r in quiet])
    informed_drift = np.mean([r["drift_60"] for r in informed])
    assert abs(quiet_drift) < 0.004, f"drift without informed traders should be about zero, got {quiet_drift:.4f}"
    assert informed_drift < quiet_drift - 0.003, f"informed drift {informed_drift:.4f} should be clearly below {quiet_drift:.4f}"
    mk = run_session(cfg, seed=0).markouts()
    for h in (1, 10, 60):
        valid = mk[f"markout_{h}"].notna()
        assert valid.sum() > 100
        assert np.allclose(mk.loc[valid, f"markout_{h}"], mk.loc[valid, "impact"] + mk.loc[valid, f"drift_{h}"]), "markout = impact + drift, per fill"
        assert np.allclose(mk.loc[valid, f"realised_{h}"], mk.loc[valid, "side"] * (mk.loc[valid, "mid"] - mk.loc[valid, "price"]) * 0.01 + mk.loc[valid, f"markout_{h}"])


def test_summary_has_the_expected_keys_and_finite_numbers():
    s = run_session(SHORT, seed=0).summary()
    for key in ("pnl", "spread_capture", "inventory_pnl", "fees", "n_fills", "inv_std", "avg_spread", "mid_vol", "events_per_sec", "markout_1", "adverse_10"):
        assert key in s and np.isfinite(s[key])
    assert s["pnl"] == pytest.approx(s["spread_capture"] + s["inventory_pnl"] - s["fees"])


def test_depth_samples_are_recorded_when_asked():
    res = run_session(SHORT, seed=0, sample_interval=5.0)
    assert len(res.samples) == pytest.approx(120 / 5, abs=1)
    t, mid, prices, bids, asks = res.samples[3]
    assert len(prices) == 2 * SHORT.sample_levels + 1 and bids.sum() > 0 and asks.sum() > 0
    assert prices[0] < mid < prices[-1]
    assert run_session(SHORT, seed=0).samples == []


def test_side_constants():
    assert BUY == -SELL == 1
