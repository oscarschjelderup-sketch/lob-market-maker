"""Cash, inventory, the attribution identity and the markouts, on fills written by hand and at random."""

from __future__ import annotations

import numpy as np
import pytest

from lobmm.accounting import Ledger
from lobmm.book import BUY, SELL


def test_round_trip_inside_the_spread_earns_the_spread():
    led = Ledger(tick=0.01)
    led.on_fill(0.0, BUY, 9999, 10, mid=10000.0)
    assert led.inventory == 10 and led.cash == pytest.approx(-999.9)
    led.on_fill(1.0, SELL, 10001, 10, mid=10000.0)
    assert led.inventory == 0 and led.cash == pytest.approx(0.2)
    assert led.equity(10000.0) == pytest.approx(0.2)
    assert led.spread_capture() == pytest.approx(0.2)
    assert led.n_fills == 2 and led.volume == 20


def test_attribution_splits_spread_from_inventory():
    led = Ledger(tick=0.01)
    mid_t = np.array([0.0, 1.0])
    mid = np.array([10000.0, 10004.0])
    led.on_fill(0.0, BUY, 9999, 10, mid=10000.0)  # bought one tick below the mid
    led.on_fill(2.0, SELL, 10005, 10, mid=10004.0)  # sold one tick above the (new) mid
    att = led.attribution(mid_t, mid, end_mid=10004.0)
    assert att["spread_capture"] == pytest.approx(0.2)
    assert att["inventory"] == pytest.approx(0.4), "10 shares held through a 4-tick move"
    assert att["total"] == pytest.approx(0.6)
    assert att["fees"] == 0.0


def test_fees_reduce_cash_and_are_reported():
    led = Ledger(tick=0.01, maker_fee_bps=1.0)
    led.on_fill(0.0, BUY, 10000, 100, mid=10000.0)
    assert led.fees == pytest.approx(10000 * 100 * 0.01 * 1e-4)
    assert led.cash == pytest.approx(-10000.0 - led.fees)
    rebate = Ledger(tick=0.01, maker_fee_bps=-1.0)
    rebate.on_fill(0.0, BUY, 10000, 100, mid=10000.0)
    assert rebate.fees < 0 and rebate.cash == pytest.approx(-10000.0 - rebate.fees)


def test_markouts_follow_the_mid_and_are_nan_past_the_end():
    led = Ledger(tick=0.01)
    led.on_fill(0.0, BUY, 9999, 5, mid=10000.0)
    led.on_fill(0.0, SELL, 10001, 5, mid=10000.0)
    led.set_mid_after(1, 10001.0)  # the sell was hit by an order that lifted the touch by one tick
    mid_t = np.array([0.0, 1.0, 5.0])
    mid = np.array([10000.0, 10002.0, 9998.0])
    mk = led.markouts(mid_t, mid, horizons=(0.5, 2.0, 10.0), end_time=8.0)
    assert mk["markout_0.5"].tolist() == pytest.approx([0.0, 0.0])
    assert mk["markout_2"].tolist() == pytest.approx([0.02, -0.02]), "the buy gains and the sell loses from the +2 move"
    assert mk["markout_10"].isna().all(), "beyond the end of the session nothing is known"
    assert mk["impact"].tolist() == pytest.approx([0.0, -0.01]), "only the sell moved the touch, against the maker"
    assert mk["drift_2"].tolist() == pytest.approx([0.02, -0.01]), "markout = impact + drift"
    assert mk["realised_0.5"].tolist() == pytest.approx([0.01, 0.01]), "one tick inside the mid, still worth a tick"
    assert mk["realised_2"].tolist() == pytest.approx([0.03, -0.01])
    assert mk["realised_10"].isna().all()
    assert mk["mid_after"].tolist() == [10000.0, 10001.0]
    empty = Ledger(tick=0.01).markouts(mid_t, mid, horizons=(1.0,), end_time=8.0)
    assert empty.empty and {"impact", "markout_1", "drift_1", "realised_1"} <= set(empty.columns)


def test_identity_holds_on_random_fills_and_mid_paths():
    rng = np.random.default_rng(3)
    for _ in range(20):
        led = Ledger(tick=0.01, maker_fee_bps=float(rng.uniform(-1, 1)))
        n_moves = int(rng.integers(5, 60))
        mid_t = np.sort(rng.uniform(0, 100, n_moves))
        mid_t[0] = 0.0
        mid = 10000.0 + np.cumsum(rng.integers(-3, 4, n_moves)) / 2.0
        n_fills = int(rng.integers(1, 40))
        for ts in np.sort(rng.uniform(0, 100, n_fills)):
            j = int(np.searchsorted(mid_t, ts, side="right") - 1)
            side = BUY if rng.random() < 0.5 else SELL
            price = int(mid[j] - side * rng.integers(1, 4))
            led.on_fill(float(ts), side, price, int(rng.integers(1, 10)), mid=float(mid[j]))
        att = led.attribution(mid_t, mid, end_mid=float(mid[-1]))
        assert att["total"] == pytest.approx(att["spread_capture"] + att["inventory"] - att["fees"], abs=1e-9)
