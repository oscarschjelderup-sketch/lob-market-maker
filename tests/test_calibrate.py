"""The calibration recovers known parameters from synthetic counts, and works end to end on probe sessions."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from lobmm.analytics import realised_vol
from lobmm.calibrate import calibrate, fit_intensity, fit_markout
from lobmm.config import SimConfig


def test_fit_recovers_a_known_intensity_curve():
    A, k, tick = 0.5, 40.0, 0.01
    exposure = {float(d): 1000.0 for d in range(1, 9)}
    fills = {d: int(round(A * np.exp(-k * d * tick) * 1000.0)) for d in exposure}
    fit = fit_intensity(exposure, fills, tick)
    assert fit.k == pytest.approx(k, rel=0.03)
    assert fit.A == pytest.approx(A, rel=0.05)
    assert fit.k_ticks == pytest.approx(fit.k * tick)
    assert fit.r2 > 0.99
    assert fit.intensity(0.0) == pytest.approx(fit.A)
    assert list(fit.table.columns) == ["distance_ticks", "exposure_s", "fills", "intensity", "fitted"]


def test_fit_refuses_too_little_data():
    with pytest.raises(ValueError):
        fit_intensity({1.0: 100.0}, {1.0: 5}, 0.01)
    with pytest.raises(ValueError):
        fit_intensity({1.0: 100.0, 2.0: 1.0}, {1.0: 5, 2.0: 1}, 0.01)


def _cost_fills(a: float, b: float, tick: float = 0.01, per_bucket: int = 30, stale: float = 0.002) -> pd.DataFrame:
    """Fills whose realised spread is exactly distance - (a + b * distance), ``stale`` of the cost before the fill."""
    rows = []
    for d in range(1, 9):
        cost = a + b * d * tick
        for _ in range(per_bucket):
            rows.append({"distance": float(d), "qty": 5, "markout": -(cost - stale), "realised": d * tick - cost})
    return pd.DataFrame(rows)


def test_markout_fit_recovers_a_known_cost_line_and_the_closed_form_optimum():
    tick, k = 0.01, 40.0
    a, b = 0.005, 0.4  # half a tick at zero distance, then 40 % of every extra tick of spread lost again
    fit = fit_markout(_cost_fills(a, b), tick, horizon=10.0)
    assert fit.horizon == 10.0
    assert fit.a == pytest.approx(a) and fit.b == pytest.approx(b) and fit.r2 == pytest.approx(1.0)
    assert fit.cost(0.03) == pytest.approx(a + b * 0.03)
    assert fit.adverse_term == pytest.approx(a / (1 - b))
    assert fit.optimal_distance(k) == pytest.approx(1 / k + a / (1 - b))
    grid = np.linspace(0.0, 0.2, 20001)
    profit = np.exp(-k * grid) * (grid - fit.cost(grid))
    assert grid[np.argmax(profit)] == pytest.approx(fit.optimal_distance(k), abs=2e-5), "the closed form is the argmax of lambda (delta - cost)"
    assert list(fit.table.columns) == ["distance_ticks", "fills", "shares", "markout_ticks", "stale_ticks", "realised_ticks", "cost_ticks", "fitted_ticks"]
    table = fit.table.set_index("distance_ticks")
    assert (table["fills"] == 30).all() and (table["shares"] == 150).all()
    assert table.loc[3.0, "cost_ticks"] == pytest.approx((a + b * 0.03) / tick)
    assert table.loc[3.0, "stale_ticks"] == pytest.approx(0.2), "the part of the cost the mid had closed before the fill"
    assert table.loc[3.0, "markout_ticks"] == pytest.approx(-(a + b * 0.03 - 0.002) / tick)
    assert table.loc[3.0, "realised_ticks"] + table.loc[3.0, "cost_ticks"] == pytest.approx(3.0)
    assert np.allclose(table["fitted_ticks"], table["cost_ticks"])


def test_markout_fit_refuses_too_little_data_and_a_cost_that_outruns_the_distance():
    one_bucket = pd.DataFrame({"distance": [1.0] * 30, "qty": [1] * 30, "markout": [-0.01] * 30, "realised": [0.0] * 30})
    with pytest.raises(ValueError):
        fit_markout(one_bucket, 0.01, 10.0)
    fit = fit_markout(_cost_fills(0.0, 2.0), 0.01, 10.0)  # every tick of spread loses two
    assert fit.b == pytest.approx(2.0)
    with pytest.raises(ValueError):
        _ = fit.adverse_term
    assert fit_markout(_cost_fills(0.0, 0.5), 0.01, 10.0).adverse_term == pytest.approx(0.0), "no cost at zero distance: nothing to add"


def test_realised_vol_of_a_random_walk():
    rng = np.random.default_rng(0)
    steps = rng.normal(0, 0.7, 5000)
    mid_t = np.arange(5000, dtype=float)
    mid = 10000 + np.cumsum(steps)
    assert realised_vol(mid_t, mid, 5000.0, dt=1.0) == pytest.approx(0.7, rel=0.05)
    assert realised_vol(mid_t, mid, 5000.0, dt=4.0) == pytest.approx(0.7, rel=0.15), "a random walk has a flat signature"


def test_calibration_end_to_end_on_short_probe_sessions():
    cfg = SimConfig().with_(**{"market.horizon": 600.0})  # shorter sessions leave the far buckets too thin to fit the cost
    cal = calibrate(cfg, seeds=(0, 1), vol_dt=10.0)
    assert cal.sessions == 2 and cal.vol_dt == 10.0
    assert cal.fair_vol_ticks == pytest.approx(np.sqrt(0.5**2 + 64.0 / 120.0)), "diffusion plus news"
    assert cal.sigma_ticks > 0 and cal.sigma == pytest.approx(cal.sigma_ticks * cfg.market.tick)
    assert cal.fit.k > 0 and cal.fit.A > 0
    table = cal.fit.table.set_index("distance_ticks")
    near = table.loc[table.index <= 2.0, "intensity"].mean()
    far = table.loc[table.index >= 6.0, "intensity"].mean()
    assert near > far, "fills come faster close to the mid"
    assert set(cal.signature.index) >= {1.0, 10.0, 60.0}
    assert cal.signature[1.0] > cal.signature[60.0], "microstructure noise inflates the short-horizon volatility"
    m = cal.markout
    assert m.horizon == 10.0 and np.isfinite(m.a) and np.isfinite(m.b)
    assert 0.0 < m.b < 1.0, "the cost of a fill grows with the distance, but slower than the distance"
    assert m.adverse_term > 0, "there is something to add to the spread"
    assert (m.table["fills"] > 0).all() and (m.table["realised_ticks"] + m.table["cost_ticks"] == m.table["distance_ticks"]).all()
    over = cal.as_overrides()
    assert over["maker.sigma"] == cal.sigma and over["maker.k"] == cal.fit.k and over["maker.adverse"] == m.adverse_term
