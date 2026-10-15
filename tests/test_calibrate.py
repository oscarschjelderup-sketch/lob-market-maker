"""The calibration recovers known parameters from synthetic counts, and works end to end on probe sessions."""

from __future__ import annotations

import numpy as np
import pytest

from lobmm.analytics import realised_vol
from lobmm.calibrate import calibrate, fit_intensity
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


def test_realised_vol_of_a_random_walk():
    rng = np.random.default_rng(0)
    steps = rng.normal(0, 0.7, 5000)
    mid_t = np.arange(5000, dtype=float)
    mid = 10000 + np.cumsum(steps)
    assert realised_vol(mid_t, mid, 5000.0, dt=1.0) == pytest.approx(0.7, rel=0.05)
    assert realised_vol(mid_t, mid, 5000.0, dt=4.0) == pytest.approx(0.7, rel=0.15), "a random walk has a flat signature"


def test_calibration_end_to_end_on_short_probe_sessions():
    cfg = SimConfig().with_(**{"market.horizon": 300.0})
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
    over = cal.as_overrides()
    assert over["maker.sigma"] == cal.sigma and over["maker.k"] == cal.fit.k
