"""The quoting rules: known values for Avellaneda-Stoikov, the direction of the skew, the naive and probe makers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from lobmm.agents import AvellanedaStoikovMaker, NaiveMaker, NoMaker, ProbeMaker, Snapshot, make_maker
from lobmm.config import SimConfig

RNG = np.random.default_rng(0)


def as_maker(gamma=0.1, sigma=0.02, k=40.0, horizon=3600.0, tick=0.01) -> AvellanedaStoikovMaker:
    return AvellanedaStoikovMaker(gamma, sigma, k, horizon, tick)


def test_reservation_price_by_hand():
    m = as_maker()
    # 100 - q * gamma * sigma^2 * tau = 100 - 10 * 0.1 * 0.0004 * 3600 = 100 - 1.44
    assert m.reservation_price(100.0, 10, 0.0) == pytest.approx(98.56)
    assert m.reservation_price(100.0, -10, 0.0) == pytest.approx(101.44)
    assert m.reservation_price(100.0, 10, 3600.0) == pytest.approx(100.0), "no time left, no inventory penalty"


def test_optimal_spread_by_hand():
    m = as_maker()
    # gamma sigma^2 tau + (2/gamma) ln(1 + gamma/k) = 0.144 + 20 ln(1.0025)
    assert m.optimal_spread(0.0) == pytest.approx(0.144 + 20 * math.log(1.0025), abs=1e-9)
    assert m.optimal_spread(3600.0) == pytest.approx(20 * math.log(1.0025), abs=1e-9)
    assert m.optimal_spread(0.0) > m.optimal_spread(1800.0) > m.optimal_spread(3600.0)


def test_quotes_are_symmetric_with_no_inventory_and_skew_with_it():
    m = as_maker()
    snap = Snapshot(0.0, 9999, 10001, 10000.0)
    bid0, ask0 = m.quotes(snap, 0, RNG)
    assert bid0 < 10000 < ask0
    assert 10000 - bid0 == ask0 - 10000
    bid_long, ask_long = m.quotes(snap, 10, RNG)
    bid_short, ask_short = m.quotes(snap, -10, RNG)
    assert bid_long < bid0 and ask_long < ask0, "long inventory: both quotes move down to sell"
    assert bid_short > bid0 and ask_short > ask0, "short inventory: both quotes move up to buy"
    assert ask_long > bid_long and ask_short > bid_short


def test_spread_widens_with_risk_aversion_and_volatility():
    snap = Snapshot(0.0, 9999, 10001, 10000.0)
    base = as_maker(gamma=0.1)
    more_averse = as_maker(gamma=0.5)
    more_volatile = as_maker(sigma=0.04)
    for m in (more_averse, more_volatile):
        b0, a0 = base.quotes(snap, 0, RNG)
        b1, a1 = m.quotes(snap, 0, RNG)
        assert a1 - b1 > a0 - b0


def test_as_quotes_never_cross_even_for_tiny_spreads():
    m = AvellanedaStoikovMaker(gamma=1e-6, sigma=1e-6, k=1e6, horizon=1.0, tick=0.01)
    bid, ask = m.quotes(Snapshot(0.0, 10000, 10001, 10000.5), 0, RNG)
    assert ask > bid


def test_as_rejects_non_positive_parameters():
    with pytest.raises(ValueError):
        as_maker(gamma=0.0)
    with pytest.raises(ValueError):
        as_maker(k=-1.0)


def test_naive_maker_is_symmetric_around_the_mid():
    m = NaiveMaker(3)
    assert m.quotes(Snapshot(0.0, 10000, 10001, 10000.5), 25, RNG) == (9997, 10004)
    assert m.quotes(Snapshot(0.0, 9999, 10001, 10000.0), -25, RNG) == (9997, 10003)
    with pytest.raises(ValueError):
        NaiveMaker(0)


def test_probe_maker_quotes_within_its_range():
    m = ProbeMaker(8)
    snap = Snapshot(0.0, 10000, 10001, 10000.5)
    for _ in range(200):
        bid, ask = m.quotes(snap, 0, RNG)
        assert 0.5 <= snap.mid - bid <= 8.5 and 0.5 <= ask - snap.mid <= 8.5


def test_make_maker_factory():
    assert isinstance(make_maker(SimConfig().with_(**{"maker.strategy": "none"})), NoMaker)
    assert isinstance(make_maker(SimConfig().with_(**{"maker.strategy": "naive"})), NaiveMaker)
    assert isinstance(make_maker(SimConfig().with_(**{"maker.strategy": "probe"})), ProbeMaker)
    m = make_maker(SimConfig())
    assert isinstance(m, AvellanedaStoikovMaker)
    assert m.sigma == SimConfig().maker.sigma and m.k == SimConfig().maker.k
    fallback = make_maker(SimConfig().with_(**{"maker.sigma": None}))
    assert fallback.sigma == pytest.approx(SimConfig().flow.fair_vol * SimConfig().market.tick), "sigma=None: the fair-value volatility"
    with pytest.raises(ValueError):
        make_maker(SimConfig().with_(**{"maker.strategy": "hft"}))


def test_config_overrides_are_checked():
    cfg = SimConfig().with_(**{"maker.gamma": 0.3, "flow.informed_rate": 0.0, "sample_interval": 2.0})
    assert cfg.maker.gamma == 0.3 and cfg.flow.informed_rate == 0.0 and cfg.sample_interval == 2.0
    assert SimConfig().maker.gamma == 0.003, "with_ returns a copy"
    with pytest.raises(KeyError):
        SimConfig().with_(**{"maker.gama": 0.3})
    assert SimConfig().with_(**{"maker.sigma": 0.5}).sigma == 0.5
