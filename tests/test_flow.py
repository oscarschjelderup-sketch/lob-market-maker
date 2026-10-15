"""The synthetic flow: sizes and waits have the right means, noise limits never cross, the fair value has the right variance."""

from __future__ import annotations

import numpy as np
import pytest

from lobmm.book import BUY, SELL, OrderBook
from lobmm.config import FlowConfig
from lobmm.flow import FairValue, FlowGenerator


def generator(seed: int = 0, **kw) -> FlowGenerator:
    return FlowGenerator(FlowConfig(**kw), np.random.default_rng(seed), v0=10000)


def test_sizes_and_waits_have_the_configured_means():
    g = generator()
    sizes = np.array([g.size(5.0) for _ in range(20_000)])
    waits = np.array([g.wait(2.0) for _ in range(20_000)])
    lifetimes = np.array([g.lifetime() for _ in range(20_000)])
    assert sizes.min() >= 1 and sizes.mean() == pytest.approx(5.0, rel=0.05)
    assert waits.mean() == pytest.approx(0.5, rel=0.05)
    assert lifetimes.mean() == pytest.approx(FlowConfig().lifetime, rel=0.05)


def test_seeded_book_is_symmetric_around_the_initial_value():
    g = generator(seed_levels=4, seed_size=7)
    book = OrderBook()
    ids = g.seed_book(book)
    assert len(ids) == 8
    assert book.depth(BUY, 10) == [(9999, 7), (9998, 7), (9997, 7), (9996, 7)]
    assert book.depth(SELL, 10) == [(10001, 7), (10002, 7), (10003, 7), (10004, 7)]
    assert book.mid() == 10000.0


def test_noise_limit_orders_never_cross_and_often_improve_the_touch():
    g = generator(depth_mean=3.0)
    book = OrderBook()
    g.seed_book(book)
    improved = 0
    for i in range(2000):
        side = BUY if i % 2 == 0 else SELL
        if i % 7 == 0:
            book.submit_market(side, 25, "noise", float(i))  # widen the spread now and then
        before = book.best(side)
        price = g.limit_price(book, side, book.mid() or 10000.0)
        opposite = book.best(-side)
        if opposite is not None:
            assert price < opposite if side == BUY else price > opposite
        sub = book.submit_limit(side, price, 3, "noise", float(i))
        assert sub.fills == [], "a noise limit order never takes"
        if before is not None and ((price > before) if side == BUY else (price < before)):
            improved += 1
    assert improved > 50
    book.check_invariants()


def test_limit_price_falls_back_when_a_side_is_empty():
    g = generator()
    book = OrderBook()
    book.submit_limit(BUY, 9990, 1, "a", 0.0)
    price = g.limit_price(book, SELL, 10000.0)
    assert price > 9990, "with no asks, a new ask goes above the best bid"
    assert generator().limit_price(OrderBook(), BUY, 10000.0) <= 10000, "with an empty book, around the last mid"


def test_informed_order_sides_and_prices():
    g = generator(informed_noise=0.0)
    book = OrderBook()
    g.seed_book(book)
    assert g.informed_order(book, 10003.6) == (BUY, 10003)
    assert g.informed_order(book, 9997.2) == (SELL, 9998)
    assert g.informed_order(book, 10000.3) is None, "inside the spread there is nothing to do"


def test_fair_value_variance_with_and_without_jumps():
    rng = np.random.default_rng(1)
    diffusion = FairValue(0.0, vol=0.5, rng=rng)
    for t in range(1, 20_001):
        diffusion.advance(float(t))
    steps = np.diff(diffusion.history_v)
    assert steps.var() == pytest.approx(0.25, rel=0.05)
    assert diffusion.n_jumps == 0

    jumpy = FairValue(0.0, vol=0.5, rng=np.random.default_rng(2), jump_rate=1 / 120, jump_std=8.0)
    for t in range(1, 60_001):
        jumpy.advance(float(t))
    steps = np.diff(jumpy.history_v)
    assert steps.var() == pytest.approx(0.25 + 64.0 / 120.0, rel=0.15)
    assert jumpy.n_jumps == pytest.approx(500, rel=0.2)
    assert len(jumpy.history_t) == 60_001
