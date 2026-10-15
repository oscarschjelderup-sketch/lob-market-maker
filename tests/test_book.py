"""The matching engine: known-value tests, then randomised and property-based tests against a reference book."""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, invariant, rule

from lobmm.book import BUY, SELL, OrderBook


def test_limit_rests_and_sets_the_touch():
    book = OrderBook()
    b = book.submit_limit(BUY, 100, 10, "a", 0.0)
    s = book.submit_limit(SELL, 102, 5, "b", 1.0)
    assert b.fills == [] and b.resting == 10
    assert s.fills == [] and s.resting == 5
    assert book.best_bid == 100 and book.best_ask == 102
    assert book.mid() == 101 and book.spread() == 2
    assert book.depth(BUY) == [(100, 10)] and book.depth(SELL) == [(102, 5)]
    book.check_invariants()


def test_crossing_limit_fills_at_the_resting_price():
    book = OrderBook()
    book.submit_limit(SELL, 101, 10, "maker", 0.0)
    sub = book.submit_limit(BUY, 103, 4, "taker", 1.0)
    assert [(f.price, f.qty, f.maker_owner, f.taker_owner) for f in sub.fills] == [(101, 4, "maker", "taker")]
    assert sub.resting == 0
    assert book.volume_at(SELL, 101) == 6
    assert book.best_bid is None
    book.check_invariants()


def test_partial_fill_rests_the_remainder():
    book = OrderBook()
    book.submit_limit(SELL, 101, 3, "maker", 0.0)
    sub = book.submit_limit(BUY, 101, 10, "taker", 1.0)
    assert sub.filled == 3 and sub.resting == 7
    assert book.best_ask is None and book.best_bid == 101 and book.volume_at(BUY, 101) == 7
    book.check_invariants()


def test_price_priority_then_time_priority():
    book = OrderBook()
    first = book.submit_limit(SELL, 102, 5, "first", 0.0).order_id
    second = book.submit_limit(SELL, 102, 5, "second", 1.0).order_id
    better = book.submit_limit(SELL, 101, 2, "better", 2.0).order_id
    fills = book.submit_market(BUY, 9, "taker", 3.0).fills
    assert [(f.price, f.qty, f.maker_id) for f in fills] == [(101, 2, better), (102, 5, first), (102, 2, second)]
    assert book.queue_ahead(second) == 0 and book.volume_at(SELL, 102) == 3
    book.check_invariants()


def test_queue_ahead_counts_only_live_quantity_before_the_order():
    book = OrderBook()
    a = book.submit_limit(BUY, 100, 5, "a", 0.0).order_id
    b = book.submit_limit(BUY, 100, 7, "b", 1.0).order_id
    c = book.submit_limit(BUY, 100, 1, "c", 2.0).order_id
    assert book.queue_ahead(c) == 12
    book.cancel(a)
    assert book.queue_ahead(c) == 7
    book.submit_market(SELL, 4, "t", 3.0)
    assert book.queue_ahead(c) == 3 and book.queue_ahead(b) == 0


def test_cancel_removes_the_order_and_empties_the_level():
    book = OrderBook()
    oid = book.submit_limit(BUY, 100, 10, "a", 0.0).order_id
    book.submit_limit(BUY, 99, 10, "a", 0.0)
    assert book.cancel(oid) == 10
    assert book.best_bid == 99 and book.volume_at(BUY, 100) == 0
    assert book.cancel(oid) == 0, "a second cancel is a no-op"
    book.check_invariants()


def test_cancel_after_full_fill_is_a_no_op():
    book = OrderBook()
    oid = book.submit_limit(SELL, 101, 5, "a", 0.0).order_id
    book.submit_market(BUY, 5, "t", 1.0)
    assert book.cancel(oid) == 0
    book.check_invariants()


def test_market_order_walks_the_book_and_reports_what_it_could_not_fill():
    book = OrderBook()
    book.submit_limit(SELL, 101, 5, "a", 0.0)
    book.submit_limit(SELL, 103, 5, "b", 0.0)
    sub = book.submit_market(BUY, 12, "t", 1.0)
    assert [(f.price, f.qty) for f in sub.fills] == [(101, 5), (103, 5)]
    assert sub.unfilled == 2 and sub.filled == 10
    assert book.best_ask is None and book.last_trade == 103
    book.check_invariants()


def test_post_only_is_rejected_when_it_would_cross():
    book = OrderBook()
    book.submit_limit(SELL, 101, 5, "a", 0.0)
    sub = book.submit_limit(BUY, 101, 5, "mm", 1.0, post_only=True)
    assert sub.rejected and sub.fills == [] and sub.resting == 0
    assert book.volume_at(SELL, 101) == 5 and book.rejected == 5
    ok = book.submit_limit(BUY, 100, 5, "mm", 1.0, post_only=True)
    assert not ok.rejected and ok.resting == 5
    book.check_invariants()


def test_ioc_takes_what_it_can_and_rests_nothing():
    book = OrderBook()
    book.submit_limit(SELL, 101, 5, "a", 0.0)
    book.submit_limit(SELL, 103, 5, "b", 0.0)
    sub = book.submit_limit(BUY, 102, 20, "informed", 1.0, ioc=True)
    assert [(f.price, f.qty) for f in sub.fills] == [(101, 5)]
    assert sub.resting == 0 and sub.unfilled == 15
    assert book.best_bid is None and book.best_ask == 103
    book.check_invariants()


def test_best_price_survives_a_level_being_emptied_and_recreated():
    book = OrderBook()
    a = book.submit_limit(SELL, 101, 5, "a", 0.0).order_id
    book.submit_limit(SELL, 102, 5, "b", 0.0)
    book.cancel(a)
    assert book.best_ask == 102
    book.submit_limit(SELL, 101, 1, "c", 1.0)
    assert book.best_ask == 101
    book.submit_market(BUY, 1, "t", 2.0)
    assert book.best_ask == 102
    book.check_invariants()


def test_bad_orders_are_refused():
    book = OrderBook()
    with pytest.raises(ValueError):
        book.submit_limit(BUY, 100, 0, "a", 0.0)
    with pytest.raises(ValueError):
        book.submit_limit(BUY, 0, 1, "a", 0.0)
    with pytest.raises(ValueError):
        book.submit_market(0, 1, "a", 0.0)


# ---------------------------------------------------------------- a reference book, deliberately naive
class ReferenceBook:
    """Every order in one list; best prices and matching found by scanning. Slow and obviously right."""

    def __init__(self) -> None:
        self.live: list[dict] = []
        self.next_id = 1
        self.seq = 0

    def best(self, side: int) -> int | None:
        prices = [o["price"] for o in self.live if o["side"] == side]
        if not prices:
            return None
        return max(prices) if side == BUY else min(prices)

    def depth(self, side: int) -> dict[int, int]:
        out: dict[int, int] = {}
        for o in self.live:
            if o["side"] == side:
                out[o["price"]] = out.get(o["price"], 0) + o["remaining"]
        return out

    def _match(self, side: int, limit: int | None, qty: int) -> list[tuple[int, int, int]]:
        fills = []
        while qty > 0:
            candidates = [o for o in self.live if o["side"] == -side]
            if limit is not None:
                candidates = [o for o in candidates if (o["price"] <= limit if side == BUY else o["price"] >= limit)]
            if not candidates:
                break
            candidates.sort(key=lambda o: ((-o["price"] if side == SELL else o["price"]), o["seq"]))
            maker = candidates[0]
            traded = min(qty, maker["remaining"])
            maker["remaining"] -= traded
            qty -= traded
            fills.append((maker["price"], traded, maker["id"]))
            if maker["remaining"] == 0:
                self.live.remove(maker)
        return fills

    def submit_limit(self, side: int, price: int, qty: int, post_only: bool = False):
        oid = self.next_id
        self.next_id += 1
        opp = self.best(-side)
        if post_only and opp is not None and (opp <= price if side == BUY else opp >= price):
            return oid, [], 0, True
        fills = self._match(side, price, qty)
        rem = qty - sum(f[1] for f in fills)
        if rem > 0:
            self.live.append({"id": oid, "side": side, "price": price, "remaining": rem, "seq": self.seq})
            self.seq += 1
        return oid, fills, rem, False

    def submit_market(self, side: int, qty: int):
        self.next_id += 1
        return self._match(side, None, qty)

    def cancel(self, oid: int) -> int:
        for o in self.live:
            if o["id"] == oid:
                self.live.remove(o)
                return o["remaining"]
        return 0


def _same(book: OrderBook, ref: ReferenceBook) -> None:
    book.check_invariants()
    assert book.best(BUY) == ref.best(BUY)
    assert book.best(SELL) == ref.best(SELL)
    for side in (BUY, SELL):
        assert dict(book.depth(side, levels=10_000)) == ref.depth(side)
    assert len(book) == len(ref.live)


def test_randomised_stress_against_the_reference_book():
    rng = np.random.default_rng(7)
    book, ref = OrderBook(), ReferenceBook()
    ids: list[int] = []
    for step in range(20_000):
        action = rng.random()
        side = BUY if rng.random() < 0.5 else SELL
        if action < 0.55:
            price = int(rng.integers(95, 106))
            qty = int(rng.integers(1, 20))
            post_only = bool(rng.random() < 0.2)
            sub = book.submit_limit(side, price, qty, "x", float(step), post_only=post_only)
            oid, fills, rem, rejected = ref.submit_limit(side, price, qty, post_only=post_only)
            assert sub.order_id == oid
            assert sub.rejected == rejected
            assert [(f.price, f.qty, f.maker_id) for f in sub.fills] == fills
            assert sub.resting == rem
            if rem > 0:
                ids.append(oid)
        elif action < 0.75:
            qty = int(rng.integers(1, 40))
            sub = book.submit_market(side, qty, "x", float(step))
            fills = ref.submit_market(side, qty)
            assert [(f.price, f.qty, f.maker_id) for f in sub.fills] == fills
        elif ids:
            oid = ids[int(rng.integers(len(ids)))]
            assert book.cancel(oid) == ref.cancel(oid)
        if step % 97 == 0:
            _same(book, ref)
    _same(book, ref)
    assert book.n_fills > 1000, "the stress test should actually trade"


class BookMachine(RuleBasedStateMachine):
    """Hypothesis drives both books through the same operations and checks they agree after each."""

    def __init__(self) -> None:
        super().__init__()
        self.book = OrderBook()
        self.ref = ReferenceBook()
        self.ids: list[int] = []
        self.t = 0.0

    @rule(side=st.sampled_from([BUY, SELL]), price=st.integers(98, 103), qty=st.integers(1, 12), post_only=st.booleans())
    def limit(self, side, price, qty, post_only):
        self.t += 1
        sub = self.book.submit_limit(side, price, qty, "h", self.t, post_only=post_only)
        oid, fills, rem, rejected = self.ref.submit_limit(side, price, qty, post_only=post_only)
        assert sub.order_id == oid and sub.rejected == rejected and sub.resting == rem
        assert [(f.price, f.qty, f.maker_id) for f in sub.fills] == fills
        if rem > 0:
            self.ids.append(oid)

    @rule(side=st.sampled_from([BUY, SELL]), qty=st.integers(1, 30))
    def market(self, side, qty):
        self.t += 1
        sub = self.book.submit_market(side, qty, "h", self.t)
        assert [(f.price, f.qty, f.maker_id) for f in sub.fills] == self.ref.submit_market(side, qty)

    @rule(data=st.data())
    def cancel(self, data):
        if not self.ids:
            return
        oid = data.draw(st.sampled_from(self.ids))
        assert self.book.cancel(oid) == self.ref.cancel(oid)

    @invariant()
    def books_agree(self):
        _same(self.book, self.ref)


BookMachine.TestCase.settings = settings(
    max_examples=60, stateful_step_count=60, suppress_health_check=list(HealthCheck), deadline=None
)
TestBookMachine = BookMachine.TestCase
