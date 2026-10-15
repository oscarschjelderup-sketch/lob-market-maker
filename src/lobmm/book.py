"""The matching engine: a limit order book with price-time priority.

Prices are integer ticks and quantities integers, so there is never a floating-point comparison in the
matching path. Each side keeps one FIFO queue per price level; the best price is found through a heap of
level prices with lazy deletion; cancels are O(1) through a map from order id to order (the order is
flagged dead and skipped when its queue is matched, so nothing is removed from the middle of a deque).

Every submission reports its fills. The book also counts submitted, filled, cancelled, rejected and unfilled
volume, so a test can check that volume is conserved after any sequence of events.
"""

from __future__ import annotations

import heapq
from collections import deque
from dataclasses import dataclass, field

BUY = 1
SELL = -1


def side_name(side: int) -> str:
    return "buy" if side == BUY else "sell"


@dataclass(slots=True)
class Order:
    id: int
    side: int
    price: int
    qty: int
    remaining: int
    ts: float
    owner: str
    live: bool = True


@dataclass(slots=True, frozen=True)
class Fill:
    """One trade. ``taker_side`` is the side of the aggressor; the maker is on the other side."""

    ts: float
    price: int
    qty: int
    taker_side: int
    taker_id: int
    taker_owner: str
    maker_id: int
    maker_owner: str


@dataclass(slots=True)
class Submission:
    order_id: int
    fills: list[Fill] = field(default_factory=list)
    resting: int = 0  # quantity left in the book (limit orders)
    unfilled: int = 0  # quantity the book could not absorb (market orders)
    rejected: bool = False  # a post-only limit order that would have crossed

    @property
    def filled(self) -> int:
        return sum(f.qty for f in self.fills)


class OrderBook:
    """A price-time-priority limit order book on an integer tick grid."""

    def __init__(self) -> None:
        self._queues: dict[int, dict[int, deque[Order]]] = {BUY: {}, SELL: {}}
        self._depth: dict[int, dict[int, int]] = {BUY: {}, SELL: {}}
        self._heaps: dict[int, list[int]] = {BUY: [], SELL: []}
        self.orders: dict[int, Order] = {}
        self._next_id = 1
        self.last_trade: int | None = None
        self.n_fills = 0
        # volume counters: submitted == live + 2 * filled + cancelled + rejected + unfilled, always
        self.submitted = 0
        self.filled = 0
        self.cancelled = 0
        self.rejected = 0
        self.unfilled = 0

    # ------------------------------------------------------------------ queries
    def best(self, side: int) -> int | None:
        """Best price on ``side`` (highest bid, lowest ask) or None if the side is empty."""
        heap = self._heaps[side]
        depth = self._depth[side]
        while heap:
            price = -heap[0] if side == BUY else heap[0]
            if price in depth:
                return price
            heapq.heappop(heap)
        return None

    @property
    def best_bid(self) -> int | None:
        return self.best(BUY)

    @property
    def best_ask(self) -> int | None:
        return self.best(SELL)

    def mid(self) -> float | None:
        bid, ask = self.best(BUY), self.best(SELL)
        if bid is None or ask is None:
            return None
        return (bid + ask) / 2

    def spread(self) -> int | None:
        bid, ask = self.best(BUY), self.best(SELL)
        if bid is None or ask is None:
            return None
        return ask - bid

    def volume_at(self, side: int, price: int) -> int:
        return self._depth[side].get(price, 0)

    def depth(self, side: int, levels: int = 5) -> list[tuple[int, int]]:
        """The best ``levels`` price levels of ``side`` as (price, live quantity), best first."""
        prices = sorted(self._depth[side], reverse=(side == BUY))[:levels]
        return [(p, self._depth[side][p]) for p in prices]

    def live_volume(self, side: int | None = None) -> int:
        if side is None:
            return self.live_volume(BUY) + self.live_volume(SELL)
        return sum(self._depth[side].values())

    def queue_ahead(self, order_id: int) -> int:
        """Live quantity queued before ``order_id`` at its price level (its queue position)."""
        order = self.orders[order_id]
        ahead = 0
        for o in self._queues[order.side][order.price]:
            if o.id == order_id:
                return ahead
            if o.live:
                ahead += o.remaining
        raise KeyError(order_id)

    def __len__(self) -> int:
        return len(self.orders)

    # ------------------------------------------------------------------ orders
    def submit_limit(
        self, side: int, price: int, qty: int, owner: str, ts: float, post_only: bool = False, ioc: bool = False
    ) -> Submission:
        """A limit order: matched against the other side while it crosses, the rest rests in the book.

        ``post_only`` rejects the order instead of letting it take liquidity; ``ioc`` (immediate-or-cancel)
        drops whatever could not be matched at once instead of resting it.
        """
        if side not in (BUY, SELL) or qty <= 0 or price <= 0:
            raise ValueError(f"bad order: side={side} price={price} qty={qty}")
        order_id = self._next_id
        self._next_id += 1
        self.submitted += qty
        sub = Submission(order_id)
        if post_only:
            opposite = self.best(-side)
            if opposite is not None and (opposite <= price if side == BUY else opposite >= price):
                self.rejected += qty
                sub.rejected = True
                return sub
        remaining = self._match(side, price, qty, order_id, owner, ts, sub.fills)
        if remaining > 0:
            if ioc:
                self.unfilled += remaining
                sub.unfilled = remaining
            else:
                self._rest(Order(order_id, side, price, qty, remaining, ts, owner))
                sub.resting = remaining
        return sub

    def submit_market(self, side: int, qty: int, owner: str, ts: float) -> Submission:
        """A market order: takes liquidity until filled or the other side is empty."""
        if side not in (BUY, SELL) or qty <= 0:
            raise ValueError(f"bad order: side={side} qty={qty}")
        order_id = self._next_id
        self._next_id += 1
        self.submitted += qty
        sub = Submission(order_id)
        remaining = self._match(side, None, qty, order_id, owner, ts, sub.fills)
        if remaining > 0:
            self.unfilled += remaining
            sub.unfilled = remaining
        return sub

    def cancel(self, order_id: int) -> int:
        """Cancel a resting order. Returns the quantity removed (0 if the order was gone already)."""
        order = self.orders.pop(order_id, None)
        if order is None:
            return 0
        order.live = False
        removed = order.remaining
        order.remaining = 0
        self.cancelled += removed
        side, price = order.side, order.price
        self._depth[side][price] -= removed
        if self._depth[side][price] == 0:
            self._drop_level(side, price)
        return removed

    # ------------------------------------------------------------------ internals
    def _match(self, side: int, limit: int | None, qty: int, taker_id: int, owner: str, ts: float, fills: list[Fill]) -> int:
        opposite = -side
        queues = self._queues[opposite]
        depth = self._depth[opposite]
        while qty > 0:
            best = self.best(opposite)
            if best is None:
                break
            if limit is not None and (best > limit if side == BUY else best < limit):
                break
            queue = queues[best]
            while qty > 0 and queue:
                maker = queue[0]
                if not maker.live:
                    queue.popleft()
                    continue
                traded = min(qty, maker.remaining)
                maker.remaining -= traded
                qty -= traded
                depth[best] -= traded
                self.filled += traded
                self.n_fills += 1
                self.last_trade = best
                fills.append(Fill(ts, best, traded, side, taker_id, owner, maker.id, maker.owner))
                if maker.remaining == 0:
                    maker.live = False
                    queue.popleft()
                    del self.orders[maker.id]
            if depth[best] == 0:
                self._drop_level(opposite, best)
        return qty

    def _rest(self, order: Order) -> None:
        side, price = order.side, order.price
        queue = self._queues[side].get(price)
        if queue is None:
            queue = deque()
            self._queues[side][price] = queue
            self._depth[side][price] = 0
            heapq.heappush(self._heaps[side], -price if side == BUY else price)
        queue.append(order)
        self._depth[side][price] += order.remaining
        self.orders[order.id] = order

    def _drop_level(self, side: int, price: int) -> None:
        del self._queues[side][price]
        del self._depth[side][price]

    # ------------------------------------------------------------------ invariants
    def check_invariants(self) -> None:
        """Raise AssertionError if the book is in an impossible state. The tests call this after every event."""
        bid, ask = self.best(BUY), self.best(SELL)
        assert bid is None or ask is None or bid < ask, f"crossed book: bid {bid} >= ask {ask}"
        live = 0
        seen: set[int] = set()
        for side in (BUY, SELL):
            for price, queue in self._queues[side].items():
                total = 0
                for o in queue:
                    if o.live:
                        assert o.remaining > 0, f"live order {o.id} with remaining {o.remaining}"
                        assert o.side == side and o.price == price, f"order {o.id} queued at the wrong level"
                        assert self.orders.get(o.id) is o, f"live order {o.id} missing from the id map"
                        total += o.remaining
                        seen.add(o.id)
                assert total == self._depth[side][price] > 0, f"level {price} depth {self._depth[side][price]} != {total}"
                live += total
        assert seen == set(self.orders), "the id map and the queues disagree about which orders are live"
        conserved = live + 2 * self.filled + self.cancelled + self.rejected + self.unfilled
        assert conserved == self.submitted, f"volume not conserved: {conserved} != {self.submitted}"
