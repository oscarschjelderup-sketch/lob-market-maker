"""One session: the event loop that drives the book, the flow and the market maker.

Order of play inside one event: the handler reads the mid the market maker would have seen (the one recorded
after the previous event), submits or cancels, and books any fills of the market maker against that mid; only
then is the touch re-read and a new mid recorded. So a fill is always marked against the mid *before* the
order that caused it, and the markouts measure what happened next.
"""

from __future__ import annotations

import math
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lobmm.accounting import Ledger
from lobmm.agents import MarketMaker, Snapshot, make_maker
from lobmm.book import BUY, SELL, Fill, OrderBook
from lobmm.config import SimConfig
from lobmm.events import EventQueue, Kind
from lobmm.flow import FairValue, FlowGenerator

MM = "mm"


def distance_bucket(distance: float) -> float:
    """Quote distances are multiples of half a tick; bucket them exactly."""
    return round(distance * 2.0) / 2.0


@dataclass
class SimResult:
    cfg: SimConfig
    seed: int
    strategy: str
    mid_t: np.ndarray  # times at which the mid changed (starts at 0)
    mid: np.ndarray  # the mid after each change, ticks
    fair_t: np.ndarray
    fair: np.ndarray  # the latent fair value, ticks, whenever an informed trader looked at it
    series: pd.DataFrame  # one row per wake of the market maker
    fills: pd.DataFrame  # the fills of the market maker
    ledger: Ledger
    samples: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]]  # (t, mid, prices, bid qty, ask qty)
    exposure: dict[float, float]  # seconds a quote stood at each distance bucket (ticks)
    fills_by_distance: dict[float, int]  # fills of the maker at each distance bucket
    counts: dict[str, int]
    n_events: int
    runtime: float
    killed_at: float | None
    end_mid: float
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def end_time(self) -> float:
        return self.cfg.market.horizon

    def attribution(self) -> dict[str, float]:
        return self.ledger.attribution(self.mid_t, self.mid, self.end_mid)

    def markouts(self) -> pd.DataFrame:
        return self.ledger.markouts(self.mid_t, self.mid, self.cfg.horizons, self.end_time)

    def summary(self) -> dict[str, float]:
        from lobmm.analytics import summarize

        return summarize(self)


class Simulator:
    def __init__(self, cfg: SimConfig, seed: int = 0, maker: MarketMaker | None = None) -> None:
        self.cfg = cfg
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.book = OrderBook()
        self.queue = EventQueue()
        self.tick = cfg.market.tick
        v0 = int(round(cfg.market.initial_price / self.tick))
        self.fair = FairValue(v0, cfg.flow.fair_vol, self.rng, cfg.flow.jump_rate, cfg.flow.jump_std)
        self.flow = FlowGenerator(cfg.flow, self.rng, v0)
        self.maker = maker if maker is not None else make_maker(cfg)
        self.ledger = Ledger(self.tick, cfg.maker.maker_fee_bps, cfg.maker.taker_fee_bps)
        self._mm_orders: dict[int, int | None] = {BUY: None, SELL: None}
        self._mm_distance: dict[int, float] = {}
        self._mid = float(v0)
        self._touch: tuple[int | None, int | None] = (None, None)
        self.mid_t: list[float] = [0.0]
        self.mid: list[float] = [float(v0)]
        self._rows: list[tuple] = []
        self.samples: list[tuple[float, float, np.ndarray, np.ndarray, np.ndarray]] = []
        self.exposure: dict[float, float] = defaultdict(float)
        self.fills_by_distance: dict[float, int] = defaultdict(int)
        self.counts: dict[str, int] = defaultdict(int)
        self.killed_at: float | None = None
        self._peak_equity = 0.0
        self.n_events = 0
        self._handlers = {
            Kind.LIMIT: self._on_limit,
            Kind.MARKET: self._on_market,
            Kind.INFORMED: self._on_informed,
            Kind.CANCEL: self._on_cancel,
            Kind.MM_WAKE: self._on_wake,
            Kind.MM_DECIDE: self._on_decide,
            Kind.MM_ORDERS: self._on_orders,
            Kind.SAMPLE: self._on_sample,
        }

    # ------------------------------------------------------------------ the loop
    def run(self) -> SimResult:
        started = time.perf_counter()
        cfg, flow, q = self.cfg, self.flow, self.queue
        horizon = cfg.market.horizon
        for oid in flow.seed_book(self.book):
            q.push(flow.lifetime(), Kind.CANCEL, oid)
        self._record_mid(0.0)
        for side in (BUY, SELL):
            q.push(flow.wait(cfg.flow.limit_rate), Kind.LIMIT, side)
            if cfg.flow.market_rate > 0:
                q.push(flow.wait(cfg.flow.market_rate), Kind.MARKET, side)
        if cfg.flow.informed_rate > 0:
            q.push(flow.wait(cfg.flow.informed_rate), Kind.INFORMED)
        if cfg.maker.strategy != "none":
            q.push(0.0, Kind.MM_WAKE)
        if cfg.sample_interval > 0:
            q.push(0.0, Kind.SAMPLE)
        handlers = self._handlers
        ledger = self.ledger
        while q:
            t, kind, payload = q.pop()
            if t > horizon:
                break
            fills_before = ledger.n_fills
            handlers[kind](t, payload)
            self.n_events += 1
            self._record_mid(t)
            if ledger.n_fills > fills_before:
                ledger.set_mid_after(fills_before, self._mid)  # the mid once the order that filled the maker is done
        self.runtime = time.perf_counter() - started
        self.counts["jumps"] = self.fair.n_jumps
        return self._result()

    # ------------------------------------------------------------------ noise and informed flow
    def _on_limit(self, t: float, side: int) -> None:
        price = self.flow.limit_price(self.book, side, self._mid)
        sub = self.book.submit_limit(side, price, self.flow.size(self.cfg.flow.limit_size), "noise", t)
        self._handle_fills(t, sub.fills)
        if sub.resting > 0:
            self.queue.push(t + self.flow.lifetime(), Kind.CANCEL, sub.order_id)
        self.counts["limit"] += 1
        self.queue.push(t + self.flow.wait(self.cfg.flow.limit_rate), Kind.LIMIT, side)

    def _on_market(self, t: float, side: int) -> None:
        sub = self.book.submit_market(side, self.flow.size(self.cfg.flow.market_size), "noise", t)
        self._handle_fills(t, sub.fills)
        self.counts["market"] += 1
        self.queue.push(t + self.flow.wait(self.cfg.flow.market_rate), Kind.MARKET, side)

    def _on_informed(self, t: float, _: Any) -> None:
        fair = self.fair.advance(t)
        order = self.flow.informed_order(self.book, fair)
        self.counts["informed_looks"] += 1
        if order is not None:
            side, price = order
            sub = self.book.submit_limit(side, price, self.flow.size(self.cfg.flow.informed_size), "informed", t, ioc=True)
            self._handle_fills(t, sub.fills)
            self.counts["informed_trades"] += 1
        self.queue.push(t + self.flow.wait(self.cfg.flow.informed_rate), Kind.INFORMED)

    def _on_cancel(self, t: float, order_id: int) -> None:
        if self.book.cancel(order_id) > 0:
            self.counts["cancel"] += 1

    def _on_sample(self, t: float, _: Any) -> None:
        centre = int(round(self._mid))
        levels = self.cfg.sample_levels
        prices = np.arange(centre - levels, centre + levels + 1)
        bids = np.array([self.book.volume_at(BUY, int(p)) for p in prices])
        asks = np.array([self.book.volume_at(SELL, int(p)) for p in prices])
        self.samples.append((t, self._mid, prices, bids, asks))
        self.queue.push(t + self.cfg.sample_interval, Kind.SAMPLE)

    # ------------------------------------------------------------------ the market maker
    def _on_wake(self, t: float, _: Any) -> None:
        m = self.cfg.maker
        if self.killed_at is None and m.kill_drawdown is not None:
            equity = self.ledger.equity(self._mid)
            self._peak_equity = max(self._peak_equity, equity)
            if self._peak_equity - equity > m.kill_drawdown:
                self.killed_at = t
                for side in (BUY, SELL):
                    self._cancel_mm(side)
        if self.killed_at is None:
            snap = Snapshot(t, self.book.best(BUY), self.book.best(SELL), self._mid)
            if m.latency > 0:
                self.queue.push(t + m.latency, Kind.MM_DECIDE, snap)
            else:
                self._on_decide(t, snap)
        for side in (BUY, SELL):
            oid = self._mm_orders[side]
            if oid is not None and oid in self.book.orders:
                self.exposure[distance_bucket(self._mm_distance[oid])] += m.wake_interval
        self._record_row(t)
        if self.killed_at is None:
            self.queue.push(t + m.wake_interval, Kind.MM_WAKE)

    def _on_decide(self, t: float, snap: Snapshot) -> None:
        bid, ask = self.maker.quotes(snap, self.ledger.inventory, self.rng)
        inv, limit = self.ledger.inventory, self.cfg.maker.max_inventory
        if inv >= limit:
            bid = None
        if inv <= -limit:
            ask = None
        if self.cfg.maker.latency > 0:
            self.queue.push(t + self.cfg.maker.latency, Kind.MM_ORDERS, (bid, ask))
        else:
            self._on_orders(t, (bid, ask))

    def _on_orders(self, t: float, wanted: tuple[int | None, int | None]) -> None:
        if self.killed_at is not None:
            return
        book, size = self.book, self.cfg.maker.size
        for side, want in ((BUY, wanted[0]), (SELL, wanted[1])):
            if want is not None:
                # post-only: never cross the touch as it is when the order reaches the exchange
                opposite = book.best(-side)
                if opposite is not None and (want >= opposite if side == BUY else want <= opposite):
                    want = opposite - 1 if side == BUY else opposite + 1
                if want <= 0:
                    want = None
            current = self._mm_orders[side]
            if current is not None:
                order = book.orders.get(current)
                if order is None:
                    self._mm_orders[side] = None
                    self._mm_distance.pop(current, None)
                elif want is None or order.price != want:
                    self._cancel_mm(side)
                else:
                    continue  # same price: keep the queue position
            if want is None:
                continue
            sub = book.submit_limit(side, want, size, MM, t, post_only=True)
            if sub.resting > 0:
                self._mm_orders[side] = sub.order_id
                self._mm_distance[sub.order_id] = abs(want - self._mid)
                self.counts["mm_orders"] += 1

    def _cancel_mm(self, side: int) -> None:
        oid = self._mm_orders[side]
        if oid is not None:
            self.book.cancel(oid)
            self._mm_distance.pop(oid, None)
            self._mm_orders[side] = None

    def _handle_fills(self, t: float, fills: list[Fill]) -> None:
        for f in fills:
            if f.maker_owner != MM:
                continue
            side = -f.taker_side
            distance = self._mm_distance.get(f.maker_id, math.nan)
            self.ledger.on_fill(t, side, f.price, f.qty, self._mid, distance, maker=True)
            if not math.isnan(distance):
                self.fills_by_distance[distance_bucket(distance)] += 1
            if f.maker_id not in self.book.orders:
                if self._mm_orders[side] == f.maker_id:
                    self._mm_orders[side] = None
                self._mm_distance.pop(f.maker_id, None)
            inv, limit = self.ledger.inventory, self.cfg.maker.max_inventory
            if inv >= limit:
                self._cancel_mm(BUY)
            if inv <= -limit:
                self._cancel_mm(SELL)

    # ------------------------------------------------------------------ recording
    def _record_mid(self, t: float) -> None:
        bid, ask = self.book.best(BUY), self.book.best(SELL)
        if (bid, ask) != self._touch:
            self._touch = (bid, ask)
            if bid is not None and ask is not None:
                mid = (bid + ask) / 2.0
                if mid != self._mid:
                    self._mid = mid
                    self.mid_t.append(t)
                    self.mid.append(mid)

    def _quote(self, side: int) -> float:
        oid = self._mm_orders[side]
        if oid is None:
            return math.nan
        order = self.book.orders.get(oid)
        return math.nan if order is None else float(order.price)

    def _record_row(self, t: float) -> None:
        bid, ask = self.book.best(BUY), self.book.best(SELL)  # as it is now, including the quotes just placed
        self._rows.append(
            (
                t,
                self.ledger.inventory,
                self.ledger.cash,
                self.ledger.equity(self._mid),
                self._quote(BUY),
                self._quote(SELL),
                math.nan if bid is None else bid,
                math.nan if ask is None else ask,
                self._mid,
            )
        )

    def _result(self) -> SimResult:
        series = pd.DataFrame(
            self._rows, columns=["t", "inventory", "cash", "equity", "bid", "ask", "best_bid", "best_ask", "mid"]
        )
        return SimResult(
            cfg=self.cfg,
            seed=self.seed,
            strategy=self.maker.name,
            mid_t=np.asarray(self.mid_t),
            mid=np.asarray(self.mid),
            fair_t=np.asarray(self.fair.history_t),
            fair=np.asarray(self.fair.history_v),
            series=series,
            fills=self.ledger.fills(),
            ledger=self.ledger,
            samples=self.samples,
            exposure=dict(self.exposure),
            fills_by_distance=dict(self.fills_by_distance),
            counts=dict(self.counts),
            n_events=self.n_events,
            runtime=self.runtime,
            killed_at=self.killed_at,
            end_mid=self._mid,
        )


def run_session(cfg: SimConfig | None = None, seed: int = 0, **overrides: Any) -> SimResult:
    """Run one session. ``overrides`` are dotted parameters, e.g. ``run_session(seed=3, **{"maker.gamma": 0.3})``."""
    cfg = (cfg or SimConfig()).with_(**overrides) if overrides else (cfg or SimConfig())
    return Simulator(cfg, seed).run()
