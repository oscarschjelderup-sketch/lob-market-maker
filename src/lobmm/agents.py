"""The market makers. Each one turns a snapshot of the book and its own inventory into a bid and an ask (ticks).

The simulator does the rest: it keeps a resting order per side, cancels and replaces only when the price
changes (so an unchanged quote keeps its queue position), never lets a quote cross the touch, enforces the
inventory limit and the kill switch, and applies the latency.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from lobmm.config import SimConfig


@dataclass(frozen=True, slots=True)
class Snapshot:
    """What the market maker sees: the touch and the mid, in ticks. ``mid`` is the last known mid if a side is empty."""

    t: float
    best_bid: int | None
    best_ask: int | None
    mid: float


class MarketMaker:
    name = "base"

    def quotes(self, snap: Snapshot, inventory: int, rng: np.random.Generator) -> tuple[int | None, int | None]:
        raise NotImplementedError


class NoMaker(MarketMaker):
    """Quotes nothing. Used to measure the background flow on its own, and for benchmarks."""

    name = "none"

    def quotes(self, snap: Snapshot, inventory: int, rng: np.random.Generator) -> tuple[int | None, int | None]:
        return None, None


class NaiveMaker(MarketMaker):
    """A fixed symmetric spread around the mid, whatever the inventory."""

    name = "naive"

    def __init__(self, half_spread: int) -> None:
        if half_spread < 1:
            raise ValueError("half_spread must be at least one tick")
        self.half_spread = half_spread

    def quotes(self, snap: Snapshot, inventory: int, rng: np.random.Generator) -> tuple[int | None, int | None]:
        bid = math.floor(snap.mid - self.half_spread)
        ask = math.ceil(snap.mid + self.half_spread)
        return bid, ask


class AvellanedaStoikovMaker(MarketMaker):
    """Avellaneda and Stoikov (2008): quotes around a reservation price that leans against the inventory.

    With mid ``s``, inventory ``q``, risk aversion ``gamma``, mid volatility ``sigma`` (price per sqrt(second)),
    time to the end of the session ``tau`` and fill-intensity decay ``k`` (per price unit):

        reservation  r = s - q * gamma * sigma^2 * tau
        spread       delta_a + delta_b = gamma * sigma^2 * tau + (2 / gamma) * ln(1 + gamma / k)

    Long inventory pushes both quotes down (more likely to sell, less likely to buy); higher risk aversion or
    volatility widens the spread and strengthens the skew. The spread shrinks towards (2/gamma) ln(1 + gamma/k)
    as the session ends, because there is less time for the inventory to hurt.
    """

    name = "as"

    def __init__(self, gamma: float, sigma: float, k: float, horizon: float, tick: float) -> None:
        if gamma <= 0 or sigma <= 0 or k <= 0:
            raise ValueError("gamma, sigma and k must be positive")
        self.gamma = gamma
        self.sigma = sigma
        self.k = k
        self.horizon = horizon
        self.tick = tick

    def reservation_price(self, mid_price: float, inventory: int, t: float) -> float:
        tau = max(self.horizon - t, 0.0)
        return mid_price - inventory * self.gamma * self.sigma**2 * tau

    def optimal_spread(self, t: float) -> float:
        tau = max(self.horizon - t, 0.0)
        return self.gamma * self.sigma**2 * tau + (2.0 / self.gamma) * math.log(1.0 + self.gamma / self.k)

    def quotes(self, snap: Snapshot, inventory: int, rng: np.random.Generator) -> tuple[int | None, int | None]:
        mid_price = snap.mid * self.tick
        reservation = self.reservation_price(mid_price, inventory, snap.t)
        half = self.optimal_spread(snap.t) / 2.0
        bid = math.floor((reservation - half) / self.tick)
        ask = math.ceil((reservation + half) / self.tick)
        if ask <= bid:
            ask = bid + 1
        return bid, ask


class ProbeMaker(MarketMaker):
    """Quotes at a random distance from the mid on each side, to measure how fill intensity decays with distance."""

    name = "probe"

    def __init__(self, max_distance: int) -> None:
        self.max_distance = max(1, int(max_distance))

    def quotes(self, snap: Snapshot, inventory: int, rng: np.random.Generator) -> tuple[int | None, int | None]:
        d_bid, d_ask = rng.integers(1, self.max_distance + 1, size=2)
        return math.floor(snap.mid - d_bid), math.ceil(snap.mid + d_ask)


def make_maker(cfg: SimConfig) -> MarketMaker:
    m = cfg.maker
    if m.strategy == "none":
        return NoMaker()
    if m.strategy == "naive":
        return NaiveMaker(m.half_spread)
    if m.strategy == "as":
        return AvellanedaStoikovMaker(m.gamma, cfg.sigma, m.k, cfg.market.horizon, cfg.market.tick)
    if m.strategy == "probe":
        return ProbeMaker(m.probe_max_distance)
    raise ValueError(f"unknown strategy {m.strategy!r}: use naive, as, probe or none")
