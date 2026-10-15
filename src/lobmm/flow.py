"""Synthetic order flow.

Three kinds of traders send orders to the book:

- noise traders: limit orders placed a geometric number of ticks away from the opposite best quote (so they
  never cross, but often improve the touch), market orders on a random side, and cancellations after an
  exponential lifetime. This is the zero-intelligence model of Cont, Stoikov and Talreja (2010), simplified.
- a latent fair value: arithmetic Brownian motion in ticks, advanced lazily when someone looks at it.
- informed traders: look at the fair value (with some noise), and send a market order when the book is
  mispriced against it: buy when the fair value is above the ask, sell when it is below the bid. They are what
  pulls the book after the fair value, and what makes the fills of a market maker adversely selected.
"""

from __future__ import annotations

import math

import numpy as np

from lobmm.book import BUY, SELL, OrderBook
from lobmm.config import FlowConfig


class FairValue:
    """A latent fair value in ticks: arithmetic Brownian motion (``vol`` ticks per sqrt(second)) plus news,
    normal jumps of std ``jump_std`` ticks arriving at ``jump_rate`` per second. Advanced lazily."""

    def __init__(self, v0: float, vol: float, rng: np.random.Generator, jump_rate: float = 0.0, jump_std: float = 0.0) -> None:
        self.v = float(v0)
        self.vol = vol
        self.jump_rate = jump_rate
        self.jump_std = jump_std
        self.t = 0.0
        self.rng = rng
        self.history_t: list[float] = [0.0]
        self.history_v: list[float] = [float(v0)]
        self.n_jumps = 0

    def advance(self, t: float) -> float:
        dt = t - self.t
        if dt > 0:
            self.v += self.vol * math.sqrt(dt) * self.rng.standard_normal()
            if self.jump_rate > 0:
                jumps = int(self.rng.poisson(self.jump_rate * dt))
                if jumps:
                    self.v += self.jump_std * math.sqrt(jumps) * self.rng.standard_normal()
                    self.n_jumps += jumps
            self.t = t
            self.history_t.append(t)
            self.history_v.append(self.v)
        return self.v


class FlowGenerator:
    """Draws the sizes, prices and timings of the noise flow. Stateless apart from the generator."""

    def __init__(self, cfg: FlowConfig, rng: np.random.Generator, v0: int) -> None:
        self.cfg = cfg
        self.rng = rng
        self.v0 = int(v0)

    def size(self, mean: float) -> int:
        """A geometric size with the given mean (so always >= 1)."""
        return int(self.rng.geometric(1.0 / max(mean, 1.0)))

    def wait(self, rate: float) -> float:
        """Time to the next Poisson arrival at ``rate`` per second."""
        return float(self.rng.exponential(1.0 / rate))

    def lifetime(self) -> float:
        return float(self.rng.exponential(self.cfg.lifetime))

    def seed_book(self, book: OrderBook, t: float = 0.0) -> list[int]:
        """Fill ``seed_levels`` levels on each side around the initial fair value. Returns the order ids."""
        ids = []
        for level in range(1, self.cfg.seed_levels + 1):
            for side, price in ((BUY, self.v0 - level), (SELL, self.v0 + level)):
                sub = book.submit_limit(side, price, self.cfg.seed_size, "noise", t)
                ids.append(sub.order_id)
        return ids

    def limit_price(self, book: OrderBook, side: int, last_mid: float) -> int:
        """Where a new noise limit order goes: a geometric number of ticks (>= 1) inside the opposite best."""
        reference = book.best(-side)
        if reference is None:
            same = book.best(side)
            if same is not None:
                reference = same + 1 if side == BUY else same - 1
            else:
                reference = int(round(last_mid)) + (1 if side == BUY else -1)
        distance = int(self.rng.geometric(1.0 / max(self.cfg.depth_mean, 1.0)))
        price = reference - distance if side == BUY else reference + distance
        return max(price, 1)

    def market_side(self) -> int:
        return BUY if self.rng.random() < 0.5 else SELL

    def informed_order(self, book: OrderBook, fair: float) -> tuple[int, int] | None:
        """What an informed trader does when the fair value is ``fair`` (ticks): (side, limit price) or nothing.

        The trader buys everything offered below its estimate of the fair value and sells everything bid above
        it, as an immediate-or-cancel limit order at the estimate, so the book is pulled towards the fair value.
        """
        estimate = fair + self.rng.normal(0.0, self.cfg.informed_noise) if self.cfg.informed_noise > 0 else fair
        ask, bid = book.best(SELL), book.best(BUY)
        if ask is not None and estimate > ask:
            return BUY, int(math.floor(estimate))
        if bid is not None and estimate < bid:
            return SELL, int(math.ceil(estimate))
        return None
