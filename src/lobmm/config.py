"""Every parameter of a simulation, with its default and its unit.

Times are seconds, prices are in price units (the tick is ``MarketConfig.tick``), quantities are shares.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace
from typing import Any


@dataclass(frozen=True)
class MarketConfig:
    tick: float = 0.01  # price units per tick
    initial_price: float = 100.0  # fair value at t = 0
    horizon: float = 3600.0  # length of a session, seconds


@dataclass(frozen=True)
class FlowConfig:
    """The background order flow: noise traders, a latent fair value and informed traders."""

    limit_rate: float = 2.0  # noise limit orders per second, per side
    market_rate: float = 0.5  # noise market orders per second, per side
    informed_rate: float = 1.0  # informed looks at the book per second (both sides together)
    limit_size: float = 5.0  # mean size of a noise limit order (geometric)
    market_size: float = 5.0  # mean size of a noise market order (geometric)
    informed_size: float = 15.0  # mean size of an informed order (geometric); it sweeps the book up to the estimate
    depth_mean: float = 4.0  # mean distance, in ticks, of a new limit order from the opposite best (geometric, >= 1)
    lifetime: float = 10.0  # mean seconds a noise limit order rests before it is cancelled (exponential)
    fair_vol: float = 0.5  # fair value diffusion, ticks per sqrt(second)
    jump_rate: float = 1.0 / 120.0  # news: jumps in the fair value per second (Poisson); 0 turns them off
    jump_std: float = 8.0  # std of a jump, ticks (normal)
    informed_noise: float = 1.0  # std, in ticks, of the error in the informed estimate of the fair value
    seed_levels: int = 10  # levels per side seeded at t = 0
    seed_size: int = 10  # shares per seeded level


@dataclass(frozen=True)
class MakerConfig:
    """The market maker. ``strategy`` is one of ``naive``, ``as`` (Avellaneda-Stoikov), ``probe`` or ``none``."""

    strategy: str = "as"
    size: int = 5  # shares per quote
    wake_interval: float = 0.5  # seconds between looks at the book
    max_inventory: int = 50  # stop quoting the side that would push |inventory| past this
    half_spread: int = 3  # naive: ticks each side of the mid
    gamma: float = 0.003  # Avellaneda-Stoikov risk aversion (the gamma sweep in the README picks this)
    sigma: float | None = 0.0136  # A-S mid volatility, price units per sqrt(second); None = the fair_vol of the flow * tick
    k: float = 40.0  # A-S fill-intensity decay, per price unit
    # sigma and k above are what ``lobmm calibrate`` finds on the default flow (1.36 ticks/sqrt(s) at 10 s, k = 40).
    # Change the flow and they are stale: run ``lobmm calibrate`` again, or set sigma to None for the fair-value volatility.
    maker_fee_bps: float = 0.0  # fee on each passive fill, basis points of notional; negative = rebate
    taker_fee_bps: float = 0.0  # fee on each aggressive fill (the maker only takes when it flattens)
    latency: float = 0.0  # one-way delay, seconds, on both legs: book -> maker and maker -> exchange
    kill_drawdown: float | None = None  # price units of equity drawdown that stops quoting for the session
    probe_max_distance: int = 8  # probe: quotes are placed 1..this many ticks from the mid, at random


@dataclass(frozen=True)
class SimConfig:
    market: MarketConfig = field(default_factory=MarketConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    maker: MakerConfig = field(default_factory=MakerConfig)
    horizons: tuple[float, ...] = (1.0, 10.0, 60.0)  # markout horizons, seconds
    sample_interval: float = 0.0  # > 0: record the depth of the book this often, for the heatmap
    sample_levels: int = 15  # ticks each side of the mid in a depth sample

    def with_(self, **overrides: Any) -> SimConfig:
        """A copy with dotted overrides: ``cfg.with_(**{"maker.gamma": 0.3, "flow.informed_rate": 0})``."""
        groups: dict[str, dict[str, Any]] = {}
        top: dict[str, Any] = {}
        for key, value in overrides.items():
            if "." in key:
                group, name = key.split(".", 1)
                groups.setdefault(group, {})[name] = value
            else:
                top[key] = value
        out = replace(self, **top) if top else self
        for group, values in groups.items():
            sub = getattr(out, group)
            valid = {f.name for f in fields(sub)}
            unknown = set(values) - valid
            if unknown:
                raise KeyError(f"unknown {group} parameter(s): {sorted(unknown)}")
            out = replace(out, **{group: replace(sub, **values)})
        return out

    @property
    def sigma(self) -> float:
        """The A-S volatility in price units per sqrt(second): configured, or the fair-value volatility of the flow."""
        if self.maker.sigma is not None:
            return self.maker.sigma
        return self.flow.fair_vol * self.market.tick
