"""Cash, inventory, mark-to-mid equity, markouts and the attribution of PnL.

Every fill of the market maker is recorded with the mid at the moment it happened. That is enough for

- spread capture:   sum over fills of  side * (mid - price) * qty      what the maker earned by being passive
- inventory PnL:    sum over mid moves of  inventory * d(mid)          what the position gained or lost
- fees
- markouts:         side * (mid(t + h) - mid(t)) per share, at horizons h: where the mid went after each fill.
                    A negative average markout is adverse selection: the maker was filled right before the
                    price moved against it.

Mark-to-mid equity equals spread capture + inventory PnL - fees exactly (``test_accounting`` checks it).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from lobmm.book import BUY


class Ledger:
    def __init__(self, tick: float, maker_fee_bps: float = 0.0, taker_fee_bps: float = 0.0) -> None:
        self.tick = tick
        self.maker_fee_bps = maker_fee_bps
        self.taker_fee_bps = taker_fee_bps
        self.cash = 0.0
        self.inventory = 0
        self.fees = 0.0
        self._ts: list[float] = []
        self._side: list[int] = []
        self._price: list[int] = []
        self._qty: list[int] = []
        self._mid: list[float] = []
        self._mid_after: list[float] = []
        self._distance: list[float] = []

    # ------------------------------------------------------------------ recording
    def on_fill(self, ts: float, side: int, price: int, qty: int, mid: float, distance: float = math.nan, maker: bool = True) -> None:
        """Record a fill of ``qty`` at ``price`` ticks on ``side`` (BUY or SELL) while the mid was ``mid`` ticks."""
        notional = price * qty * self.tick
        self.cash -= side * notional
        self.inventory += side * qty
        fee = notional * (self.maker_fee_bps if maker else self.taker_fee_bps) / 1e4
        self.fees += fee
        self.cash -= fee
        self._ts.append(ts)
        self._side.append(side)
        self._price.append(price)
        self._qty.append(qty)
        self._mid.append(mid)
        self._mid_after.append(mid)
        self._distance.append(distance)

    def set_mid_after(self, start: int, mid: float) -> None:
        """The mid once the order that produced fills ``start``.. is done: separates mechanical impact from drift."""
        for i in range(start, len(self._mid_after)):
            self._mid_after[i] = mid

    # ------------------------------------------------------------------ state
    def equity(self, mid: float) -> float:
        """Cash plus the inventory marked at ``mid`` ticks, in price units."""
        return self.cash + self.inventory * mid * self.tick

    @property
    def n_fills(self) -> int:
        return len(self._ts)

    @property
    def volume(self) -> int:
        return int(sum(self._qty))

    def fills(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts": np.asarray(self._ts, dtype=float),
                "side": np.asarray(self._side, dtype=int),
                "price": np.asarray(self._price, dtype=int),
                "qty": np.asarray(self._qty, dtype=int),
                "mid": np.asarray(self._mid, dtype=float),
                "mid_after": np.asarray(self._mid_after, dtype=float),
                "distance": np.asarray(self._distance, dtype=float),
            }
        )

    # ------------------------------------------------------------------ attribution
    def spread_capture(self) -> float:
        side = np.asarray(self._side, dtype=float)
        return float(np.sum(side * (np.asarray(self._mid) - np.asarray(self._price)) * np.asarray(self._qty)) * self.tick)

    def inventory_pnl(self, mid_t: np.ndarray, mid: np.ndarray) -> float:
        """The gain on the held inventory over every mid move, computed directly from the mid series."""
        if len(mid_t) < 2 or not self._ts:
            return 0.0
        fill_ts = np.asarray(self._ts)
        position = np.cumsum(np.asarray(self._side) * np.asarray(self._qty))
        # the position held when the mid moved at mid_t[j]: fills at or before that time (a fill at the same
        # time happened inside the same event, before the mid was re-read)
        count = np.searchsorted(fill_ts, mid_t[1:], side="right")
        held = np.where(count > 0, position[np.maximum(count - 1, 0)], 0)
        return float(np.sum(held * np.diff(mid)) * self.tick)

    def attribution(self, mid_t: np.ndarray, mid: np.ndarray, end_mid: float) -> dict[str, float]:
        total = self.equity(end_mid)
        spread = self.spread_capture()
        inventory = self.inventory_pnl(mid_t, mid)
        return {"total": total, "spread_capture": spread, "inventory": inventory, "fees": self.fees}

    def markouts(self, mid_t: np.ndarray, mid: np.ndarray, horizons: tuple[float, ...], end_time: float) -> pd.DataFrame:
        """Per fill and horizon, in price units per share, NaN past the end of the session:

        - ``markout_h``  = side * (mid(t + h) - mid(t)):   where the mid went after the fill, against the mid the fill
          was marked at. Negative is adverse: the maker was filled right before the price moved against it.
        - ``impact``     = side * (mid_after - mid(t)):    the mechanical part of that: the order that filled the
          maker also moved the touch. Horizon-free.
        - ``drift_h``    = side * (mid(t + h) - mid_after): the rest: where the mid went once the trade was done.
          About zero when nobody knows anything; negative when the takers were informed.
        - ``realised_h`` = side * (mid(t + h) - price):    what the fill was worth h seconds later, spread included.

        ``markout_h = impact + drift_h`` and ``realised_h = (mid(t) - price) * side + markout_h``.
        """
        out = self.fills()
        if out.empty:
            out["impact"] = pd.Series(dtype=float)
            for h in horizons:
                for name in ("markout", "drift", "realised"):
                    out[f"{name}_{h:g}"] = pd.Series(dtype=float)
            return out
        ts = out["ts"].to_numpy()
        side = out["side"].to_numpy()
        mid0 = out["mid"].to_numpy()
        mid_after = out["mid_after"].to_numpy()
        price = out["price"].to_numpy()
        out["impact"] = side * (mid_after - mid0) * self.tick
        for h in horizons:
            at = ts + h
            idx = np.searchsorted(mid_t, at, side="right") - 1
            later = mid[np.clip(idx, 0, len(mid) - 1)]
            unknown = at > end_time
            for name, reference in (("markout", mid0), ("drift", mid_after), ("realised", price)):
                value = side * (later - reference) * self.tick
                value[unknown] = np.nan
                out[f"{name}_{h:g}"] = value
        return out


def is_buy(side: int) -> bool:
    return side == BUY
