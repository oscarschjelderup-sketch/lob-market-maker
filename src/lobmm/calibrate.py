"""Estimating the Avellaneda-Stoikov inputs, and the adverse-selection term, from the simulated market itself.

- sigma: the realised volatility of the mid, from a regular grid of mid prices.
- A and k: the fill-intensity curve lambda(delta) = A * exp(-k * delta). The probe maker quotes at a random
  distance from the mid on each side, every wake. Bucketing by distance gives seconds of exposure and a fill
  count per bucket; the intensity is fills / exposure, and an OLS fit of ln(lambda) on delta gives ln(A) and -k.
- a and b: the adverse cost of a fill as a function of the quoted distance, cost(delta) = a + b * delta. The
  same probe fills, bucketed by distance, give the realised spread per share h seconds after the fill; the cost
  is the quoted distance minus that, which is the markout (where the mid went after the fill) plus the part of
  the distance the mid had already closed before the fill. A weighted OLS fit on delta gives a and b.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lobmm.analytics import realised_vol
from lobmm.config import SimConfig
from lobmm.sim import SimResult, Simulator, distance_bucket


@dataclass
class IntensityFit:
    A: float  # fills per second at zero distance
    k: float  # decay per price unit
    k_ticks: float  # decay per tick
    r2: float
    table: pd.DataFrame  # distance_ticks, exposure_s, fills, intensity, fitted

    def intensity(self, distance_price: float) -> float:
        return self.A * np.exp(-self.k * distance_price)


def fit_intensity(exposure: dict[float, float], fills: dict[float, int], tick: float, min_exposure: float = 5.0) -> IntensityFit:
    rows = []
    for distance, seconds in sorted(exposure.items()):
        n = fills.get(distance, 0)
        rows.append({"distance_ticks": distance, "exposure_s": seconds, "fills": n, "intensity": n / seconds if seconds > 0 else np.nan})
    table = pd.DataFrame(rows)
    usable = table[(table["exposure_s"] >= min_exposure) & (table["fills"] > 0)]
    if len(usable) < 2:
        raise ValueError("not enough distance buckets with fills to fit the intensity curve; run longer sessions")
    x = usable["distance_ticks"].to_numpy() * tick
    y = np.log(usable["intensity"].to_numpy())
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    ss_res = float(np.sum((y - fitted) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    A, k = float(np.exp(intercept)), float(-slope)
    table["fitted"] = A * np.exp(-k * table["distance_ticks"] * tick)
    return IntensityFit(A=A, k=k, k_ticks=k * tick, r2=r2, table=table)


@dataclass
class MarkoutFit:
    """The adverse cost of a fill against the distance it was quoted at: cost(delta) = a + b * delta, price units.

    Avellaneda and Stoikov choose the distance that maximises lambda(delta) * delta, which for small gamma is
    1/k. If a fill quoted at delta is worth only delta - a - b * delta h seconds later, the same argument
    maximises lambda(delta) * ((1 - b) * delta - a) instead, and the optimum moves out to 1/k + a / (1 - b).
    The difference, a / (1 - b), is the adverse-selection term: what to add to each side of the A-S spread.
    """

    horizon: float  # the markout horizon the cost is measured at, seconds
    a: float  # cost of a fill at zero distance, price units per share
    b: float  # cost per price unit of distance: the share of each extra unit of spread that is lost again
    r2: float
    table: pd.DataFrame  # distance_ticks, fills, shares, markout_ticks, stale_ticks, realised_ticks, cost_ticks, fitted_ticks

    def cost(self, distance_price: float) -> float:
        return self.a + self.b * distance_price

    @property
    def adverse_term(self) -> float:
        """The extra half-spread, price units: a / (1 - b)."""
        if self.b >= 1.0:
            raise ValueError(f"the adverse cost grows at least as fast as the distance (b = {self.b:.2f}): no distance pays")
        return self.a / (1.0 - self.b)

    def optimal_distance(self, k: float) -> float:
        """The risk-neutral optimum of lambda(delta) * (delta - cost(delta)) with lambda = A exp(-k delta), price units."""
        return 1.0 / k + self.adverse_term


def fit_markout(fills: pd.DataFrame, tick: float, horizon: float, min_fills: int = 5) -> MarkoutFit:
    """Fit the adverse cost on fills with columns ``distance`` (ticks, bucketed), ``qty``, ``markout`` and ``realised``
    (price units per share, at ``horizon``). Buckets are weighted by shares; those with fewer than ``min_fills``
    fills are shown in the table but left out of the fit."""
    rows = []
    for distance, g in fills.groupby("distance", sort=True):
        shares = float(g["qty"].sum())
        if shares <= 0:
            continue
        markout = float(np.sum(g["markout"] * g["qty"]) / shares) / tick
        realised = float(np.sum(g["realised"] * g["qty"]) / shares) / tick
        capture = realised - markout  # side * (mid at the fill - price): the distance left when the fill came
        rows.append(
            {
                "distance_ticks": float(distance),
                "fills": int(len(g)),
                "shares": int(shares),
                "markout_ticks": markout,
                "stale_ticks": float(distance) - capture,
                "realised_ticks": realised,
                "cost_ticks": float(distance) - realised,
            }
        )
    table = pd.DataFrame(rows)
    usable = table[table["fills"] >= min_fills] if len(table) else table
    if len(usable) < 2:
        raise ValueError("not enough distance buckets with fills to fit the adverse cost; run longer sessions")
    x = usable["distance_ticks"].to_numpy() * tick
    y = usable["cost_ticks"].to_numpy() * tick
    w = usable["shares"].to_numpy(dtype=float)
    b, a = np.polyfit(x, y, 1, w=np.sqrt(w))
    fitted = a + b * x
    ss_res = float(np.sum(w * (y - fitted) ** 2))
    y_bar = float(np.sum(w * y) / np.sum(w))
    ss_tot = float(np.sum(w * (y - y_bar) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    table["fitted_ticks"] = (a + b * table["distance_ticks"] * tick) / tick
    return MarkoutFit(horizon=horizon, a=float(a), b=float(b), r2=r2, table=table)


def probe_fills(res: SimResult, horizon: float) -> pd.DataFrame:
    """The fills of a session as ``fit_markout`` wants them: distance bucket, qty, markout and realised at ``horizon``."""
    mk = res.markouts()
    if mk.empty:
        return pd.DataFrame({"distance": [], "qty": [], "markout": [], "realised": []})
    out = pd.DataFrame(
        {
            "distance": mk["distance"].map(distance_bucket),
            "qty": mk["qty"],
            "markout": mk[f"markout_{horizon:g}"],
            "realised": mk[f"realised_{horizon:g}"],
        }
    )
    return out.dropna()


SIGNATURE_DTS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0)


@dataclass
class Calibration:
    sigma_ticks: float  # ticks per sqrt(second), realised at ``vol_dt``
    sigma: float  # price units per sqrt(second)
    vol_dt: float  # the sampling interval sigma was measured at
    fair_vol_ticks: float  # the total volatility of the fair value (diffusion and news jumps), for comparison
    signature: pd.Series  # realised volatility (ticks per sqrt(second)) at each sampling interval
    fit: IntensityFit
    markout: MarkoutFit
    sessions: int

    def as_overrides(self) -> dict[str, float]:
        return {"maker.sigma": self.sigma, "maker.k": self.fit.k, "maker.adverse": self.markout.adverse_term}


def calibrate(
    cfg: SimConfig,
    seeds: Sequence[int] = (0, 1, 2),
    max_distance: int | None = None,
    vol_dt: float = 10.0,
    markout_horizon: float = 10.0,
) -> Calibration:
    """Run probe sessions on ``cfg`` and estimate sigma, A, k and the adverse cost a + b * delta.

    The mid is noisier than the fair value at short horizons (bid-ask bounce, thin levels being swept and
    refilled), so the realised volatility depends on the sampling interval: the signature plot. The A-S
    sigma is taken at ``vol_dt`` seconds, a horizon on the scale the maker holds inventory for. The adverse
    cost is measured ``markout_horizon`` seconds after each fill.
    """
    overrides: dict[str, object] = {"maker.strategy": "probe"}
    if max_distance is not None:
        overrides["maker.probe_max_distance"] = max_distance
    probe_cfg = cfg.with_(**overrides, horizons=tuple(sorted(set(cfg.horizons) | {markout_horizon})))
    exposure: dict[float, float] = defaultdict(float)
    fills: dict[float, int] = defaultdict(int)
    dts = sorted(set(SIGNATURE_DTS) | {vol_dt})
    vols = {dt: [] for dt in dts}
    fill_frames = []
    for seed in seeds:
        res = Simulator(probe_cfg, seed).run()
        for d, s in res.exposure.items():
            exposure[d] += s
        for d, n in res.fills_by_distance.items():
            fills[d] += n
        for dt in dts:
            vols[dt].append(realised_vol(res.mid_t, res.mid, res.end_time, dt=dt))
        fill_frames.append(probe_fills(res, markout_horizon))
    signature = pd.Series({dt: float(np.nanmean(v)) for dt, v in vols.items()}, name="realised_vol_ticks")
    signature.index.name = "dt_seconds"
    sigma_ticks = float(signature[vol_dt])
    fit = fit_intensity(dict(exposure), dict(fills), cfg.market.tick)
    markout = fit_markout(pd.concat(fill_frames, ignore_index=True), cfg.market.tick, markout_horizon)
    fair_total = float(np.sqrt(cfg.flow.fair_vol**2 + cfg.flow.jump_rate * cfg.flow.jump_std**2))
    return Calibration(
        sigma_ticks=sigma_ticks,
        sigma=sigma_ticks * cfg.market.tick,
        vol_dt=vol_dt,
        fair_vol_ticks=fair_total,
        signature=signature,
        fit=fit,
        markout=markout,
        sessions=len(seeds),
    )
