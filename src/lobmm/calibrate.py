"""Estimating the Avellaneda-Stoikov inputs from the simulated market itself.

- sigma: the realised volatility of the mid, from a regular grid of mid prices.
- A and k: the fill-intensity curve lambda(delta) = A * exp(-k * delta). The probe maker quotes at a random
  distance from the mid on each side, every wake. Bucketing by distance gives seconds of exposure and a fill
  count per bucket; the intensity is fills / exposure, and an OLS fit of ln(lambda) on delta gives ln(A) and -k.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from lobmm.analytics import realised_vol
from lobmm.config import SimConfig
from lobmm.sim import Simulator


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


SIGNATURE_DTS: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0, 30.0, 60.0)


@dataclass
class Calibration:
    sigma_ticks: float  # ticks per sqrt(second), realised at ``vol_dt``
    sigma: float  # price units per sqrt(second)
    vol_dt: float  # the sampling interval sigma was measured at
    fair_vol_ticks: float  # the total volatility of the fair value (diffusion and news jumps), for comparison
    signature: pd.Series  # realised volatility (ticks per sqrt(second)) at each sampling interval
    fit: IntensityFit
    sessions: int

    def as_overrides(self) -> dict[str, float]:
        return {"maker.sigma": self.sigma, "maker.k": self.fit.k}


def calibrate(
    cfg: SimConfig, seeds: Sequence[int] = (0, 1, 2), max_distance: int | None = None, vol_dt: float = 10.0
) -> Calibration:
    """Run probe sessions on ``cfg`` and estimate sigma, A and k.

    The mid is noisier than the fair value at short horizons (bid-ask bounce, thin levels being swept and
    refilled), so the realised volatility depends on the sampling interval: the signature plot. The A-S
    sigma is taken at ``vol_dt`` seconds, a horizon on the scale the maker holds inventory for.
    """
    overrides: dict[str, object] = {"maker.strategy": "probe"}
    if max_distance is not None:
        overrides["maker.probe_max_distance"] = max_distance
    probe_cfg = cfg.with_(**overrides)
    exposure: dict[float, float] = defaultdict(float)
    fills: dict[float, int] = defaultdict(int)
    dts = sorted(set(SIGNATURE_DTS) | {vol_dt})
    vols = {dt: [] for dt in dts}
    for seed in seeds:
        res = Simulator(probe_cfg, seed).run()
        for d, s in res.exposure.items():
            exposure[d] += s
        for d, n in res.fills_by_distance.items():
            fills[d] += n
        for dt in dts:
            vols[dt].append(realised_vol(res.mid_t, res.mid, res.end_time, dt=dt))
    signature = pd.Series({dt: float(np.nanmean(v)) for dt, v in vols.items()}, name="realised_vol_ticks")
    signature.index.name = "dt_seconds"
    sigma_ticks = float(signature[vol_dt])
    fit = fit_intensity(dict(exposure), dict(fills), cfg.market.tick)
    fair_total = float(np.sqrt(cfg.flow.fair_vol**2 + cfg.flow.jump_rate * cfg.flow.jump_std**2))
    return Calibration(
        sigma_ticks=sigma_ticks,
        sigma=sigma_ticks * cfg.market.tick,
        vol_dt=vol_dt,
        fair_vol_ticks=fair_total,
        signature=signature,
        fit=fit,
        sessions=len(seeds),
    )
