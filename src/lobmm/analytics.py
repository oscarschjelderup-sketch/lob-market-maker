"""Summaries of a session, comparisons over seeds, parameter sweeps and ablations."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

from lobmm.config import SimConfig
from lobmm.sim import SimResult, Simulator


def realised_vol(mid_t: np.ndarray, mid: np.ndarray, end_time: float, dt: float = 1.0) -> float:
    """Standard deviation of mid changes over a regular ``dt`` grid, in ticks per sqrt(second)."""
    grid = np.arange(0.0, end_time, dt)
    if len(grid) < 3:
        return float("nan")
    idx = np.searchsorted(mid_t, grid, side="right") - 1
    sampled = mid[np.clip(idx, 0, len(mid) - 1)]
    return float(np.std(np.diff(sampled), ddof=1) / np.sqrt(dt))


def summarize(res: SimResult) -> dict[str, Any]:
    """The numbers one session is judged on: PnL and its attribution, inventory risk, fills, markouts, speed."""
    att = res.attribution()
    s = res.series
    out: dict[str, Any] = {
        "seed": res.seed,
        "strategy": res.strategy,
        "pnl": att["total"],
        "spread_capture": att["spread_capture"],
        "inventory_pnl": att["inventory"],
        "fees": att["fees"],
        "n_fills": res.ledger.n_fills,
        "volume": res.ledger.volume,
        "fills_per_min": res.ledger.n_fills / (res.end_time / 60.0),
        "end_inventory": res.ledger.inventory,
        "inv_mean": float(s["inventory"].mean()) if len(s) else 0.0,
        "inv_std": float(s["inventory"].std(ddof=0)) if len(s) else 0.0,
        "inv_max_abs": int(s["inventory"].abs().max()) if len(s) else 0,
        "at_limit": float((s["inventory"].abs() >= res.cfg.maker.max_inventory).mean()) if len(s) else 0.0,
        "avg_spread": float((s["best_ask"] - s["best_bid"]).mean()) if len(s) else float("nan"),
        "quote_distance": float(((s["ask"] - s["bid"]) / 2.0).mean()) if len(s) and (s["ask"] - s["bid"]).notna().any() else float("nan"),
        "mid_vol": realised_vol(res.mid_t, res.mid, res.end_time),
        "end_mid": res.end_mid,
        "events": res.n_events,
        "runtime": res.runtime,
        "events_per_sec": res.n_events / res.runtime if res.runtime > 0 else float("nan"),
        "killed_at": res.killed_at,
    }
    mk = res.markouts()
    out["impact"] = float(mk["impact"].mean()) if len(mk) else float("nan")
    for h in res.cfg.horizons:
        col = mk[f"markout_{h:g}"]
        out[f"markout_{h:g}"] = float(col.mean()) if col.notna().any() else float("nan")
        out[f"adverse_{h:g}"] = float(-(col * mk["qty"]).sum()) if col.notna().any() else 0.0
        for name in ("drift", "realised"):
            series = mk[f"{name}_{h:g}"]
            out[f"{name}_{h:g}"] = float(series.mean()) if series.notna().any() else float("nan")
    return out


def run_many(cfg: SimConfig, seeds: Iterable[int], variant: str | None = None, **overrides: Any) -> pd.DataFrame:
    """One summary row per seed for ``cfg`` with the dotted ``overrides`` applied."""
    if overrides:
        cfg = cfg.with_(**overrides)
    rows = []
    for seed in seeds:
        row = Simulator(cfg, seed).run().summary()
        if variant is not None:
            row["variant"] = variant
        rows.append(row)
    return pd.DataFrame(rows)


AGGREGATE_MEANS = [
    "spread_capture",
    "inventory_pnl",
    "fees",
    "n_fills",
    "inv_std",
    "inv_max_abs",
    "at_limit",
    "avg_spread",
    "quote_distance",
    "mid_vol",
    "impact",
]


def aggregate(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean PnL, its dispersion over seeds and a t-statistic, plus the mean of every other metric, per variant."""
    rows = []
    for variant, g in per_seed.groupby("variant", sort=False):
        n = len(g)
        pnl = g["pnl"].to_numpy(dtype=float)
        row: dict[str, Any] = {
            "variant": variant,
            "seeds": n,
            "pnl_mean": pnl.mean(),
            "pnl_std": pnl.std(ddof=1) if n > 1 else float("nan"),
            "pnl_t": pnl.mean() / (pnl.std(ddof=1) / np.sqrt(n)) if n > 1 and pnl.std(ddof=1) > 0 else float("nan"),
            "pnl_min": pnl.min(),
            "pnl_max": pnl.max(),
        }
        for col in AGGREGATE_MEANS:
            row[col] = float(g[col].mean())
        row["pnl_per_inv_risk"] = row["pnl_mean"] / row["inv_std"] if row["inv_std"] > 0 else float("nan")
        for col in g.columns:
            if col.startswith(("markout_", "adverse_", "drift_", "realised_")):
                row[col] = float(g[col].mean())
        row["killed"] = int(g["killed_at"].notna().sum())
        rows.append(row)
    return pd.DataFrame(rows).set_index("variant")


def compare(cfg: SimConfig, variants: dict[str, dict[str, Any]], seeds: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run every variant (name -> dotted overrides) on the same seeds. Returns (per-seed rows, aggregate table)."""
    frames = [run_many(cfg, seeds, variant=name, **overrides) for name, overrides in variants.items()]
    per_seed = pd.concat(frames, ignore_index=True)
    return per_seed, aggregate(per_seed)


def sweep(cfg: SimConfig, key: str, values: Sequence[Any], seeds: Sequence[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """A parameter sweep: ``key`` is dotted (``maker.gamma``), one variant per value."""
    variants = {f"{key.split('.')[-1]}={v:g}" if isinstance(v, float) else f"{key.split('.')[-1]}={v}": {key: v} for v in values}
    per_seed, agg = compare(cfg, variants, seeds)
    agg.insert(0, "value", list(values))
    return per_seed, agg


DEFAULT_STRATEGIES: dict[str, dict[str, Any]] = {
    "naive h=3": {"maker.strategy": "naive", "maker.half_spread": 3},
    "A-S gamma=0.003": {"maker.strategy": "as", "maker.gamma": 0.003},
    "A-S+adverse gamma=0.003": {"maker.strategy": "as_adverse", "maker.gamma": 0.003},
}

ABLATIONS: dict[str, dict[str, Any]] = {
    "A-S base": {"maker.strategy": "as"},
    "+ adverse-selection term": {"maker.strategy": "as_adverse"},
    "+ latency 200 ms": {"maker.strategy": "as", "maker.latency": 0.2},
    "+ latency 1 s": {"maker.strategy": "as", "maker.latency": 1.0},
    "+ maker fee 0.5 bps": {"maker.strategy": "as", "maker.maker_fee_bps": 0.5},
    "+ maker rebate 0.2 bps": {"maker.strategy": "as", "maker.maker_fee_bps": -0.2},
    "no informed flow": {"maker.strategy": "as", "flow.informed_rate": 0.0},
    "informed flow x2": {"maker.strategy": "as", "flow.informed_rate": 2.0},
    "no news jumps": {"maker.strategy": "as", "flow.jump_rate": 0.0},
    "news jumps x4": {"maker.strategy": "as", "flow.jump_rate": 4.0 / 120.0},
    "inventory limit 20": {"maker.strategy": "as", "maker.max_inventory": 20},
    "kill switch at 5.0": {"maker.strategy": "as", "maker.kill_drawdown": 5.0},
}


def ablation(cfg: SimConfig, seeds: Sequence[int], variants: dict[str, dict[str, Any]] | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    return compare(cfg, variants or ABLATIONS, seeds)


def bench(cfg: SimConfig, horizon: float = 600.0, seed: int = 0) -> pd.DataFrame:
    """Events per second of the engine with no market maker and with the A-S maker."""
    rows = []
    for strategy in ("none", "as"):
        res = Simulator(cfg.with_(**{"market.horizon": horizon, "maker.strategy": strategy}), seed).run()
        rows.append(
            {
                "strategy": strategy,
                "session_seconds": horizon,
                "events": res.n_events,
                "fills": res.ledger.n_fills if strategy != "none" else 0,
                "book_trades": sum(1 for _ in []) + res.counts.get("market", 0) + res.counts.get("informed_trades", 0),
                "runtime_s": res.runtime,
                "events_per_sec": res.n_events / res.runtime,
            }
        )
    return pd.DataFrame(rows).set_index("strategy")
