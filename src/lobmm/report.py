"""``lobmm figures``: every experiment in the README, regenerated from seeds into one folder.

Writes the PNGs, ``results.json`` (every table) and ``results.md`` (the same tables as markdown).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from lobmm import plots
from lobmm.analytics import DEFAULT_STRATEGIES, ablation, bench, compare, sweep
from lobmm.calibrate import calibrate
from lobmm.config import SimConfig
from lobmm.sim import Simulator

GAMMAS = (0.001, 0.003, 0.01, 0.03, 0.1)
HALF_SPREADS = (1, 2, 3, 5, 8)
ADVERSE_TICKS = (0.0, 1.0, 2.0, 3.0, 5.0, 8.0)  # the adverse-selection term swept, ticks each side, at the default gamma
MARKOUT_VARIANTS = ("naive h=3", "A-S gamma=0.003", "A-S+adverse gamma=0.003", "no informed flow", "no news jumps")
MARKOUT_STYLES = {
    "naive h=3": (plots.ORANGE, "-"),
    "A-S gamma=0.003": (plots.BLUE, "-"),
    "A-S+adverse gamma=0.003": (plots.BLUE, "--"),
    "no informed flow": (plots.AQUA, "-"),
    "no news jumps": (plots.YELLOW, "-"),
}


def markdown_table(df: pd.DataFrame, floatfmt: str = ".3f", index: bool = True) -> str:
    """A GitHub-flavoured markdown table, no extra dependency."""
    frame = df.reset_index() if index else df
    cols = [str(c) for c in frame.columns]

    def fmt(v: Any) -> str:
        if isinstance(v, float):
            return "" if math.isnan(v) else format(v, floatfmt)
        return str(v)

    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(fmt(v) for v in row.tolist()) + " |")
    return "\n".join(lines)


def _records(df: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for rec in df.reset_index().to_dict(orient="records"):
        out.append({k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in rec.items()})
    return out


def make_figures(
    out: Path | str = "figures",
    seeds: Sequence[int] = range(20),
    horizon: float = 3600.0,
    cfg: SimConfig | None = None,
    calibration_seeds: Sequence[int] = (0, 1, 2),
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = list(seeds)
    cfg = (cfg or SimConfig()).with_(**{"market.horizon": horizon})
    started = time.perf_counter()
    results: dict[str, Any] = {"seeds": seeds, "horizon": horizon}

    log("calibrating sigma, A, k and the adverse cost from probe sessions")
    cal = calibrate(cfg, seeds=calibration_seeds)
    tick = cfg.market.tick
    results["calibration"] = {
        "sigma_ticks": cal.sigma_ticks,
        "sigma": cal.sigma,
        "vol_dt": cal.vol_dt,
        "fair_vol_ticks": cal.fair_vol_ticks,
        "A": cal.fit.A,
        "k": cal.fit.k,
        "k_ticks": cal.fit.k_ticks,
        "r2": cal.fit.r2,
        "sessions": cal.sessions,
        "signature": {f"{dt:g}": v for dt, v in cal.signature.items()},
        "intensity_table": _records(cal.fit.table.set_index("distance_ticks")),
        "markout_horizon": cal.markout.horizon,
        "adverse_a": cal.markout.a,
        "adverse_a_ticks": cal.markout.a / tick,
        "adverse_b": cal.markout.b,
        "adverse_r2": cal.markout.r2,
        "adverse_term": cal.markout.adverse_term,
        "adverse_term_ticks": cal.markout.adverse_term / tick,
        "risk_neutral_distance_ticks": 1.0 / cal.fit.k_ticks,
        "optimal_distance_ticks": cal.markout.optimal_distance(cal.fit.k) / tick,
        "markout_table": _records(cal.markout.table.set_index("distance_ticks")),
    }
    plots.plot_intensity(cal.fit, out / "intensity_fit.png")
    plots.plot_signature(cal.signature, cal.fair_vol_ticks, out / "volatility_signature.png")
    plots.plot_markout_fit(cal.markout, cal.fit, tick, out / "markout_fit.png")
    cfg = cfg.with_(**cal.as_overrides())
    results["config"] = {"market": cfg.market.__dict__, "flow": cfg.flow.__dict__, "maker": cfg.maker.__dict__, "horizons": cfg.horizons}

    log("one session of each maker on seed 0")
    as0 = Simulator(cfg.with_(**{"maker.strategy": "as", "sample_interval": 1.0}), 0).run()
    naive0 = Simulator(cfg.with_(**{"maker.strategy": "naive"}), 0).run()
    plots.plot_session(as0, out / "session_as.png")
    plots.plot_session(naive0, out / "session_naive.png")
    plots.plot_book_heatmap(as0, out / "book_heatmap.png")
    plots.plot_naive_vs_as(naive0, as0, out / "naive_vs_as.png")
    results["session_seed0"] = {"as": as0.summary(), "naive": naive0.summary()}

    log(f"naive against Avellaneda-Stoikov, with and without the adverse-selection term, over {len(seeds)} seeds")
    per_seed, agg = compare(cfg, DEFAULT_STRATEGIES, seeds)
    plots.plot_attribution(agg, out / "pnl_attribution.png")
    results["compare"] = _records(agg)
    results["compare_per_seed"] = _records(per_seed.set_index("variant"))

    log("sweeping gamma (with and without the term) and the naive half-spread")
    _, agg_gamma = sweep(cfg, "maker.gamma", list(GAMMAS), seeds)
    _, agg_gamma_adv = sweep(cfg.with_(**{"maker.strategy": "as_adverse"}), "maker.gamma", list(GAMMAS), seeds)
    _, agg_h = sweep(cfg.with_(**{"maker.strategy": "naive"}), "maker.half_spread", list(HALF_SPREADS), seeds)
    plots.plot_frontier(agg_gamma, agg_h, out / "risk_return_frontier.png", agg_adverse=agg_gamma_adv)
    results["sweep_gamma"] = _records(agg_gamma)
    results["sweep_gamma_adverse"] = _records(agg_gamma_adv)
    results["sweep_half_spread"] = _records(agg_h)

    log("sweeping the adverse-selection term")
    adverse_variants = {f"adverse={a:g} ticks": {"maker.strategy": "as_adverse", "maker.adverse": a * tick} for a in ADVERSE_TICKS}
    _, agg_adv = compare(cfg, adverse_variants, seeds)
    agg_adv.insert(0, "value", list(ADVERSE_TICKS))
    results["sweep_adverse"] = _records(agg_adv)

    log(f"ablations over {len(seeds)} seeds")
    _, agg_ab = ablation(cfg, seeds)
    results["ablation"] = _records(agg_ab)
    both = pd.concat([agg, agg_ab])
    plots.plot_markouts(both.loc[[v for v in MARKOUT_VARIANTS if v in both.index]], cfg.horizons, cfg.market.tick, out / "markouts.png", styles=MARKOUT_STYLES)

    log("benchmark")
    b = bench(cfg)
    results["bench"] = _records(b)
    results["runtime_s"] = time.perf_counter() - started

    (out / "results.json").write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    (out / "results.md").write_text(
        results_markdown(results, agg, agg_gamma, agg_gamma_adv, agg_h, agg_adv, agg_ab, b, cal.signature, cal.fit.table, cal.markout.table),
        encoding="utf-8",
    )
    log(f"done in {results['runtime_s']:.0f} s; figures and tables in {out}")
    return results


COMPARE_COLUMNS = ["seeds", "pnl_mean", "pnl_std", "pnl_t", "spread_capture", "inventory_pnl", "fees", "n_fills", "quote_distance", "inv_std", "inv_max_abs", "at_limit", "impact", "drift_10", "markout_10", "realised_10", "adverse_10"]
SWEEP_COLUMNS = ["value", "pnl_mean", "pnl_std", "spread_capture", "inventory_pnl", "n_fills", "quote_distance", "inv_std", "inv_max_abs", "markout_10", "realised_10"]
ABLATION_COLUMNS = ["pnl_mean", "pnl_std", "spread_capture", "inventory_pnl", "fees", "n_fills", "quote_distance", "inv_std", "impact", "drift_10", "markout_10", "realised_10", "killed"]


def results_markdown(results, agg, agg_gamma, agg_gamma_adv, agg_h, agg_adv, agg_ab, b, signature, intensity_table, markout_table) -> str:
    cal = results["calibration"]
    parts = [
        "# Results\n",
        f"Seeds: {len(results['seeds'])}, session length {results['horizon']:g} s. Generated by `lobmm figures`.\n",
        "## Calibration\n",
        f"sigma = {cal['sigma_ticks']:.3f} ticks/sqrt(s) at dt = {cal['vol_dt']:g} s (fair value: {cal['fair_vol_ticks']:.3f}); "
        f"A = {cal['A']:.3f} fills/s, k = {cal['k']:.1f} per price unit ({cal['k_ticks']:.3f} per tick), R² = {cal['r2']:.3f}, {cal['sessions']} probe sessions.\n",
        f"Adverse cost of a fill at {cal['markout_horizon']:g} s: a = {cal['adverse_a_ticks']:.3f} ticks, b = {cal['adverse_b']:.3f} per tick of distance, R² = {cal['adverse_r2']:.3f}; "
        f"the term a / (1 − b) = {cal['adverse_term_ticks']:.3f} ticks moves the risk-neutral distance from 1/k = {cal['risk_neutral_distance_ticks']:.3f} to {cal['optimal_distance_ticks']:.3f} ticks.\n",
        "### Volatility signature\n",
        markdown_table(signature.to_frame(), ".3f"),
        "\n### Fill intensity by distance\n",
        markdown_table(intensity_table.set_index("distance_ticks"), ".3f"),
        "\n### Adverse cost by distance (ticks per share)\n",
        markdown_table(markout_table.set_index("distance_ticks"), ".3f"),
        "\n## Naive against Avellaneda-Stoikov\n",
        markdown_table(agg[COMPARE_COLUMNS], ".3f"),
        "\n## Gamma sweep (A-S)\n",
        markdown_table(agg_gamma[SWEEP_COLUMNS], ".3f"),
        "\n## Gamma sweep (A-S with the adverse-selection term)\n",
        markdown_table(agg_gamma_adv[SWEEP_COLUMNS], ".3f"),
        "\n## Half-spread sweep (naive)\n",
        markdown_table(agg_h[SWEEP_COLUMNS], ".3f"),
        "\n## Adverse-term sweep (A-S, ticks each side)\n",
        markdown_table(agg_adv[SWEEP_COLUMNS], ".3f"),
        "\n## Ablations (A-S)\n",
        markdown_table(agg_ab[ABLATION_COLUMNS], ".3f"),
        "\n## Benchmark\n",
        markdown_table(b, ".0f"),
        "",
    ]
    return "\n".join(parts)
