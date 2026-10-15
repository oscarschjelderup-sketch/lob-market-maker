"""The command line: ``lobmm run``, ``compare``, ``sweep``, ``calibrate``, ``ablate``, ``bench``, ``figures``.

Every command takes ``--set group.name=value`` to override any parameter in ``lobmm.config``.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pandas as pd
import typer
from rich.console import Console
from rich.table import Table

from lobmm.config import SimConfig

app = typer.Typer(add_completion=False, help="lob-market-maker: a limit order book market-making simulator.")
console = Console()

SetOption = typer.Option(None, "--set", "-s", help="override a parameter, e.g. --set maker.gamma=0.3 --set flow.informed_rate=0")


def _parse(value: str) -> Any:
    lowered = value.lower()
    if lowered in ("none", "null"):
        return None
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def _config(horizon: float | None, sets: list[str] | None, **extra: Any) -> SimConfig:
    overrides: dict[str, Any] = {}
    if horizon is not None:
        overrides["market.horizon"] = horizon
    for item in sets or []:
        if "=" not in item:
            raise typer.BadParameter(f"--set expects group.name=value, got {item!r}")
        key, value = item.split("=", 1)
        overrides[key.strip()] = _parse(value.strip())
    overrides.update({k: v for k, v in extra.items() if v is not None})
    try:
        return SimConfig().with_(**overrides)
    except KeyError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _fmt(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        if math.isnan(v):
            return ""
        return f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.3f}"
    return str(v)


def _print_frame(df: pd.DataFrame, title: str, index: bool = True) -> None:
    table = Table(title=title, title_justify="left", header_style="bold")
    frame = df.reset_index() if index else df
    for col in frame.columns:
        table.add_column(str(col), justify="right" if pd.api.types.is_numeric_dtype(frame[col]) else "left")
    for _, row in frame.iterrows():
        table.add_row(*[_fmt(v) for v in row.tolist()])
    console.print(table)


def _print_dict(d: dict[str, Any], title: str) -> None:
    table = Table(title=title, title_justify="left", show_header=False)
    table.add_column("metric")
    table.add_column("value", justify="right")
    for k, v in d.items():
        table.add_row(k, _fmt(v))
    console.print(table)


@app.command()
def run(
    strategy: str = typer.Option("as", help="naive, as, as_adverse, probe or none"),
    seed: int = typer.Option(0),
    horizon: float = typer.Option(None, help="session length in seconds (default 3600)"),
    plot: bool = typer.Option(False, help="write the session figures to --out"),
    out: Path = typer.Option(Path("figures"), help="folder for the figures"),
    sets: list[str] = SetOption,
) -> None:
    """One session: the summary, the PnL attribution and the markouts."""
    from lobmm.sim import Simulator

    cfg = _config(horizon, sets, **{"maker.strategy": strategy, "sample_interval": 1.0 if plot else None})
    res = Simulator(cfg, seed).run()
    s = res.summary()
    _print_dict({k: v for k, v in s.items() if not k.startswith("markout_") and not k.startswith("adverse_")}, f"{strategy} maker, seed {seed}, {cfg.market.horizon:g} s")
    _print_dict(res.attribution(), "PnL attribution (price units)")
    _print_dict({f"mean markout at {h:g} s (ticks/share)": s[f"markout_{h:g}"] / cfg.market.tick for h in cfg.horizons} | {f"adverse selection cost at {h:g} s": s[f"adverse_{h:g}"] for h in cfg.horizons}, "markouts")
    _print_dict(res.counts, "events")
    if plot:
        from lobmm import plots

        paths = [plots.plot_session(res, out / f"session_{strategy}_seed{seed}.png")]
        if res.samples:
            paths.append(plots.plot_book_heatmap(res, out / f"book_heatmap_{strategy}_seed{seed}.png"))
        for p in paths:
            console.print(f"wrote {p}")


@app.command()
def compare(
    seeds: int = typer.Option(20, help="number of seeds"),
    horizon: float = typer.Option(None),
    sets: list[str] = SetOption,
) -> None:
    """The naive maker against Avellaneda-Stoikov on the same seeds."""
    from lobmm.analytics import DEFAULT_STRATEGIES
    from lobmm.analytics import compare as _compare
    from lobmm.report import COMPARE_COLUMNS

    cfg = _config(horizon, sets)
    _, agg = _compare(cfg, DEFAULT_STRATEGIES, list(range(seeds)))
    _print_frame(agg[COMPARE_COLUMNS], f"naive against A-S, {seeds} seeds, {cfg.market.horizon:g} s")


@app.command()
def sweep(
    param: str = typer.Option("maker.gamma", help="dotted parameter to sweep"),
    values: str = typer.Option("0.01,0.03,0.1,0.3,1", help="comma-separated values"),
    seeds: int = typer.Option(20),
    horizon: float = typer.Option(None),
    sets: list[str] = SetOption,
) -> None:
    """A parameter sweep, one aggregate row per value."""
    from lobmm.analytics import sweep as _sweep
    from lobmm.report import SWEEP_COLUMNS

    cfg = _config(horizon, sets)
    parsed = [_parse(v.strip()) for v in values.split(",")]
    _, agg = _sweep(cfg, param, parsed, list(range(seeds)))
    _print_frame(agg[SWEEP_COLUMNS], f"sweep of {param}, {seeds} seeds")


@app.command()
def calibrate(
    seeds: int = typer.Option(3, help="number of probe sessions"),
    horizon: float = typer.Option(None),
    vol_dt: float = typer.Option(10.0, help="sampling interval for sigma, seconds"),
    markout_horizon: float = typer.Option(10.0, help="horizon the adverse cost of a fill is measured at, seconds"),
    sets: list[str] = SetOption,
) -> None:
    """Estimate sigma, A, k and the adverse-selection term from probe sessions: the inputs of the A-S quotes."""
    from lobmm.calibrate import calibrate as _calibrate

    cfg = _config(horizon, sets)
    cal = _calibrate(cfg, seeds=list(range(seeds)), vol_dt=vol_dt, markout_horizon=markout_horizon)
    tick = cfg.market.tick
    _print_dict(
        {
            "sigma (ticks / sqrt s)": cal.sigma_ticks,
            "sigma (price / sqrt s)": cal.sigma,
            "fair value vol (ticks / sqrt s)": cal.fair_vol_ticks,
            "A (fills / s at the mid)": cal.fit.A,
            "k (per price unit)": cal.fit.k,
            "k (per tick)": cal.fit.k_ticks,
            "R² of the intensity fit": cal.fit.r2,
            f"adverse cost at {markout_horizon:g} s: a (ticks)": cal.markout.a / tick,
            "adverse cost: b (per tick of distance)": cal.markout.b,
            "R² of the cost fit": cal.markout.r2,
            "adverse term a / (1 - b) (ticks)": cal.markout.adverse_term / tick,
            "risk-neutral distance 1/k (ticks)": 1.0 / cal.fit.k_ticks,
            "with the term (ticks)": cal.markout.optimal_distance(cal.fit.k) / tick,
            "probe sessions": cal.sessions,
        },
        "calibration",
    )
    _print_frame(cal.signature.to_frame(), "volatility signature")
    _print_frame(cal.fit.table.set_index("distance_ticks"), "fill intensity by distance from the mid")
    _print_frame(cal.markout.table.set_index("distance_ticks"), f"adverse cost by distance, {markout_horizon:g} s after the fill (ticks per share)")
    console.print("use with:  " + " ".join(f"--set {k}={v:.5g}" for k, v in cal.as_overrides().items()))


@app.command()
def ablate(
    seeds: int = typer.Option(10),
    horizon: float = typer.Option(None),
    sets: list[str] = SetOption,
) -> None:
    """Which friction hurts the A-S maker most: latency, fees, informed flow, limits."""
    from lobmm.analytics import ablation
    from lobmm.report import ABLATION_COLUMNS

    cfg = _config(horizon, sets)
    _, agg = ablation(cfg, list(range(seeds)))
    _print_frame(agg[ABLATION_COLUMNS], f"ablations, {seeds} seeds")


@app.command()
def bench(horizon: float = typer.Option(600.0), sets: list[str] = SetOption) -> None:
    """Events per second of the engine."""
    from lobmm.analytics import bench as _bench

    b = _bench(_config(None, sets), horizon=horizon)
    _print_frame(b, "benchmark")


@app.command()
def figures(
    out: Path = typer.Option(Path("figures")),
    seeds: int = typer.Option(20),
    horizon: float = typer.Option(3600.0),
    sets: list[str] = SetOption,
) -> None:
    """Regenerate every figure and table of the README into --out (results.json, results.md)."""
    from lobmm.report import make_figures

    make_figures(out, seeds=range(seeds), horizon=horizon, cfg=_config(None, sets), log=console.print)


if __name__ == "__main__":
    app()
