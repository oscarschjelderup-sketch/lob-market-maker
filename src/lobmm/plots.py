"""The figures: one palette, thin marks, a recessive grid, a legend whenever there is more than one series.

Blue is the Avellaneda-Stoikov maker, orange the naive maker, aqua the fair value; depth in the heatmap is
one blue ramp for bids and one orange ramp for asks; text is ink, never a series colour.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from lobmm.book import BUY  # noqa: E402
from lobmm.calibrate import IntensityFit  # noqa: E402
from lobmm.sim import SimResult  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
ORANGE_RAMP = ["#fde3d8", "#f9c0a9", "#f59c7a", "#eb6834", "#c85222", "#9d3e17", "#6f2a0d"]
VARIANT_COLOURS = [BLUE, ORANGE, AQUA, YELLOW]


def _style(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=INK2, labelsize=9)
    if title:
        ax.set_title(title, loc="left", color=INK, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel, color=INK2, fontsize=9)
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def _legend(ax, **kw) -> None:
    leg = ax.legend(frameon=False, fontsize=8, labelcolor=INK2, **kw)
    for text in leg.get_texts():
        text.set_color(INK2)


def _figure(nrows: int = 1, ncols: int = 1, width: float = 9.0, height: float = 4.2, sharex: bool = False):
    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height), sharex=sharex)
    fig.patch.set_facecolor(SURFACE)
    return fig, axes


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- one session
def plot_session(res: SimResult, path: Path, window: tuple[float, float] | None = None) -> Path:
    """Mid, fair value, the quotes and fills of the maker; its inventory; its equity. One session."""
    tick = res.cfg.market.tick
    s = res.series
    t0, t1 = window if window else (0.0, res.end_time)
    fig, (ax_p, ax_i, ax_e) = _figure(3, 1, height=9.0, sharex=True)

    ax_p.step(res.mid_t, res.mid * tick, where="post", color=INK, lw=0.9, label="mid")
    ax_p.plot(res.fair_t, res.fair * tick, color=AQUA, lw=1.2, label="fair value")
    ax_p.plot(s["t"], s["bid"] * tick, color=BLUE, lw=0.8, ls="--", label="maker bid")
    ax_p.plot(s["t"], s["ask"] * tick, color=BLUE, lw=0.8, ls=":", label="maker ask")
    f = res.fills
    buys, sells = f[f["side"] == BUY], f[f["side"] != BUY]
    ax_p.scatter(buys["ts"], buys["price"] * tick, marker="^", s=22, color=BLUE, edgecolor=SURFACE, lw=0.6, label="buy fill", zorder=3)
    ax_p.scatter(sells["ts"], sells["price"] * tick, marker="v", s=22, color=ORANGE, edgecolor=SURFACE, lw=0.6, label="sell fill", zorder=3)
    inside = (res.mid_t >= t0) & (res.mid_t <= t1)
    if inside.any():
        lo, hi = res.mid[inside].min() * tick, res.mid[inside].max() * tick
        pad = max((hi - lo) * 0.15, 10 * tick)
        ax_p.set_ylim(lo - pad, hi + pad)
    _style(ax_p, f"{res.strategy} maker, seed {res.seed}: price, quotes and fills", ylabel="price")
    _legend(ax_p, ncol=3, loc="upper left")

    limit = res.cfg.maker.max_inventory
    ax_i.step(s["t"], s["inventory"], where="post", color=BLUE, lw=1.2, label="inventory")
    ax_i.axhline(limit, color=INK2, lw=0.8, ls="--", label="inventory limit")
    ax_i.axhline(-limit, color=INK2, lw=0.8, ls="--")
    ax_i.axhline(0, color=GRID, lw=1.0)
    _style(ax_i, "inventory", ylabel="shares")
    _legend(ax_i, loc="upper left")

    att = res.attribution()
    ax_e.plot(s["t"], s["equity"], color=BLUE, lw=1.4, label="mark-to-mid equity")
    ax_e.axhline(0, color=GRID, lw=1.0)
    _style(ax_e, "equity", xlabel="seconds", ylabel="price units")
    ax_e.text(
        0.99,
        0.04,
        f"spread capture {att['spread_capture']:+.2f}   inventory {att['inventory']:+.2f}   fees {-att['fees']:+.2f}   total {att['total']:+.2f}",
        transform=ax_e.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
        color=INK2,
    )
    _legend(ax_e, loc="upper left")
    ax_e.set_xlim(t0, t1)
    return _save(fig, path)


def plot_book_heatmap(res: SimResult, path: Path, window: tuple[float, float] = (0.0, 300.0)) -> Path:
    """Depth of the book over time (bids blue, asks orange), with the mid and the quotes of the maker."""
    if not res.samples:
        raise ValueError("no depth samples: run the session with sample_interval > 0")
    tick = res.cfg.market.tick
    samples = [smp for smp in res.samples if window[0] <= smp[0] <= window[1]]
    times = np.array([smp[0] for smp in samples])
    p_lo = min(int(smp[2][0]) for smp in samples)
    p_hi = max(int(smp[2][-1]) for smp in samples)
    prices = np.arange(p_lo, p_hi + 1)
    bids = np.zeros((len(prices), len(samples)))
    asks = np.zeros_like(bids)
    for j, (_, _, px, b, a) in enumerate(samples):
        rows = px - p_lo
        bids[rows, j] = b
        asks[rows, j] = a
    vmax = float(np.percentile(np.concatenate([bids[bids > 0], asks[asks > 0]]), 97)) if (bids > 0).any() else 1.0
    dt = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    extent = (times[0] - dt / 2, times[-1] + dt / 2, (p_lo - 0.5) * tick, (p_hi + 0.5) * tick)

    fig, ax = _figure(height=5.2)
    im_b = ax.imshow(np.ma.masked_where(bids <= 0, bids), cmap=LinearSegmentedColormap.from_list("bids", BLUE_RAMP), extent=extent, aspect="auto", origin="lower", interpolation="nearest", vmin=0, vmax=vmax)
    im_a = ax.imshow(np.ma.masked_where(asks <= 0, asks), cmap=LinearSegmentedColormap.from_list("asks", ORANGE_RAMP), extent=extent, aspect="auto", origin="lower", interpolation="nearest", vmin=0, vmax=vmax)
    inside = (res.mid_t >= window[0]) & (res.mid_t <= window[1])
    ax.step(res.mid_t[inside], res.mid[inside] * tick, where="post", color=INK, lw=1.0, label="mid")
    s = res.series[(res.series["t"] >= window[0]) & (res.series["t"] <= window[1])]
    ax.plot(s["t"], s["bid"] * tick, color=INK2, lw=0.9, ls="--", label="maker bid / ask")
    ax.plot(s["t"], s["ask"] * tick, color=INK2, lw=0.9, ls="--")
    ax.set_xlim(extent[0], extent[1])
    _style(ax, f"the book around the mid, {res.strategy} maker, seed {res.seed}", xlabel="seconds", ylabel="price")
    ax.grid(False)
    cb_b = fig.colorbar(im_b, ax=ax, fraction=0.03, pad=0.01)
    cb_a = fig.colorbar(im_a, ax=ax, fraction=0.03, pad=0.04)
    cb_b.set_label("bid depth, shares", color=INK2, fontsize=8)
    cb_a.set_label("ask depth, shares", color=INK2, fontsize=8)
    for cb in (cb_b, cb_a):
        cb.ax.tick_params(colors=INK2, labelsize=8)
        cb.outline.set_edgecolor(GRID)
    _legend(ax, loc="upper left")
    return _save(fig, path)


def plot_naive_vs_as(naive: SimResult, avst: SimResult, path: Path) -> Path:
    """The same seed under the two makers: inventory and equity side by side."""
    fig, (ax_i, ax_e) = _figure(2, 1, height=6.5, sharex=True)
    limit = avst.cfg.maker.max_inventory
    for res, colour, label in ((naive, ORANGE, "naive"), (avst, BLUE, "Avellaneda-Stoikov")):
        ax_i.step(res.series["t"], res.series["inventory"], where="post", color=colour, lw=1.2, label=label)
        ax_e.plot(res.series["t"], res.series["equity"], color=colour, lw=1.4, label=label)
    ax_i.axhline(limit, color=INK2, lw=0.8, ls="--", label="inventory limit")
    ax_i.axhline(-limit, color=INK2, lw=0.8, ls="--")
    ax_i.axhline(0, color=GRID, lw=1.0)
    ax_e.axhline(0, color=GRID, lw=1.0)
    _style(ax_i, f"inventory, seed {avst.seed}", ylabel="shares")
    _style(ax_e, "mark-to-mid equity", xlabel="seconds", ylabel="price units")
    _legend(ax_i, loc="upper left", ncol=3)
    _legend(ax_e, loc="upper left")
    return _save(fig, path)


# ---------------------------------------------------------------- over seeds
def plot_attribution(agg: pd.DataFrame, path: Path, title: str = "PnL attribution, mean over seeds") -> Path:
    """Spread capture, inventory PnL and fees per variant, with the total as a marker."""
    names = list(agg.index)
    x = np.arange(len(names))
    width = 0.26
    fig, ax = _figure(height=4.4)
    parts = [("spread_capture", "spread capture", BLUE), ("inventory_pnl", "inventory", ORANGE), ("fees", "fees", AQUA)]
    for i, (col, label, colour) in enumerate(parts):
        values = agg[col].to_numpy(dtype=float)
        if col == "fees":
            values = -values
        ax.bar(x + (i - 1) * width, values, width=width - 0.03, color=colour, edgecolor=SURFACE, linewidth=1.5, label=label)
    ax.scatter(x, agg["pnl_mean"], marker="D", s=36, color=INK, zorder=4, label="total PnL")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0 if len(names) <= 4 else 25, ha="center" if len(names) <= 4 else "right", color=INK2, fontsize=9)
    _style(ax, title, ylabel="price units per session")
    _legend(ax, ncol=4, loc="upper left")
    return _save(fig, path)


def plot_frontier(agg_as: pd.DataFrame, agg_naive: pd.DataFrame, path: Path) -> Path:
    """Mean PnL against inventory risk: the A-S maker over its risk aversion, the naive maker over its spread."""
    fig, ax = _figure(height=4.8)
    # label placements alternate so that neighbouring points do not write over each other; the naive points
    # cluster at the right edge, so their odd labels go left-below, the A-S labels stay to the right of the curve
    series = (
        (agg_naive, ORANGE, "naive, by half-spread (ticks)", "h=", (((6, 6), "left"), ((-6, -12), "right"))),
        (agg_as, BLUE, "Avellaneda-Stoikov, by gamma", "γ=", (((6, 6), "left"), ((6, -12), "left"))),
    )
    for agg, colour, label, prefix, placements in series:
        err = agg["pnl_std"] / np.sqrt(agg["seeds"])
        ax.errorbar(agg["inv_std"], agg["pnl_mean"], yerr=err, color=colour, lw=1.2, marker="o", ms=5, capsize=2, label=label)
        for i, (value, xv, yv) in enumerate(zip(agg["value"], agg["inv_std"], agg["pnl_mean"], strict=True)):
            offset, align = placements[i % 2]
            ax.annotate(f"{prefix}{value:g}", (xv, yv), textcoords="offset points", xytext=offset, ha=align, fontsize=8, color=INK2)
    ax.axhline(0, color=GRID, lw=1.0)
    _style(ax, "PnL against inventory risk, mean over seeds (bars: standard error)", xlabel="standard deviation of inventory, shares", ylabel="mean PnL per session, price units")
    _legend(ax, loc="lower right")
    return _save(fig, path)


def plot_markouts(agg: pd.DataFrame, horizons: tuple[float, ...], tick: float, path: Path) -> Path:
    """Per variant and horizon: the markout against the pre-trade mid, its drift part (after the mechanical impact
    of the order that filled the maker), and what the fill was worth. Below zero is adverse."""
    fig, (ax_m, ax_d, ax_r) = _figure(1, 3, width=14.0, height=4.4)
    for i, name in enumerate(agg.index):
        colour = VARIANT_COLOURS[i % len(VARIANT_COLOURS)]
        for ax, prefix in ((ax_m, "markout"), (ax_d, "drift"), (ax_r, "realised")):
            ax.plot(horizons, [agg.loc[name, f"{prefix}_{h:g}"] / tick for h in horizons], color=colour, lw=1.6, marker="o", ms=5, label=name)
    for ax, title, ylabel in (
        (ax_m, "markout: mid(t+h) against the pre-trade mid", "ticks per share, from the point of view of the maker"),
        (ax_d, "drift: mid(t+h) against the post-trade mid", "ticks per share"),
        (ax_r, "realised: mid(t+h) against the fill price", "ticks per share"),
    ):
        ax.axhline(0, color=INK2, lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks(list(horizons))
        ax.set_xticklabels([f"{h:g} s" for h in horizons])
        _style(ax, title, xlabel="horizon after the fill", ylabel=ylabel)
    _legend(ax_m, loc="lower left")
    return _save(fig, path)


def plot_intensity(fit: IntensityFit, path: Path) -> Path:
    """Fill intensity against the distance of the quote from the mid, with the exponential fit."""
    t = fit.table[fit.table["exposure_s"] > 0]
    fig, ax = _figure(height=4.4)
    ax.scatter(t["distance_ticks"], t["intensity"], color=BLUE, s=28, edgecolor=SURFACE, lw=0.6, zorder=3, label="measured (fills / seconds quoted)")
    xs = np.linspace(0, t["distance_ticks"].max() + 0.5, 100)
    ax.plot(xs, fit.A * np.exp(-fit.k_ticks * xs), color=ORANGE, lw=1.6, label=f"fit  λ(δ) = {fit.A:.2f}·exp(−{fit.k:.1f} δ)   R² {fit.r2:.2f}")
    ax.set_yscale("log")
    _style(ax, "fill intensity against distance from the mid (probe maker)", xlabel="distance of the quote from the mid, ticks", ylabel="fills per second (log scale)")
    _legend(ax, loc="upper right")
    return _save(fig, path)


def plot_signature(signature: pd.Series, fair_vol: float, path: Path) -> Path:
    """Realised volatility of the mid against the sampling interval, with the fair-value volatility as reference."""
    fig, ax = _figure(height=4.0)
    ax.plot(signature.index, signature.values, color=BLUE, lw=1.6, marker="o", ms=5, label="realised volatility of the mid")
    ax.axhline(fair_vol, color=AQUA, lw=1.2, ls="--", label="volatility of the fair value (diffusion and news)")
    ax.set_xscale("log")
    ax.set_xticks(list(signature.index))
    ax.set_xticklabels([f"{dt:g} s" for dt in signature.index])
    ax.set_ylim(0, max(signature.max(), fair_vol) * 1.15)
    _style(ax, "volatility signature: the mid is noisier than the fair value at short horizons", xlabel="sampling interval", ylabel="ticks per sqrt(second)")
    _legend(ax, loc="upper right")
    return _save(fig, path)
