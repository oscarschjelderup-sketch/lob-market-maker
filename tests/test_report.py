"""The figure pipeline end to end on a tiny configuration: every file is written and the tables are consistent."""

from __future__ import annotations

import json

import pandas as pd

from lobmm.config import SimConfig
from lobmm.report import make_figures, markdown_table

EXPECTED = [
    "intensity_fit.png",
    "volatility_signature.png",
    "markout_fit.png",
    "session_as.png",
    "session_naive.png",
    "book_heatmap.png",
    "naive_vs_as.png",
    "pnl_attribution.png",
    "risk_return_frontier.png",
    "markouts.png",
    "results.json",
    "results.md",
]


def test_make_figures_writes_everything(tmp_path):
    messages = []
    make_figures(tmp_path, seeds=[0, 1], horizon=600.0, cfg=SimConfig(), calibration_seeds=(0, 1), log=messages.append)
    for name in EXPECTED:
        assert (tmp_path / name).exists() and (tmp_path / name).stat().st_size > 0, name
    assert any("done" in m for m in messages)
    loaded = json.loads((tmp_path / "results.json").read_text(encoding="utf-8"))
    assert loaded["calibration"]["k"] > 0 and loaded["config"]["maker"]["k"] == loaded["calibration"]["k"]
    assert {r["variant"] for r in loaded["compare"]} == {"naive h=3", "A-S gamma=0.003", "A-S+adverse gamma=0.003"}
    assert len(loaded["sweep_gamma"]) == 5 and len(loaded["sweep_gamma_adverse"]) == 5 and len(loaded["sweep_half_spread"]) == 5
    assert len(loaded["sweep_adverse"]) == 6 and loaded["sweep_adverse"][0]["value"] == 0.0
    assert len(loaded["compare_per_seed"]) == 6
    assert loaded["config"]["maker"]["adverse"] == loaded["calibration"]["adverse_term"] > 0
    md = (tmp_path / "results.md").read_text(encoding="utf-8")
    assert "## Naive against Avellaneda-Stoikov" in md and "| A-S gamma=0.003 |" in md and "| A-S+adverse gamma=0.003 |" in md
    assert "### Adverse cost by distance" in md and "## Adverse-term sweep" in md


def test_markdown_table_formats_floats_and_blanks_nan():
    df = pd.DataFrame({"a": [1.23456, float("nan")], "b": ["x", "y"]}, index=pd.Index(["r1", "r2"], name="row"))
    text = markdown_table(df, ".2f")
    assert text.splitlines()[0] == "| row | a | b |"
    assert "| r1 | 1.23 | x |" in text and "| r2 |  | y |" in text
