"""
common.py — shared constants and helpers for the game-analysis pipeline
(analyze_games.py, stats_games.py, plot_analysis.py).
"""

import csv

import numpy as np
import matplotlib as mpl

# thresholds in centipawns (Lichess conventions)
INACCURACY, MISTAKE, BLUNDER = 50, 100, 300
CP_CAP = 1000          # clamp evals so one lost game doesn't dominate the mean
MATE_CP = 1000         # mate scores mapped to +/- this

# plot palette: h8 = blue, h1 = orange (Okabe-Ito, colourblind-safe)
C8, C1 = "#0072B2", "#D55E00"
INK, GRID = "#222222", "#D9D9D9"

PAIR_LABEL = {
    "A_h8-30k_vs_h1-30k": "A: h8@30k vs h1@30k\n(matched steps)",
    "B_h8-30k_vs_h1-60k": "B: h8@30k vs h1@60k\n(peak vs peak)",
    "C_h8-60k_vs_h1-60k": "C: h8@60k vs h1@60k\n(matched steps)",
}

# game-phase ply bins: (lo, hi, plot label, stats label)
PHASES = [(0, 20, "opening\n0–20", "opening 0-20"),
          (20, 60, "early mid\n20–60", "early mid 20-60"),
          (60, 100, "late mid\n60–100", "late mid 60-100"),
          (100, 10**9, "endgame\n100+", "endgame 100+")]


def apply_plot_style():
    mpl.rcParams.update({
        "figure.dpi": 150, "savefig.dpi": 150,
        "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
        "axes.edgecolor": INK, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7,
        "axes.axisbelow": True,
    })


def save_fig(fig, out_dir, name):
    """Write a figure as both pdf and png and close it."""
    import matplotlib.pyplot as plt
    fig.savefig(out_dir / f"{name}.pdf")
    fig.savefig(out_dir / f"{name}.png")
    plt.close(fig)


def is_h8(name):
    return name.startswith("h8")


def split_players(rows, pairing):
    """Return (h8_name, h1_name) for one pairing."""
    names = sorted({r["player"] for r in rows if r["pairing"] == pairing})
    h8 = next(n for n in names if is_h8(n))
    h1 = next(n for n in names if not is_h8(n))
    return h8, h1


def load_positions(path, skip_book, eval_band=None):
    """Read positions.csv into dicts with ply/game/cp_loss/eval_before coerced."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if skip_book and r["in_book"].lower() == "true":
                continue
            r["ply"] = int(r["ply"]); r["game"] = int(r["game"])
            r["cp_loss"] = float(r["cp_loss"])
            r["eval_before"] = float(r["eval_before"])
            if eval_band is not None and abs(r["eval_before"]) > eval_band:
                continue
            rows.append(r)
    return rows


def bootstrap_ci(x, rng, n_boot=10000, alpha=0.05, stat=np.mean):
    """Percentile bootstrap CI. Pass the caller's RNG so runs stay reproducible."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float(stat(x)) if len(x) else 0.0, 0.0, 0.0
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    boots = stat(x[idx], axis=1)
    return float(stat(x)), float(np.percentile(boots, 100 * alpha / 2)), \
        float(np.percentile(boots, 100 * (1 - alpha / 2)))
