import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

_ap = argparse.ArgumentParser(description="Build the thesis figures from results/.")
_ap.add_argument("--results", type=Path, default=Path("results"))
_ap.add_argument("--out", type=Path, default=Path("figures"))
_args = _ap.parse_args()
RES, OUT = _args.results, _args.out
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
# Liberation Serif is a TrueType Times clone, so it embeds cleanly as Type 42.
# Matplotlib falls back to its default serif font when it is not installed.
for f in Path("/usr/share/fonts/truetype/liberation").glob("LiberationSerif-*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.edgecolor": "#8a8984",
    "axes.linewidth": 0.6,
    "axes.labelcolor": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": "#e6e5e0",
    "grid.linewidth": 0.6,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})
TEXT, MUTED = "#0b0b0b", "#52514e"
H8, H1, REP = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3, validated
WIDTH = 6.3                                       # \textwidth at A4, 2.5 cm margins


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=200)
    plt.close(fig)


def read_rows(path):
    return [[float(x) for x in r] for r in csv.reader(open(path)) if r]


# ---------------------------------------------------------------- data
logs = RES / "training_logs"
ev = {m: np.array(read_rows(logs / d / "eval_log.csv")) for m, d in (("h1", "h1"), ("h8", "h8"), ("h8rep", "h8_rep"))}
tr = {m: np.array(read_rows(logs / d / "train_log.csv")) for m, d in (("h1", "h1"), ("h8", "h8"), ("h8rep", "h8_rep"))}
stats = json.load(open(RES / "dataset_stats.json"))


def read_pgn_headers(path, focus):
    """Score of the focus player and opening ID for every game in a PGN file."""
    games, tags = [], {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r'\[(\w+) "(.*)"\]', line.strip())
        if m:
            tags[m.group(1)] = m.group(2)
        elif tags and line.strip() and not line.startswith("["):
            if "Result" in tags and "OpeningGameId" in tags:
                white = tags["White"].startswith(focus + "_") or tags["White"].startswith(focus + "@")
                res = tags["Result"]
                sc = 0.5 if res == "1/2-1/2" else float((res == "1-0") == white)
                games.append({"op": tags["OpeningGameId"], "score": sc})
                tags = {}
    return games


base = RES / "matches" / "original"
rep = RES / "matches" / "h8_rep"
G = {}
for cond in ("off", "on"):
    for pair in ("A_h8-30k_vs_h1-30k", "B_h8-30k_vs_h1-60k", "C_h8-60k_vs_h1-60k"):
        G[f"{cond}|{pair}"] = read_pgn_headers(base / cond / f"{pair}.pgn", "h8")
for name in ("rep_vs_h8", "rep_vs_h1_filter_off", "rep_vs_h1_filter_on"):
    G[f"rep|{name}"] = read_pgn_headers(rep / f"{name}.pgn", "h8_rep")
assert all(len(g) == 100 for g in G.values()), {k: len(g) for k, g in G.items()}

summary = {}  # every value that goes into a figure, written out for checking


# ---------------------------------------------------------------- figure 1
def rolling_median(y, k=21):
    pad = k // 2
    yp = np.pad(y, pad, mode="edge")
    return np.array([np.median(yp[i:i + k]) for i in range(len(y))])


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(WIDTH, 4.9), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1], "hspace": 0.12})
sel = {"h8": 30000, "h1": 60000, "h8rep": 90000}
LAB = {"h8": "h8", "h1": "h1", "h8rep": "h8+rep"}
for m, col, mk in (("h8", H8, "o"), ("h1", H1, "s"), ("h8rep", REP, "^")):
    s, acc = ev[m][:, 0], 100 * ev[m][:, 2]
    ax1.plot(s / 1000, acc, color=col, lw=1.6, marker=mk, ms=3.8, zorder=3, label=LAB[m])
    i = int(np.where(s == sel[m])[0][0])
    ax1.plot(s[i] / 1000, acc[i], marker=mk, ms=9.5, mfc="none", mec=col, mew=1.3, zorder=4)
    summary[f"val_{m}"] = {int(a): round(b, 2) for a, b in zip(s, acc)}
for m, dx, dy, ha in (("h8", 0, 0.5, "center"), ("h8rep", 0, 0.5, "center"), ("h1", 0, -0.62, "center")):
    i = int(np.where(ev[m][:, 0] == sel[m])[0][0])
    x, y = sel[m] / 1000, 100 * ev[m][i, 2]
    ax1.text(x + dx, y + dy, f"{y:.2f}% ({sel[m] // 1000}k)", fontsize=8.5, color=TEXT,
             ha=ha, va="center")
ax1.set_ylabel("Validation accuracy (%)")
ax1.set_ylim(36.2, 44.3)
ax1.legend(loc="lower right", ncol=3, handlelength=2.2, columnspacing=1.2)
ax1.text(0.0, 1.02, "(a) Legal top-1 accuracy on the fixed validation sample",
         transform=ax1.transAxes, fontsize=10, color=TEXT, ha="left", va="bottom")

for m, col in (("h8", H8), ("h1", H1), ("h8rep", REP)):
    s, loss = tr[m][:, 0], tr[m][:, 1]
    ax2.plot(s / 1000, loss, color=col, lw=0.4, alpha=0.35, zorder=2)
    ax2.plot(s / 1000, rolling_median(loss), color=col, lw=1.6, zorder=3, label=LAB[m])
    summary[f"train_{m}_median_at"] = {k: round(float(rolling_median(loss)[np.searchsorted(s, k)]), 3)
                                       for k in (30000, 60000, 80000, 100000, 160000, 200000)
                                       if k <= s[-1]}
spike = tr["h8"][np.argmax(tr["h8"][:, 1])]
summary["h8_spike"] = {"step": int(spike[0]), "loss": round(float(spike[1]), 1)}
ax2.annotate(f"h8 spike at step {int(spike[0]):,}\n(loss {spike[1]:,.0f}, off scale)",
             xy=(spike[0] / 1000, 3.85), xytext=(22, 3.45), fontsize=9, color=TEXT,
             ha="left", va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
ax2.set_ylim(0.8, 3.9)
ax2.set_ylabel("Training loss")
ax2.set_xlabel("Training step (thousands)")
ax2.text(0.0, 1.02, "(b) Training loss (policy plus value)",
         transform=ax2.transAxes, fontsize=10, color=TEXT, ha="left", va="bottom")

for ax in (ax1, ax2):
    for x in (160, 180):
        ax.axvline(x, color="#b5b4ae", lw=0.8, zorder=1)
    ax.set_xlim(0, 204)
    ax.set_xticks(range(0, 201, 20))
ax2.text(161.5, 3.8, "learning-rate\ndrops of\nh1 and h8", fontsize=8, color=MUTED, ha="left", va="top")
save(fig, "fig_training_curves")


# ---------------------------------------------------------------- figure 2
def score(key):
    g = G[key]
    return 100 * sum(x["score"] for x in g) / len(g)


def boot_ci(key):
    """Opening-ID bootstrap, 10,000 draws, seed 0, as in the thesis."""
    per = {}
    for x in G[key]:
        p = per.setdefault(x["op"], [0.0, 0])
        p[0] += x["score"]
        p[1] += 1
    ids = sorted(per)
    pts = np.array([per[i][0] for i in ids])
    n = np.array([per[i][1] for i in ids], float)
    d = np.random.default_rng(0).integers(0, len(ids), size=(10000, len(ids)))
    b = 100 * pts[d].sum(1) / n[d].sum(1)
    return np.percentile(b, [2.5, 97.5])


rows = [  # label, key, colour, filter state, bold
    ("A, filter off", "off|A_h8-30k_vs_h1-30k", H8, "off", False),
    ("A, filter on", "on|A_h8-30k_vs_h1-30k", H8, "on", False),
    ("B, filter off", "off|B_h8-30k_vs_h1-60k", H8, "off", True),
    ("B, filter on", "on|B_h8-30k_vs_h1-60k", H8, "on", True),
    ("C, filter off", "off|C_h8-60k_vs_h1-60k", H8, "off", False),
    ("C, filter on", "on|C_h8-60k_vs_h1-60k", H8, "on", False),
    ("against h8@30k", "rep|rep_vs_h8", REP, "none", False),
    ("against h1@60k, filter off", "rep|rep_vs_h1_filter_off", REP, "off", False),
    ("against h1@60k, filter on", "rep|rep_vs_h1_filter_on", REP, "on", False),
]
ypos = [9.6, 8.8, 7.6, 6.8, 5.6, 4.8, 2.6, 1.6, 0.8]
fig, ax = plt.subplots(figsize=(WIDTH, 3.9))
summary["scores"] = {}
for (lab, key, col, state, bold), y in zip(rows, ypos):
    s, (lo, hi) = score(key), boot_ci(key)
    summary["scores"][lab + " | " + key] = [round(s, 2), round(lo, 2), round(hi, 2)]
    ax.plot([lo, hi], [y, y], color=col, lw=1.6, solid_capstyle="round", zorder=2)
    mk = "D" if state == "none" else "o"
    face = "white" if state == "on" else col
    ax.plot(s, y, marker=mk, ms=7 if mk == "o" else 6, mfc=face, mec=col, mew=1.5, zorder=3)
ax.axvline(50, color="#8a8984", lw=0.9, zorder=1)
ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows])
for t, r in zip(ax.get_yticklabels(), rows):
    t.set_color(TEXT)
    if r[4]:
        t.set_fontweight("bold")
ax.set_ylim(0.2, 10.9)
ax.set_xlim(35, 65)
ax.set_xticks(range(35, 66, 5))
ax.grid(axis="y", visible=False)
ax.set_xlabel("Score (%) with 95% confidence interval")
ax.text(35.4, 10.55, "h8's score against h1 (original pairings, B primary)", fontsize=9.5,
        color=TEXT, ha="left", va="center", style="italic",
        bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
ax.text(35.4, 3.45, "h8+rep@90k's score (supplementary)", fontsize=9.5,
        color=TEXT, ha="left", va="center", style="italic",
        bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
ax.text(50.4, 0.35, "50% = even", fontsize=8, color=MUTED, ha="left", va="center")
from matplotlib.lines import Line2D
handles = [Line2D([], [], marker="o", ls="", ms=7, mfc=MUTED, mec=MUTED, label="h1 filter off"),
           Line2D([], [], marker="o", ls="", ms=7, mfc="white", mec=MUTED, mew=1.5, label="h1 filter on"),
           Line2D([], [], marker="D", ls="", ms=6, mfc=MUTED, mec=MUTED, label="no filter (against h8)")]
ax.legend(handles=handles, loc="lower right", ncol=1, handletextpad=0.4)
save(fig, "fig_match_scores")


# ---------------------------------------------------------------- figure A1
hist = stats["game_length_hist"]
counts = np.array(hist["counts"])
edges = hist["edges"]
fig, (axa, axb) = plt.subplots(1, 2, figsize=(WIDTH, 2.6),
                               gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.42})
x = np.arange(len(counts))
axa.bar(x, counts / 1000, width=0.86, color=H8, zorder=2)
axa.set_xticks(x[::4] - 0.5)
axa.set_xticklabels([str(e) for e in edges[:-1:4]])
axa.set_xlim(-0.8, len(counts) - 0.2)
axa.text(len(counts) - 1, counts[-1] / 1000 + 0.35, "400+", fontsize=8, color=MUTED, ha="center")
med = stats["game_length_summary"]["median"]
mx = (med - 0) / 20 - 0.5
axa.axvline(mx, color=TEXT, lw=0.9, zorder=3)
axa.text(mx + 0.3, 9.6, f"median {med:.0f}", fontsize=8.5, color=TEXT, ha="left", va="top")
axa.set_xlabel("Game length (plies)")
axa.set_ylabel("Games (thousands)")
axa.grid(axis="x", visible=False)
axa.text(0.0, 1.03, "(a) Game length", transform=axa.transAxes, ha="left", va="bottom")
summary["game_length"] = {"total_games": int(counts.sum()), "median": med,
                          "mean": round(stats["game_length_summary"]["mean"], 1)}

names = [("train", "Train"), ("val", "Validation"), ("test", "Test")]
cols = [H8, H1, REP]
labels = ["White wins", "Draw", "Black wins"]
summary["outcome_share_positions"] = {}
for j, (k, nm) in enumerate(names):
    r = stats["result_by_split_ply"][k]
    tot = sum(r.values())
    left = 0.0
    shares = [100 * r[c] / tot for c in ("0", "1", "2")]
    summary["outcome_share_positions"][nm] = [round(v, 1) for v in shares]
    for i, (sh, c) in enumerate(zip(shares, cols)):
        axb.barh(2 - j, sh - 0.6, left=left + 0.3, height=0.62, color=c, zorder=2)
        axb.text(left + sh / 2, 2 - j, f"{sh:.0f}", ha="center", va="center", fontsize=8.5,
                 color="white" if c == H8 else TEXT)
        left += sh
axb.set_yticks([2, 1, 0])
axb.set_yticklabels([n for _, n in names])
axb.set_xlim(0, 100)
axb.set_xlabel("Share of positions (%)")
axb.grid(axis="y", visible=False)
axb.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=c) for c in cols], labels=labels,
           loc="upper center", bbox_to_anchor=(0.5, -0.24), ncol=3, handlelength=1.0,
           columnspacing=0.9, handletextpad=0.4)
axb.text(0.0, 1.03, "(b) Game outcome by split", transform=axb.transAxes, ha="left", va="bottom")
save(fig, "fig_dataset")

json.dump(summary, open(OUT / "figure_values.json", "w"), indent=1)
print(f"wrote figures and figure_values.json to {OUT}")
