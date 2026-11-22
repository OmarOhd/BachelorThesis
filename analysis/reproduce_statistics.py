import argparse
import collections
import csv
import json
from pathlib import Path

import chess.pgn
import numpy as np

N_BOOT = 10_000
BLUNDER_CP = 300
ORIGINAL = ("A_h8-30k_vs_h1-30k", "B_h8-30k_vs_h1-60k", "C_h8-60k_vs_h1-60k")
REP = ("rep_vs_h8", "rep_vs_h1_filter_off", "rep_vs_h1_filter_on")
OUT = {}


def ci(samples):
    lo, hi = np.percentile(samples, [2.5, 97.5], method="linear")
    return [float(lo), float(hi)]


def show(title, rows, header):
    print(f"\n{title}")
    print("  " + " | ".join(header))
    for r in rows:
        print("  " + " | ".join(str(x) for x in r))


def fmt_ci(c, nd=2):
    return f"[{c[0]:.{nd}f}, {c[1]:.{nd}f}]"


# ------------------------------------------------------------------ data set
def dataset(res):
    s = json.load(open(res / "dataset_stats.json"))
    rows = [(k, f"{s['split_sizes_games'][k]:,}", f"{s['split_sizes_plies'][k]:,}")
            for k in ("train", "val", "test")]
    rows.append(("total", f"{sum(s['split_sizes_games'].values()):,}", f"{s['n_plies']:,}"))
    show("Dataset partitions", rows, ("split", "games", "position-move pairs"))
    OUT["dataset"] = {"games": s["split_sizes_games"], "positions": s["split_sizes_plies"]}


# ------------------------------------------------------------ validation logs
def validation(res):
    out, rows = {}, []
    for m in ("h1", "h8", "h8_rep"):
        log = [(int(r[0]), 100 * float(r[2])) for r in csv.reader(open(res / f"training_logs/{m}/eval_log.csv")) if r]
        best = max(log, key=lambda x: x[1])
        out[m] = {"selected_step": best[0], "selected_acc": round(best[1], 2),
                  "final_step": log[-1][0], "final_acc": round(log[-1][1], 2)}
        rows.append((m, f"{best[0]:,}", f"{best[1]:.2f}", f"{log[-1][0]:,}", f"{log[-1][1]:.2f}"))
    show("Validation accuracy, selected and final checkpoints (%)", rows,
         ("model", "selected step", "accuracy", "final step", "accuracy"))
    OUT["validation"] = out


# -------------------------------------------------------- test-set prediction
def prediction(res):
    files = {"h1@60k": "h1_60k.csv", "h8@30k": "h8_30k.csv", "h8+rep@90k": "h8_rep_90k.csv"}
    data = {}
    for name, f in files.items():
        data[name] = {r["game_id"]: (int(r["n"]), int(r["correct_legal"]), int(r["correct_raw"]),
                                     int(r["raw_illegal"])) for r in csv.DictReader(open(res / "test_eval" / f))}
    ids = sorted(data["h1@60k"])
    assert all(sorted(d) == ids for d in data.values())
    n = np.array([data["h1@60k"][g][0] for g in ids])
    assert all((np.array([d[g][0] for g in ids]) == n).all() for d in data.values())
    arr = {k: np.array([[d[g][j] for g in ids] for j in (1, 2, 3)]) for k, d in data.items()}
    rows, out = [], {"games": len(ids), "positions": int(n.sum())}
    for k, a in arr.items():
        legal, raw, ill = (100 * a[j].sum() / n.sum() for j in range(3))
        out[k] = {"legal_top1": round(legal, 2), "raw_top1": round(raw, 2), "raw_illegal": round(ill, 2)}
        rows.append((k, f"{legal:.2f}", f"{raw:.2f}", f"{ill:.2f}"))
    show(f"Test prediction, {len(ids):,} games, {n.sum():,} positions (%)", rows,
         ("checkpoint", "legal top-1", "raw top-1", "raw illegal"))
    draws = np.random.default_rng(0).integers(0, len(ids), size=(N_BOOT, len(ids)))
    nd = n[draws].sum(1)
    rows = []
    for a, b in (("h8@30k", "h1@60k"), ("h8+rep@90k", "h1@60k"), ("h8+rep@90k", "h8@30k")):
        x, y = arr[a][0], arr[b][0]
        d = 100 * (x.sum() - y.sum()) / n.sum()
        c = ci(100 * (x[draws].sum(1) - y[draws].sum(1)) / nd)
        out[f"{a} - {b}"] = {"diff_pp": round(d, 2), "ci95": [round(v, 2) for v in c]}
        rows.append((f"{a} - {b}", f"{d:+.2f}", fmt_ci(c)))
    show("Paired legal top-1 differences (pp), game bootstrap", rows, ("comparison", "difference", "95% CI"))
    OUT["prediction"] = out


# ------------------------------------------------------------------ matches
def load_pgn(path, focus):
    games = []
    with open(path, encoding="utf-8") as f:
        while (g := chess.pgn.read_game(f)) is not None:
            h = g.headers
            moves = list(g.mainline_moves())
            board = g.board()
            for mv in moves:
                board.push(mv)
            focus_white = h["White"].split("@")[0] == focus
            res = h["Result"]
            score = 0.5 if res == "1/2-1/2" else float((res == "1-0") == focus_white)
            n_open = int(h.get("OpeningPlies", 8))
            # the side that made the final move
            last_white = len(moves) % 2 == 1
            last_name = h["White"] if last_white else h["Black"]
            h1_is_white = h["White"].startswith("h1_")
            h1_moves = sum(1 for i in range(n_open, len(moves)) if (i % 2 == 0) == h1_is_white) \
                if ("h1_" in h["White"] or "h1_" in h["Black"]) else 0
            h1_interv = int(h["WhiteRepetitionInterventions"] if h1_is_white else h["BlackRepetitionInterventions"]) \
                if ("h1_" in h["White"] or "h1_" in h["Black"]) else 0
            games.append(dict(op=h["OpeningGameId"], score=score, res=res, focus_white=focus_white,
                              three=h.get("Termination") == "threefold", last=last_name,
                              h1_moves=h1_moves, h1_interv=h1_interv,
                              line=" ".join(m.uci() for m in moves[:n_open])))
    return games


def matches(res):
    G = {}
    for cond in ("off", "on"):
        for p in ORIGINAL:
            G[f"{p[0]} {cond}"] = load_pgn(res / "matches/original" / cond / f"{p}.pgn", "h8_masked")
    for p in REP:
        G[p] = load_pgn(res / "matches/h8_rep" / f"{p}.pgn",
                        "h8_rep_lr50k_20261113T181553407907Z")
    assert all(len(g) == 100 for g in G.values())
    ref = {g["op"]: g["line"] for g in G["B off"]}
    assert all({g["op"]: g["line"] for g in gs} == ref for gs in G.values()), "openings differ"
    ids = sorted(ref)
    lines = sorted(set(ref.values()))

    def per_op(gs, key):
        d = collections.defaultdict(float)
        for g in gs:
            d[g["op"]] += g[key]
        return d

    def vec(cfg, key, unit):
        if key == "n":
            d = collections.Counter(g["op"] for g in G[cfg])
        else:
            d = per_op(G[cfg], key)
        if unit == "id":
            return np.array([d[i] for i in ids], float)
        return np.array([np.mean([d[i] for i in ids if ref[i] == L]) for L in lines])

    out = {"opening_ids": len(ids), "opening_lines": len(lines)}
    rows = []
    for cfg, gs in G.items():
        w = sum(g["score"] == 1 for g in gs); d = sum(g["score"] == 0.5 for g in gs)
        l = sum(g["score"] == 0 for g in gs)
        out[cfg] = {"W": w, "D": d, "L": l, "threefold": sum(g["three"] for g in gs)}
    for unit in ("id", "line"):
        k = len(ids) if unit == "id" else len(lines)
        draws = np.random.default_rng(0).integers(0, k, size=(N_BOOT, k))

        def rate(cfg, key):
            num, den = vec(cfg, key, unit), vec(cfg, "n", unit)
            return 100 * num.sum() / den.sum(), 100 * num[draws].sum(1) / den[draws].sum(1)

        tag = "" if unit == "id" else " (45 lines)"
        rows = []
        for cfg in G:
            p, b = rate(cfg, "score")
            out.setdefault(cfg, {})[f"score{tag}"] = [round(p, 2)] + [round(v, 2) for v in ci(b)]
            o = out[cfg]
            rows.append((cfg, f"{o['W']}/{o['D']}/{o['L']}", f"{p:.2f}", fmt_ci(ci(b))))
        show(f"Match scores, focus model's perspective{tag}", rows, ("configuration", "W/D/L", "score %", "95% CI"))
        rows = []
        pairs = [(f"{p} on", f"{p} off", p) for p in "ABC"] + [("rep_vs_h1_filter_on", "rep_vs_h1_filter_off", "h8+rep vs h1")]
        for on, off, lab in pairs:
            ps1, bs1 = rate(on, "score"); ps0, bs0 = rate(off, "score")
            pt1, bt1 = rate(on, "three"); pt0, bt0 = rate(off, "three")
            out[f"filter effect {lab}{tag}"] = {"score_diff": [round(ps1 - ps0, 2)] + [round(v, 2) for v in ci(bs1 - bs0)],
                                               "threefold_diff": [round(pt1 - pt0, 2)] + [round(v, 2) for v in ci(bt1 - bt0)]}
            rows.append((lab, f"{pt0:.2f} -> {pt1:.2f}", f"{pt1 - pt0:+.2f} {fmt_ci(ci(bt1 - bt0))}",
                         f"{ps1 - ps0:+.2f} {fmt_ci(ci(bs1 - bs0))}"))
        show(f"Filter on minus off{tag}", rows,
             ("comparison", "threefold per 100", "threefold change (pp)", "focus score change (pp)"))
        rows = []
        for f, b in (("off", "B off"), ("on", "B on")):
            r = f"rep_vs_h1_filter_{f}"
            p1, b1 = rate(r, "score"); p0, b0 = rate(b, "score")
            t1, c1 = rate(r, "three"); t0, c0 = rate(b, "three")
            out[f"h8+rep vs h8 against h1, filter {f}{tag}"] = {
                "score_diff": [round(p1 - p0, 2)] + [round(v, 2) for v in ci(b1 - b0)],
                "threefold_diff": [round(t1 - t0, 2)] + [round(v, 2) for v in ci(c1 - c0)]}
            rows.append((f"filter {f}", f"{p1 - p0:+.2f} {fmt_ci(ci(b1 - b0))}", f"{t1 - t0:+.2f} {fmt_ci(ci(c1 - c0))}"))
        show(f"h8+rep minus h8, same h1 opponent{tag}", rows, ("condition", "score (pp)", "threefold (pp)"))

    rows, total_three, by_opp = [], 0, 0
    for cfg in ("A on", "B on", "C on", "rep_vs_h1_filter_on"):
        gs = G[cfg]
        iv = sum(g["h1_interv"] for g in gs); mv = sum(g["h1_moves"] for g in gs)
        aff = sum(g["h1_interv"] > 0 for g in gs)
        three = [g for g in gs if g["three"]]
        opp = sum(not g["last"].startswith("h1_") for g in three)
        total_three += len(three); by_opp += opp
        out[f"interventions {cfg}"] = {"interventions": iv, "h1_moves": mv, "rate_pct": round(100 * iv / mv, 2),
                                       "affected_games": aff}
        rows.append((cfg, iv, f"{mv:,}", f"{100 * iv / mv:.2f}", aff, f"{opp} of {len(three)}"))
    show("Filter interventions (filter on)", rows,
         ("configuration", "interventions", "h1 moves", "rate %", "affected games", "threefolds completed by opponent"))
    out["threefolds completed by opponent, filter on"] = [by_opp, total_three]
    three = [g for g in G["rep_vs_h1_filter_off"] if g["three"]]
    own = sum(g["last"].startswith("h8_rep") for g in three)
    out["h8+rep completed own threefolds, filter off"] = [own, len(three)]
    print(f"\n  threefolds completed by the opponent with the filter on: {by_opp} of {total_three}")
    print(f"  h8+rep completed {own} of its {len(three)} threefolds against h1 with the filter off")
    print(f"  direct h8+rep vs h8 match: {out['rep_vs_h8']['threefold']} of 100 games ended in threefold")
    OUT["matches"] = out
    return ref, ids, lines


# ------------------------------------------------------------- move quality
def move_quality(res, ref, ids, lines):
    agg = collections.defaultdict(lambda: np.zeros(3))  # loss, moves, blunders
    for cond in ("off", "on"):
        for r in csv.DictReader(open(res / "matches/original/stockfish" / cond / "positions.csv")):
            if r["in_book"] == "True":
                continue
            role = "h8" if r["player"].startswith("h8_") else "h1"
            l = int(r["cp_loss"])
            agg[(f"{r['pairing'][0]} {cond}", r["opening_game_id"], role)] += (l, 1, l >= BLUNDER_CP)
    for r in csv.DictReader(open(res / "matches/h8_rep/stockfish/positions.csv")):
        if r["in_book"] == "True":
            continue
        role = "first" if r["player"].startswith("h8_rep") else "second"
        l = int(r["cp_loss"])
        agg[(r["pairing"], r["opening_game_id"], role)] += (l, 1, l >= BLUNDER_CP)
    # the original pairings compare h8 (first) with h1 (second)
    roles = {**{f"{p} {c}": ("h8", "h1") for p in "ABC" for c in ("off", "on")},
             **{p: ("first", "second") for p in REP}}
    out = {}
    for unit in ("id", "line"):
        k = len(ids) if unit == "id" else len(lines)
        draws = np.random.default_rng(0).integers(0, k, size=(N_BOOT, k))

        def v(cfg, role, j):
            a = np.array([agg[(cfg, i, role)][j] for i in ids])
            if unit == "id":
                return a
            return np.array([a[[ref[i] == L for i in ids]].mean() for L in lines])

        tag = "" if unit == "id" else " (45 lines)"
        rows = []
        for cfg, (r1, r2) in roles.items():
            res_ = {}
            for role in (r1, r2):
                L, M, B = (v(cfg, role, j) for j in range(3))
                res_[role] = (L.sum() / M.sum(), L[draws].sum(1) / M[draws].sum(1),
                              100 * B.sum() / M.sum(), 100 * B[draws].sum(1) / M[draws].sum(1), M.sum())
            a, b = res_[r1], res_[r2]
            d_acpl, c_acpl = a[0] - b[0], ci(a[1] - b[1])
            d_bl, c_bl = a[2] - b[2], ci(a[3] - b[3])
            out[f"{cfg}{tag}"] = {"acpl_first": round(a[0], 2), "acpl_second": round(b[0], 2),
                                  "blunder_first": round(a[2], 2), "blunder_second": round(b[2], 2),
                                  "moves_first": int(a[4]), "moves_second": int(b[4]),
                                  "d_acpl": [round(d_acpl, 2)] + [round(x, 2) for x in c_acpl],
                                  "d_blunder_pp": [round(d_bl, 3)] + [round(x, 3) for x in c_bl]}
            rows.append((cfg, f"{a[0]:.2f}", f"{b[0]:.2f}", f"{a[2]:.2f}", f"{b[2]:.2f}",
                         f"{d_acpl:+.2f} {fmt_ci(c_acpl)}", f"{d_bl:+.3f} {fmt_ci(c_bl, 3)}"))
        show(f"Move quality after the opening{tag}. First model is h8 (A-C) or h8+rep (rep_vs_*)", rows,
             ("configuration", "ACPL first", "ACPL second", "blunder % first", "blunder % second",
              "ACPL diff (cp)", "blunder diff (pp)"))
    OUT["move_quality"] = out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--results", type=Path, default=Path("results"))
    ap.add_argument("--json", type=Path, help="also write all values to this JSON file")
    args = ap.parse_args()
    dataset(args.results)
    validation(args.results)
    prediction(args.results)
    ref, ids, lines = matches(args.results)
    move_quality(args.results, ref, ids, lines)
    if args.json:
        args.json.write_text(json.dumps(OUT, indent=1) + "\n", encoding="utf-8")
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
