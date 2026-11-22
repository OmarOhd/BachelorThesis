"""
analyze_games.py — per-position centipawn analysis of the dumped games.

Usage (from repo root):
  python -m scripts.analysis.analyze_games --pgn games/ --stockfish $(which stockfish) --depth 16

Why local Stockfish and not the Lichess API:
  Lichess's /api/cloud-eval only returns evaluations for positions ALREADY in its
  cloud database (popular human positions). Our models leave book within ~10
  moves into positions no human has played, so the hit rate would be near zero.
  Lichess's own game analysis is Stockfish server-side; running Stockfish locally
  computes the same metric for EVERY position, reproducibly and without rate
  limits. The PGN files remain importable to Lichess for qualitative review.

Outputs:
  - positions.csv : one row per ply (game, ply, mover, eval before/after, cp loss)
  - summary.csv   : per-checkpoint aggregate statistics
  - printed report with ACPL, blunder/mistake/inaccuracy rates and CIs
"""

import argparse
import statistics
from pathlib import Path

import chess
import chess.pgn
import chess.engine

from scripts.analysis.common import INACCURACY, MISTAKE, BLUNDER, CP_CAP, MATE_CP


def score_to_cp(score, pov_color):
    """PovScore -> centipawns from pov_color's perspective, clamped."""
    s = score.pov(pov_color)
    if s.is_mate():
        return MATE_CP if s > chess.engine.Cp(0) else -MATE_CP
    return max(-CP_CAP, min(CP_CAP, s.score()))


def analyse_game(game, engine, limit):
    """Yield one dict per ply: who moved, eval before/after (mover's POV), cp loss."""
    board = game.board()
    opening_plies = int(game.headers.get("OpeningPlies", 0))
    rows = []
    # eval of the starting position (mover = white to move at ply 0)
    info = engine.analyse(board, limit)
    prev_eval_white = score_to_cp(info["score"], chess.WHITE)

    for ply, move in enumerate(game.mainline_moves()):
        mover = board.turn
        eval_before = prev_eval_white if mover == chess.WHITE else -prev_eval_white
        board.push(move)
        info = engine.analyse(board, limit)
        eval_after_white = score_to_cp(info["score"], chess.WHITE)
        eval_after = eval_after_white if mover == chess.WHITE else -eval_after_white
        # centipawn loss: how much the position worsened for the player who moved
        cp_loss = max(0, eval_before - eval_after)
        rows.append({
            "ply": ply,
            "in_book": ply < opening_plies,
            "mover": "white" if mover == chess.WHITE else "black",
            "player": game.headers["White"] if mover == chess.WHITE else game.headers["Black"],
            "move": move.uci(),
            "eval_before": eval_before,
            "eval_after": eval_after,
            "cp_loss": cp_loss,
        })
        prev_eval_white = eval_after_white

    # The recorded match may already have ended in a draw even when an
    # engine search from the final board returns a nonzero continuation score.
    if rows and game.headers.get("Result") == "1/2-1/2":
        outcome = board.outcome(claim_draw=False)
        drawn = outcome is not None and outcome.winner is None
        termination = game.headers.get("Termination")
        drawn = drawn or (termination == "threefold" and board.is_repetition(3))
        drawn = drawn or (termination == "fifty-move" and board.is_fifty_moves())
        if drawn:
            rows[-1]["eval_after"] = 0
            rows[-1]["cp_loss"] = max(0, rows[-1]["eval_before"])
    return rows


def mean_ci(values):
    """Mean and 95% CI half-width (normal approximation)."""
    if not values:
        return 0.0, 0.0
    m = statistics.fmean(values)
    if len(values) < 2:
        return m, 0.0
    sd = statistics.stdev(values)
    return m, 1.96 * sd / (len(values) ** 0.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pgn", default="games", help="PGN file or directory")
    ap.add_argument("--stockfish", required=True, help="path to the stockfish binary")
    ap.add_argument("--depth", type=int, default=16)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--hash", type=int, default=512, help="engine hash in MB")
    ap.add_argument("--skip-book", action="store_true",
                    help="exclude the replayed opening plies from the statistics")
    ap.add_argument("--out", default="analysis")
    args = ap.parse_args()

    pgn_path = Path(args.pgn)
    files = sorted(pgn_path.glob("*.pgn")) if pgn_path.is_dir() else [pgn_path]
    files = [f for f in files if f.name != "all_games.pgn"] or files
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    engine = chess.engine.SimpleEngine.popen_uci(args.stockfish)
    engine.configure({"Threads": args.threads, "Hash": args.hash})
    limit = chess.engine.Limit(depth=args.depth)
    print(f"stockfish: {args.stockfish}  depth={args.depth} threads={args.threads}")

    all_rows = []
    try:
        for f in files:
            pairing = f.stem
            with open(f, encoding="utf-8") as fh:
                g_idx = 0
                while True:
                    game = chess.pgn.read_game(fh)
                    if game is None:
                        break
                    g_idx += 1
                    rows = analyse_game(game, engine, limit)
                    for r in rows:
                        r["pairing"] = pairing
                        r["game"] = g_idx
                        r["result"] = game.headers.get("Result", "*")
                        r["opening_game_id"] = game.headers.get("OpeningGameId", "")
                        r["avoid_repetition"] = game.headers.get("AvoidRepetition", "")
                        r["white"] = game.headers["White"]
                        r["black"] = game.headers["Black"]
                    all_rows.extend(rows)
                    print(f"  {pairing} game {g_idx}: {len(rows)} plies analysed")
    finally:
        engine.quit()

    # ---- write per-position CSV -------------------------------------------
    pos_csv = out_dir / "positions.csv"
    cols = ["pairing", "game", "ply", "in_book", "mover", "player", "move",
            "eval_before", "eval_after", "cp_loss", "result",
    "opening_game_id", "avoid_repetition", "white", "black",]
    with open(pos_csv, "w", encoding="utf-8") as fh:
        fh.write(",".join(cols) + "\n")
        for r in all_rows:
            fh.write(",".join(str(r[c]) for c in cols) + "\n")
    print(f"\nwrote {pos_csv}  ({len(all_rows)} positions)")

    # ---- aggregate per (pairing, player) -----------------------------------
    rows = [r for r in all_rows if not (args.skip_book and r["in_book"])]
    groups = {}
    for r in rows:
        groups.setdefault((r["pairing"], r["player"]), []).append(r)

    sum_csv = out_dir / "summary.csv"
    header = ["pairing", "player", "moves", "acpl", "acpl_ci95", "median_cp_loss",
              "inaccuracies", "mistakes", "blunders", "blunder_rate_pct"]
    lines = []
    print("\n" + "=" * 96)
    print(f"{'pairing':22s} {'player':22s} {'moves':>6s} {'ACPL':>14s} "
          f"{'med':>5s} {'inacc':>6s} {'mist':>5s} {'blund':>6s} {'blund%':>7s}")
    print("-" * 96)
    for (pairing, player), rs in sorted(groups.items()):
        losses = [r["cp_loss"] for r in rs]
        acpl, ci = mean_ci(losses)
        med = statistics.median(losses) if losses else 0
        inacc = sum(1 for x in losses if INACCURACY <= x < MISTAKE)
        mist = sum(1 for x in losses if MISTAKE <= x < BLUNDER)
        blund = sum(1 for x in losses if x >= BLUNDER)
        rate = 100 * blund / len(losses) if losses else 0
        print(f"{pairing:22s} {player:22s} {len(losses):6d} "
              f"{acpl:7.1f} ±{ci:5.1f} {med:5.0f} {inacc:6d} {mist:5d} "
              f"{blund:6d} {rate:6.2f}%")
        lines.append([pairing, player, len(losses), f"{acpl:.2f}", f"{ci:.2f}",
                      f"{med:.0f}", inacc, mist, blund, f"{rate:.2f}"])
    print("=" * 96)
    with open(sum_csv, "w", encoding="utf-8") as fh:
        fh.write(",".join(header) + "\n")
        for l in lines:
            fh.write(",".join(str(x) for x in l) + "\n")
    print(f"wrote {sum_csv}")
    print("\nACPL = average centipawn loss per move (lower is better).")
    print("Thresholds: inaccuracy >=50cp, mistake >=100cp, blunder >=300cp "
          f"(evals clamped at +/-{CP_CAP}cp, mate = {MATE_CP}cp).")
    if not args.skip_book:
        print("NOTE: replayed opening plies are INCLUDED — rerun with --skip-book "
              "to measure only moves the models actually chose.")


if __name__ == "__main__":
    main()
