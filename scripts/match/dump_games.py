"""
dump_games.py — play the three checkpoint pairings and write PGN files.

Usage (from repo root):
  DATA_ROOT=/tmp/data python -m scripts.match.dump_games --games 30 --out games/

Plays three pairings (each --games games, colours alternating):
  A) h8@30k vs h1@30k   (matched steps, early)
  B) h8@30k vs h1@60k   (peak vs peak)
  C) h8@60k vs h1@60k   (matched steps, later)

Writes one PGN per pairing (all games concatenated) plus a combined file.
PGN is standard and importable to Lichess; per-position centipawn analysis is
done afterwards by analyze_games.py using a local Stockfish.

Move selection is identical to play_match.py: argmax over legal moves scored at
their encode_move index. No search, no temperature.
"""

import argparse
import datetime
from pathlib import Path

import chess
import chess.pgn

from scripts.match.common import Agent, collect_test_openings, pick_device, play_game

PAIRINGS = [
    ("A_h8-30k_vs_h1-30k", "checkpoints/h8_masked/model_step_30000.pt",
                            "checkpoints/h1_masked/model_step_30000.pt"),
    ("B_h8-30k_vs_h1-60k", "checkpoints/h8_masked/model_step_30000.pt",
                            "checkpoints/h1_masked/model_step_60000.pt"),
    ("C_h8-60k_vs_h1-60k", "checkpoints/h8_masked/model_step_60000.pt",
                            "checkpoints/h1_masked/model_step_60000.pt"),
]



def board_to_pgn(board, white, black, result, termination, n_opening,
                 pairing, game_no, opening_id, avoid_repetition, interventions):
    game = chess.pgn.Game.from_board(board)
    game.headers["Event"] = f"Chessformer thesis match — {pairing}"
    game.headers["Site"] = "local"
    game.headers["Date"] = datetime.date.today().strftime("%Y.%m.%d")
    game.headers["Round"] = str(game_no)
    game.headers["White"] = white.name
    game.headers["Black"] = black.name
    game.headers["Result"] = result
    # custom headers carried through to the analysis stage
    game.headers["WhiteHistory"] = str(white.history)
    game.headers["BlackHistory"] = str(black.history)
    game.headers["WhiteRepetitionFeatures"] = white.repetition_features
    game.headers["BlackRepetitionFeatures"] = black.repetition_features
    game.headers["OpeningGameId"] = str(opening_id)
    game.headers["OpeningPlies"] = str(n_opening)
    game.headers["Termination"] = termination
    game.headers["PlyCount"] = str(board.ply())
    game.headers["AvoidRepetition"] = str(avoid_repetition)
    game.headers["WhiteAvoidRepetition"] = str(avoid_repetition and white.history == 1)
    game.headers["BlackAvoidRepetition"] = str(avoid_repetition and black.history == 1)
    game.headers["WhiteRepetitionInterventions"] = str(interventions[chess.WHITE])
    game.headers["BlackRepetitionInterventions"] = str(interventions[chess.BLACK])
    return game


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-a", type=Path, help="Optional custom pairing; requires --ckpt-b")
    ap.add_argument("--ckpt-b", type=Path, help="Optional custom pairing; requires --ckpt-a")
    ap.add_argument("--games", type=int, default=30,
                    help="games per pairing (colours alternate, so use an even number)")
    ap.add_argument("--max-plies", type=int, default=300)
    ap.add_argument("--out", default="games")
    ap.add_argument("--avoid-repetition", action="store_true", default=True)
    ap.add_argument("--no-avoid-repetition", dest="avoid_repetition", action="store_false")
    args = ap.parse_args()
    if (args.ckpt_a is None) != (args.ckpt_b is None):
        ap.error("Supply both --ckpt-a and --ckpt-b for a custom pairing")
    pairings = ([("custom_pairing", args.ckpt_a, args.ckpt_b)]
                if args.ckpt_a is not None else PAIRINGS)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device()
    print(f"device: {device}")

    n_openings = (args.games + 1) // 2          # each opening played twice
    openings = collect_test_openings(n_openings)

    combined = out_dir / "all_games.pgn"
    with open(combined, "w", encoding="utf-8") as comb:
        for pairing, ckpt_a, ckpt_b in pairings:
            print(f"\n=== {pairing} ===")
            A = Agent(Path(ckpt_a), device)
            B = Agent(Path(ckpt_b), device)
            pgn_path = out_dir / f"{pairing}.pgn"
            game_no = 0
            tally = {A.name: [0, 0, 0], B.name: [0, 0, 0]}   # W D L
            with open(pgn_path, "w", encoding="utf-8") as fh:
                for gid, moves in openings:
                    for white, black in ((A, B), (B, A)):
                        if game_no >= args.games:
                            break
                        game_no += 1
                        n_open = len(moves)
                        board, result, termination, interventions = play_game(
                            white, black, moves, args.max_plies, args.avoid_repetition)
                        game = board_to_pgn(board, white, black, result, termination,
                                            n_open, pairing, game_no, gid, args.avoid_repetition, interventions)
                        text = str(game)
                        fh.write(text + "\n\n")
                        comb.write(text + "\n\n")
                        if result == "1-0":
                            tally[white.name][0] += 1; tally[black.name][2] += 1
                        elif result == "0-1":
                            tally[black.name][0] += 1; tally[white.name][2] += 1
                        else:
                            tally[white.name][1] += 1; tally[black.name][1] += 1
                        print(f"  game {game_no:3d}  {white.name} (W) vs "
                              f"{black.name} (B)  ->  {result}  ({board.ply()} plies)")
                    if game_no >= args.games:
                        break
            for name, (w, d, l) in tally.items():
                pts = w + d / 2
                print(f"  {name:24s} {pts:5.1f}/{game_no}  +{w} ={d} -{l}")
            print(f"  wrote {pgn_path}")

    print(f"\nwrote combined PGN: {combined}")
    print("next: python -m scripts.analysis.analyze_games --pgn games/ "
          "--stockfish /path/to/stockfish")


if __name__ == "__main__":
    main()
