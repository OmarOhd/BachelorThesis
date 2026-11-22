import chess
import chess.pgn
from pathlib import Path

PGN_DIR = Path("data/tcec/events/full/events")
RESULT_MAP = {"1-0": 0, "1/2-1/2": 1, "0-1": 2}
EXCLUDE_TOKENS = ("FRC", "DFRC", "FRD") # non standard chess variants are excluded
SKIP_BOOK_MOVES = False # true to skip opening book moves, we keep them by default since they are part of the real game distribution
MAX_GAMES = None 
suffix = f"_dryrun_{MAX_GAMES}" if MAX_GAMES else ""
GAMES_PATH = Path(f"data/tcec/processed/games{suffix}.jsonl")
PLIES_PATH = Path(f"data/tcec/processed/plies{suffix}.jsonl")


def keep_filename(p: Path) -> bool:
    return not any(tok in p.name for tok in EXCLUDE_TOKENS)

def parse_int(s):
    try:
        return int(s)
    except (TypeError, ValueError):
        return None

def stream_games(pgn_path):
    """Yield (game_meta, plies) per valid game."""
    with open(pgn_path) as f:
        game_idx = 0
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                return
            h = game.headers
            result = RESULT_MAP.get(h.get("Result", "*"))
            if result is None:
                continue
            if game.board().fen() != chess.STARTING_FEN:
                continue

            game_id = f"{pgn_path.stem}:{game_idx}"
            game_idx += 1

            game_meta = {
                "id": game_id,
                "event": h.get("Event"),
                "white": h.get("White"),
                "black": h.get("Black"),
                "white_elo": parse_int(h.get("WhiteElo")),
                "black_elo": parse_int(h.get("BlackElo")),
                "eco": h.get("ECO"),
                "opening": h.get("Opening"),
                "time_control": h.get("TimeControl"),
                "result": result,
                "ply_count": parse_int(h.get("PlyCount")),
                "termination": h.get("TerminationDetails"),
                "date": h.get("Date"),
            }

            plies = []
            board = game.board()
            node = game
            ply = 0
            while node.variations:
                node = node.variations[0]
                move = node.move
                is_book = "book" in (node.comment or "").lower()
                if not (SKIP_BOOK_MOVES and is_book):
                    plies.append({
                        "game_id": game_id,
                        "ply": ply,
                        "fen": board.fen(),
                        "move": move.uci(),
                        "result": result,
                    })
                board.push(move)
                ply += 1

            yield game_meta, plies


if __name__ == "__main__":
    import json
    files = sorted(p for p in PGN_DIR.glob("*.pgn") if keep_filename(p))
    print(f"Processing {len(files)} files...")
    GAMES_PATH.parent.mkdir(parents=True, exist_ok=True)

    n_games = n_plies = errored = 0
    done = False
    with open(GAMES_PATH, "w") as gf, open(PLIES_PATH, "w") as pf:
        for i, pgn in enumerate(files):
            if done:
                break
            try:
                for game_meta, plies in stream_games(pgn):
                    gf.write(json.dumps(game_meta) + "\n")
                    for ply in plies:
                        pf.write(json.dumps(ply) + "\n")
                    n_games += 1
                    n_plies += len(plies)
                    if MAX_GAMES is not None and n_games >= MAX_GAMES:
                        done = True
                        break
            except Exception as e:
                print(f"  ! {pgn.name}: {e}")
                errored += 1
            if (i + 1) % 50 == 0:
                print(f"  [{i+1}/{len(files)}] {n_games:,} games, {n_plies:,} plies")
    print(f"Done. {n_games:,} games, {n_plies:,} plies, {errored} files errored.")
    