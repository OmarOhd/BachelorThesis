import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import chess
import numpy as np

from data_pipeline.repetition_cache import CACHE_SCHEMA, cache_paths, sha256_file


def build_repetition_flags(plies_path, *, force=False, progress_every=100_000):
    plies_path = Path(plies_path)
    flags_path, metadata_path = cache_paths(plies_path)
    if not force and (flags_path.exists() or metadata_path.exists()):
        raise FileExistsError("Repetition flag output already exists; use --force to rebuild")

    source_hash = hashlib.sha256()
    flags = bytearray()
    seen_games = set()
    current_game = None
    board = None
    expected_ply = 0
    with plies_path.open("rb") as stream:
        for row_no, line in enumerate(stream, start=1):
            source_hash.update(line)
            record = json.loads(line)
            game_id = record["game_id"]
            if board is None or game_id != current_game:
                if game_id in seen_games:
                    raise ValueError(f"Row {row_no}: game {game_id!r} is not contiguous")
                seen_games.add(game_id)
                current_game = game_id
                board = chess.Board()
                expected_ply = 0
            if record["ply"] != expected_ply:
                raise ValueError(
                    f"Row {row_no}: expected ply {expected_ply}, got {record['ply']}; "
                    "flags require the complete game prefix (including book moves)"
                )
            if record["fen"] != board.fen():
                raise ValueError(f"Row {row_no}: stored FEN disagrees with the replayed game")
            move = board.parse_uci(record["move"])
            if not move:
                raise ValueError(f"Row {row_no}: null moves are not supported")
            # Record the feature BEFORE the target move. Never inspect future moves.
            flags.append(int(board.is_repetition(2)))
            board.push(move)
            expected_ply += 1
            if progress_every and row_no % progress_every == 0:
                print(f"  {row_no:,} positions, {len(seen_games):,} games", flush=True)

    if not flags:
        raise ValueError("plies.jsonl is empty")

    # A failed build never leaves a half-written cache that looks valid.
    with tempfile.TemporaryDirectory(prefix="repetition-build-", dir=plies_path.parent) as tmp:
        tmp_flags = Path(tmp) / flags_path.name
        tmp_metadata = Path(tmp) / metadata_path.name
        array = np.frombuffer(flags, dtype=np.uint8)
        np.save(tmp_flags, array, allow_pickle=False)
        metadata = {
            "schema": CACHE_SCHEMA,
            "plies_sha256": source_hash.hexdigest(),
            "flags_sha256": sha256_file(tmp_flags),
            "n_rows": len(flags),
            "n_games": len(seen_games),
            "repeated_positions": int(array.sum()),
        }
        tmp_metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp_flags, flags_path)
        os.replace(tmp_metadata, metadata_path)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    default = Path(os.environ.get("DATA_ROOT", "data")).expanduser() / "tcec/processed/plies.jsonl"
    parser.add_argument("--plies", type=Path, default=default)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    stats = build_repetition_flags(args.plies, force=args.force)
    repeated, total = stats["repeated_positions"], stats["n_rows"]
    print(f"wrote repetition flags: {total:,} positions, {stats['n_games']:,} games")
    print(f"current-position flag = 1: {repeated:,}/{total:,} ({100 * repeated / total:.3f}%)")
    print(f"files: {cache_paths(args.plies)[0]} and {cache_paths(args.plies)[1]}")


if __name__ == "__main__":
    main()
