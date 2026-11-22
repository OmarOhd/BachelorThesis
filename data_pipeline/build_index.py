import json
import numpy as np
from pathlib import Path

PLIES_PATH = Path("data/tcec/processed/plies.jsonl")
INDEX_PATH = Path("data/tcec/processed/plies.offsets.npy")


def build():
    offsets = []
    with open(PLIES_PATH, "rb") as f:  # binary mode — see note below
        pos = 0
        for line in f:
            offsets.append(pos)
            pos += len(line)
    arr = np.array(offsets, dtype=np.int64)
    np.save(INDEX_PATH, arr)
    return arr


def sanity_check(offsets):
    """Seek to a few random offsets and confirm the line parses correctly."""
    rng = np.random.default_rng(0)
    sample_indices = rng.integers(0, len(offsets), size=5)
    with open(PLIES_PATH, "rb") as f:
        for i in sorted(sample_indices):
            f.seek(int(offsets[i]))
            rec = json.loads(f.readline())
            print(f"  ply {i:>8}: {rec['game_id']:<40} ply_in_game={rec['ply']:>3}  move={rec['move']}")


if __name__ == "__main__":
    print(f"Indexing {PLIES_PATH} ...")
    offsets = build()
    print(f"Built index with {len(offsets):,} entries.")
    print(f"Saved to {INDEX_PATH} ({INDEX_PATH.stat().st_size / 1e6:.1f} MB)")
    print("\nSanity check — 5 random plies:")
    sanity_check(offsets)
