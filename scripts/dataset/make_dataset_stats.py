"""
make_dataset_stats.py — single-pass dataset audit for thesis figures.

Run from the REPO ROOT:   python -m scripts.dataset.make_dataset_stats
Reads:  data/tcec/processed/{games.jsonl, plies.jsonl, train/val/test_indices.npy}
Writes: dataset_stats.json   (small; paste or upload this back)

Collects ONLY aggregates (counts + pre-binned histograms), never raw rows,
so the output stays tiny. Schema-defensive: prints what games.jsonl actually
contains and only attempts Elo/engine extraction if those fields exist.
"""

import json
import numpy as np
from pathlib import Path
from collections import Counter

p = Path("data/tcec/processed/")
RESULT_NAMES = {0: "white_win", 1: "draw", 2: "black_win"}

# ----------------------------------------------------------------------
# 1. games.jsonl — load (small, ~53k records) and inspect schema
# ----------------------------------------------------------------------
games = [json.loads(line) for line in open(p / "games.jsonl", encoding="utf-8")]
game_keys = sorted(games[0].keys())
print("games.jsonl rows :", len(games))
print("games.jsonl keys :", game_keys)
print("sample record    :", games[0])

# detect Elo-ish and engine/name-ish fields by key name (case-insensitive)
elo_keys = [k for k in game_keys if "elo" in k.lower() or "rating" in k.lower()]
name_keys = [k for k in game_keys
             if k.lower() in ("white", "black", "white_player", "black_player",
                              "white_engine", "black_engine", "white_name", "black_name")]
result_key = next((k for k in game_keys if k.lower() == "result"), None)
print("detected elo keys  :", elo_keys)
print("detected name keys :", name_keys)
print("detected result key:", result_key)

# ----------------------------------------------------------------------
# 2. split labels (ply-index -> 0/1/2) via fancy indexing
# ----------------------------------------------------------------------
tr = np.load(p / "train_indices.npy")
va = np.load(p / "val_indices.npy")
te = np.load(p / "test_indices.npy")
N = len(tr) + len(va) + len(te)
split_label = np.empty(N, dtype=np.uint8)
split_label[tr] = 0
split_label[va] = 1
split_label[te] = 2
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}

# ----------------------------------------------------------------------
# 3. single pass over plies.jsonl
# ----------------------------------------------------------------------
plies_per_game = Counter()          # game_id -> #plies
result_by_ply = Counter()           # 0/1/2 -> count
result_by_split = np.zeros((3, 3), dtype=np.int64)   # [split, result]
games_in_split = [set(), set(), set()]

with open(p / "plies.jsonl", encoding="utf-8") as f:
    for i, line in enumerate(f):
        d = json.loads(line)
        gid = d["game_id"]
        r = int(d["result"])
        s = int(split_label[i])
        plies_per_game[gid] += 1
        result_by_ply[r] += 1
        result_by_split[s, r] += 1
        games_in_split[s].add(gid)

n_plies = sum(result_by_ply.values())

# per-game result: prefer games.jsonl's own field, else first ply's result
if result_key:
    result_by_game = Counter(int(g[result_key]) for g in games if result_key in g)
    result_by_game_source = f"games.jsonl['{result_key}']"
else:
    # fall back: one result per game from the ply scan
    first_result = {}
    for line in open(p / "plies.jsonl", encoding="utf-8"):
        d = json.loads(line)
        first_result.setdefault(d["game_id"], int(d["result"]))
    result_by_game = Counter(first_result.values())
    result_by_game_source = "first ply per game (games.jsonl had no result field)"

# ----------------------------------------------------------------------
# 4. game-length histogram (pre-binned so the JSON stays tiny)
# ----------------------------------------------------------------------
lengths = np.fromiter(plies_per_game.values(), dtype=np.int64)
len_edges = list(range(0, 401, 20)) + [10**9]      # 0-20,...,380-400, 400+
len_counts, _ = np.histogram(lengths, bins=len_edges)
len_pcts = {
    "min": int(lengths.min()), "p10": float(np.percentile(lengths, 10)),
    "median": float(np.median(lengths)), "mean": float(lengths.mean()),
    "p90": float(np.percentile(lengths, 90)), "max": int(lengths.max()),
}

# ----------------------------------------------------------------------
# 5. Elo / engine — only if the fields exist
# ----------------------------------------------------------------------
elo_block = None
if elo_keys:
    elos = []
    for g in games:
        for k in elo_keys:
            try:
                elos.append(float(g[k]))
            except (KeyError, TypeError, ValueError):
                pass
    if elos:
        elos = np.array(elos)
        lo, hi = int(elos.min()) // 100 * 100, int(elos.max()) // 100 * 100 + 100
        elo_edges = list(range(lo, hi + 1, 50))
        elo_counts, _ = np.histogram(elos, bins=elo_edges)
        elo_block = {
            "keys_used": elo_keys,
            "n_values": len(elos),
            "min": float(elos.min()), "mean": float(elos.mean()), "max": float(elos.max()),
            "hist_edges": elo_edges, "hist_counts": [int(c) for c in elo_counts],
        }

engine_block = None
if name_keys:
    engines = Counter()
    for g in games:
        for k in name_keys:
            v = g.get(k)
            if isinstance(v, str):
                engines[v] += 1
    if engines:
        engine_block = {"keys_used": name_keys,
                        "games_per_engine": dict(engines.most_common())}

# ----------------------------------------------------------------------
# 6. assemble + write
# ----------------------------------------------------------------------
stats = {
    "n_games_in_games_jsonl": len(games),
    "n_games_with_plies": len(plies_per_game),
    "n_plies": int(n_plies),
    "games_jsonl_keys": game_keys,
    "result_names": RESULT_NAMES,
    "result_by_ply": {str(k): int(v) for k, v in sorted(result_by_ply.items())},
    "result_by_game": {str(k): int(v) for k, v in sorted(result_by_game.items())},
    "result_by_game_source": result_by_game_source,
    "split_sizes_plies": {SPLIT_NAMES[s]: int((split_label == s).sum()) for s in range(3)},
    "split_sizes_games": {SPLIT_NAMES[s]: len(games_in_split[s]) for s in range(3)},
    "result_by_split_ply": {
        SPLIT_NAMES[s]: {str(r): int(result_by_split[s, r]) for r in range(3)}
        for s in range(3)
    },
    "game_length_hist": {"edges": len_edges[:-1] + ["400+"],
                         "counts": [int(c) for c in len_counts]},
    "game_length_summary": len_pcts,
    "elo": elo_block,
    "engines": engine_block,
}

with open("dataset_stats.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)

print("\n" + "=" * 60)
print("wrote dataset_stats.json — paste or upload this file back.")
print("=" * 60)
print(json.dumps(stats, indent=2))
