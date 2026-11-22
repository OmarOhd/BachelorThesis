import json
import hashlib
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from model.encoder import encode_position, encode_move, legal_move_mask
from model.repetition_features import validate_repetition_features
from data_pipeline.repetition_cache import load_repetition_flags
import torch
import chess

import os
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data")).expanduser()
PLIES_PATH   = DATA_ROOT / "tcec/processed/plies.jsonl"
OFFSETS_PATH = DATA_ROOT / "tcec/processed/plies.offsets.npy"

HISTORY = 8  # paper uses 8 board states (current + 7 prior)


class TCECPlies(Dataset):
    def __init__(self, plies_path=PLIES_PATH, offsets_path=OFFSETS_PATH, history=HISTORY,
                 return_fen=False, return_legal_mask=False, *, return_game_id=False,
                 repetition_features="none"):
        self.plies_path = Path(plies_path)
        self.offsets = np.load(offsets_path)
        self.history = history
        self.return_fen = return_fen
        self.return_legal_mask = return_legal_mask
        self.return_game_id = return_game_id
        self.repetition_features = validate_repetition_features(repetition_features)
        self._fh = None  # opened lazily per process
        self.records = []
        source_hash = hashlib.sha256()
        with open(self.plies_path, "rb") as fh:
            for line in fh:
                source_hash.update(line)
                self.records.append(json.loads(line))
        self.plies_sha256 = source_hash.hexdigest()
        
        self.game_ids = np.array([rec["game_id"] for rec in self.records])
        self.fens = [r["fen"] for r in self.records]
        self.moves = [r["move"] for r in self.records]
        self.results = np.array([r["result"] for r in self.records], dtype=np.int8)
        del self.records  # free memory; we only need the fields above
        self.repetition_flags = None
        if self.repetition_features == "binary":
            self.repetition_flags = load_repetition_flags(
                self.plies_path, self.plies_sha256, len(self.fens))

    def __len__(self):
        return len(self.fens)

    def __getstate__(self):
        # Don't try to pickle the open file handle to DataLoader workers.
        state = self.__dict__.copy()
        state["_fh"] = None
        return state

    def _read_at(self, offset):
        if self._fh is None:
            self._fh = open(self.plies_path, "rb")
        self._fh.seek(int(offset))
        return json.loads(self._fh.readline())

    def __getitem__(self, i):
        target_game_id = self.game_ids[i]
        # Walk forward from i-(history-1) up to i-1, keeping plies from same game.
        # If we hit a game boundary partway in, reset, only the consecutive
        # tail of matching plies is real history.
        history_fens = []
        history_indices = []
        start = max(0, i - (self.history - 1))
        for j in range(start, i):
            if self.game_ids[j] == target_game_id:
                history_fens.append(self.fens[j])
                history_indices.append(j)
            else:
                history_fens = []  # crossed a game boundary
                history_indices = []

        fens = history_fens + [self.fens[i]]  # current ply is always included
        board = chess.Board(fens[-1])
        turn = board.turn 

        flags = None
        if self.repetition_flags is not None:
            flags = self.repetition_flags[history_indices + [i]].tolist()
        result_tensor = torch.from_numpy(encode_position(fens, repetition_flags=flags))  # [64, 112]
        policy_target = int(encode_move(chess.Move.from_uci(self.moves[i]), turn))
        value_target = int(self.results[i])
        if turn == chess.BLACK:
            value_target = 2 - value_target  # canonicalize for black as well (win=2, draw=1, loss=0)
        out = {
            "board": result_tensor,
            "policy_target": policy_target,
            "value_target": value_target,
        }
        if self.return_fen:
            out["fen"] = fens[-1]
        if self.return_legal_mask:
            out["legal_mask"] = torch.from_numpy(legal_move_mask(board))
        if self.return_game_id:
            out["game_id"] = str(target_game_id)
        return out


if __name__ == "__main__":
    import chess
    from torch.utils.data import DataLoader

    ds = TCECPlies()
    print(f"Dataset length: {len(ds):,}")

    # ---- history assembly (unchanged behaviour, still worth guarding) ----
    # We can't read 'fens' off the sample anymore (it returns tensors now),
    # so re-derive history length by replicating the walk, or just trust the
    # sample's board and check shape. Here we check the boundary cases via a
    # small helper that peeks at the raw history length.
    def raw_history_len(i):
        target = ds._read_at(ds.offsets[i])
        gid = target["game_id"]
        n = 0
        start = max(0, i - (ds.history - 1))
        for j in range(start, i):
            rec = ds._read_at(ds.offsets[j])
            n = n + 1 if rec["game_id"] == gid else 0
        return n + 1  # + current

    print("\n-- history assembly --")
    print(f"  ds[0] history len: {raw_history_len(0)} (expect 1)")

    rng = np.random.default_rng(0)
    deep_idx = int(rng.integers(1000, len(ds) - 1000))
    print(f"  ds[{deep_idx}] history len: {raw_history_len(deep_idx)} (expect up to 8)")

    for i in range(len(ds) - 1, len(ds) - 200, -1):
        if ds._read_at(ds.offsets[i])["ply"] == 0:
            print(f"  ds[{i}] (game-start) history len: {raw_history_len(i)} (expect 1)")
            break

    # ---- single-sample shape / range / dtype ----
    print("\n-- single sample --")
    s = ds[deep_idx]
    print(f"  board: {tuple(s['board'].shape)} {s['board'].dtype} "
          f"(expect (64, 112) torch.float32)")
    print(f"  policy_target: {s['policy_target']} "
          f"(expect int in [0, 4352))")
    print(f"  value_target:  {s['value_target']} (expect 0/1/2)")
    assert s["board"].shape == (64, 112)
    assert s["board"].dtype == torch.float32
    assert 0 <= s["policy_target"] < 4352
    assert s["value_target"] in (0, 1, 2)

    # ---- batched collate (default DataLoader) ----
    print("\n-- batch of 4 via DataLoader --")
    loader = DataLoader(ds, batch_size=4, shuffle=True)
    batch = next(iter(loader))
    print(f"  board:         {tuple(batch['board'].shape)} {batch['board'].dtype}")
    print(f"  policy_target: {tuple(batch['policy_target'].shape)} {batch['policy_target'].dtype}")
    print(f"  value_target:  {tuple(batch['value_target'].shape)} {batch['value_target'].dtype}")
    assert batch["board"].shape == (4, 64, 112)
    assert batch["board"].dtype == torch.float32
    assert batch["policy_target"].shape == (4,)
    assert batch["value_target"].shape == (4,)
    assert (batch["policy_target"] >= 0).all() and (batch["policy_target"] < 4352).all()
    assert ((batch["value_target"] >= 0) & (batch["value_target"] <= 2)).all()
    print("  all batch asserts passed")

# ---- the flip: black-to-move ply with a DECISIVE result (0 or 2) ----
    print("\n-- value flip (black to move, decisive game) --")
    for i in range(len(ds)):
        raw = ds._read_at(ds.offsets[i])
        if chess.Board(raw["fen"]).turn == chess.BLACK and raw["result"] != 1:
            sample = ds[i]
            expected = 2 - raw["result"]
            print(f"  ds[{i}]: raw result={raw['result']}, "
                  f"value_target={sample['value_target']}, expect {expected}")
            assert sample["value_target"] == expected and expected != raw["result"], \
                "flip wrong or vacuous!"
            print(f"  flip OK (genuinely swapped {raw['result']} -> {expected})")
            break

    # ---- white-to-move ply with a DECISIVE result, value must be UNCHANGED ----
    print("\n-- value flip (white to move, decisive, no change) --")
    for i in range(len(ds)):
        raw = ds._read_at(ds.offsets[i])
        if chess.Board(raw["fen"]).turn == chess.WHITE and raw["result"] != 1:
            sample = ds[i]
            assert sample["value_target"] == raw["result"], "white should not flip!"
            print(f"  ds[{i}]: raw result={raw['result']}, "
                  f"value_target={sample['value_target']} (unchanged, correct)")
            break
    
    print("\n-- fen return test --")
    sample = ds[deep_idx]
    assert "fen" not in sample, "fen should not be returned by default"

    ds_fen = TCECPlies(return_fen=True)
    s = ds_fen[deep_idx]
    assert "fen" in s, "fen should be returned when return_fen=True"
    chess.Board(s["fen"])          # raises if not a valid FEN -> that's the real check
    print(f"  fen returned: {s['fen']}")

    # collate check: a batch of FENs is a length-B tuple of strings, not a tensor
    loader = DataLoader(ds_fen, batch_size=4, shuffle=False)
    b = next(iter(loader))
    assert isinstance(b["fen"], (list, tuple)) and len(b["fen"]) == 4
    assert isinstance(b["fen"][0], str)
    print(f"  batched fen is a {type(b['fen']).__name__} of {len(b['fen'])} strings — collate OK")
    print("\n-- legal mask test --")
    assert "legal_mask" not in ds[deep_idx], "mask should not be returned by default"

    ds_mask = TCECPlies(return_legal_mask=True)
    s = ds_mask[deep_idx]
    assert s["legal_mask"].shape == (4352,) and s["legal_mask"].dtype == torch.bool
    print(f"  single sample: {int(s['legal_mask'].sum())} legal moves")

    loader = DataLoader(ds_mask, batch_size=4, shuffle=True)
    mb = next(iter(loader))
    assert mb["legal_mask"].shape == (4, 4352) and mb["legal_mask"].dtype == torch.bool
    sums = mb["legal_mask"].sum(dim=1)
    print(f"  batch legal-move counts: {sums.tolist()} (expect ~1-100 each)")
    assert ((sums >= 1) & (sums <= 218)).all()   # 218 = known max legal moves in chess
    # THE check: every target move must be marked legal in its own mask
    assert mb["legal_mask"].gather(1, mb["policy_target"].unsqueeze(1)).all(), \
        "target move not in mask — mask/target space diverged!"
    print("  target-in-mask gather passed")
