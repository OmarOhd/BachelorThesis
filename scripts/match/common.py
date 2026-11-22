"""
common.py — shared machinery for play_match.py and dump_games.py.

Openings are sampled deterministically (seed 0) from TEST-split games (same md5
bucket rule as make_splits: bucket 9). Move selection is pure argmax over legal
moves scored at their encode_move index — no search, no temperature.
"""

import os
import json
import hashlib
from pathlib import Path

import numpy as np
import torch
import chess

from model.main import cf_6M_2024, _policy_flat
from model.encoder import encode_position, encode_move
from model.repetition_features import checkpoint_repetition_features, repetition_flags_for_history
from scripts.match.repetition import move_selector, occurence_tracker, record_position

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data")).expanduser()
PLIES_PATH = DATA_ROOT / "tcec/processed/plies.jsonl"

OPENING_PLIES = 8
SEED = 0


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available()
                        else "mps" if torch.backends.mps.is_available() else "cpu")


def bucket_for(game_id) -> int:
    return int(hashlib.md5(str(game_id).encode()).hexdigest(), 16) % 10


def collect_test_openings(n_openings: int):
    """Stream plies.jsonl once; return n_openings move-sequences (each a list of
    UCI strings, length OPENING_PLIES) from distinct test-split games."""
    seqs = {}  # game_id -> {ply: uci} for plies 0..OPENING_PLIES-1
    with open(PLIES_PATH, "rb") as f:
        for line in f:
            rec = json.loads(line)
            ply = rec.get("ply")
            if ply is None or ply >= OPENING_PLIES:
                continue
            gid = rec["game_id"]
            if bucket_for(gid) != 9:          # test bucket only
                continue
            seqs.setdefault(gid, {})[ply] = rec["move"]
    # keep games with a complete first-8 sequence that replays legally
    complete = []
    for gid, d in seqs.items():
        if len(d) == OPENING_PLIES:
            moves = [d[k] for k in range(OPENING_PLIES)]
            b = chess.Board()
            try:
                for u in moves:
                    b.push_uci(u)
            except ValueError:
                continue
            complete.append((gid, moves))
    complete.sort(key=lambda t: str(t[0]))    # deterministic order before sampling
    rng = np.random.default_rng(SEED)
    idx = rng.choice(len(complete), size=min(n_openings, len(complete)), replace=False)
    picked = [complete[i] for i in sorted(idx)]
    print(f"openings: {len(picked)} sampled from {len(complete)} eligible test games")
    return picked

def game_status(board: chess.Board, repetition_counts: dict, max_plies: int) -> tuple[str, str] | None:
    draw_status = board.outcome(claim_draw=False)
    if draw_status is not None:
        return draw_status.result(), draw_status.termination.name
    threefold = occurence_tracker(board, repetition_counts) >= 3
    if threefold:
        return "1/2-1/2", "threefold"
    if board.is_fifty_moves():
        return "1/2-1/2", "fifty-move"
    if board.ply() >= max_plies:
        return "1/2-1/2", "ply-cap"
    return None



class Agent:
    repetition_features = "none"

    def __init__(self, ckpt_path: Path, device):
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        self.history = int(ckpt.get("history", 8))
        self.repetition_features = checkpoint_repetition_features(ckpt)
        self.step = ckpt.get("step", "?")
        self.name = f"{Path(ckpt_path).parent.name}@{self.step}"
        self.model = cf_6M_2024().to(device)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.eval()
        self.device = device
        print(f"loaded {self.name}  (history={self.history}, masked={ckpt.get('masked')}, "
              f"repetition_features={self.repetition_features})")

    @torch.no_grad()
    def pick_move(self, board: chess.Board, game_fens: list, repetition_counts: dict, avoid_repetition: bool) -> chess.Move | None:
        fens = game_fens[-self.history:]
        repetition_check = False
        if avoid_repetition:
            if self.history == 1:
                repetition_check = True
        flags = (repetition_flags_for_history(board, fens)
                 if self.repetition_features == "binary" else None)
        enc = torch.from_numpy(encode_position(fens, repetition_flags=flags)).unsqueeze(0).to(self.device)
        pol, promo, _val = self.model(enc)
        flat = _policy_flat(pol, promo)[0]                                          # [4352]
        sorted_moves = sorted(board.legal_moves, key=lambda m: flat[encode_move(m, board.turn)].item(), reverse=True)

        best_mv = move_selector(board, sorted_moves, repetition_counts, repetition_check)

        self.last_repetition_intervention = False
        if repetition_check and best_mv is not None and best_mv != sorted_moves[0]:
            self.last_repetition_intervention = True

        return best_mv
    
def play_game(white: Agent, black: Agent, opening_moves: list, max_plies: int, avoid_repetition: bool = True) -> tuple[chess.Board, str, str, dict[chess.Color, int]]:
    board = chess.Board()
    repetition_counts = {}
    game_fens = [board.fen()]
    counter = {chess.WHITE: 0, chess.BLACK: 0}
    #record initial pos to repetition_counts dict
    record_position(board, repetition_counts)
    for u in opening_moves:
        board.push_uci(u)
        game_fens.append(board.fen())
        record_position(board, repetition_counts)
    while True:
        status = game_status(board, repetition_counts, max_plies)
        if status is not None:
            return (board, status[0], status[1], counter)  # completed board, result, termination reason
        agent = white if board.turn == chess.WHITE else black
        move = agent.pick_move(board, game_fens, repetition_counts, avoid_repetition)
        if move is None:
            raise RuntimeError(f"{agent.name} failed to pick a move at ply {board.ply()}")
        mover = board.turn
        board.push(move)
        if agent.last_repetition_intervention:
            counter[mover] += 1
        
        record_position(board, repetition_counts)
        game_fens.append(board.fen())

