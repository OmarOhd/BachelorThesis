import csv
from collections import Counter, defaultdict
import os

import torch
import chess
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, Subset
import argparse

from data_pipeline.dataset import TCECPlies
from model.encoder import encode_move
from model.main import _policy_flat, cf_6M_2024
from model.repetition_features import checkpoint_repetition_features

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data")).expanduser()

PROCESSED_DIR = DATA_ROOT / "tcec/processed"
BATCH_SIZE = 256
EVAL_SUBSET_N = 51_200
EVAL_SEED = 0
HISTORY = 8  # paper uses 8 board states (current + 7 prior)


@torch.no_grad()
def evaluate(model, val_loader, device, max_batches=None, *, by_game=False):
    was_training = model.training #snapshot
    model.eval()

    correct_raw = correct_legal = illegal_raw = total = 0
    game_counts = defaultdict(Counter)
    for b, batch in enumerate(val_loader):
        if max_batches is not None and b >= max_batches:
            break
        #forward -> flatten/cat to [B, 4352] (SAME as compute_loss)
        policy_logits, promotion_logits, value_logits = model(batch["board"].to(device))
        policy_logits = _policy_flat(policy_logits, promotion_logits)
        #pred_raw = argmax over all 4352 moves
        pred_raw = policy_logits.argmax(dim=1)
        #build legal mask [B, 4352] from batch["fen"]; logits.masked_fill(~mask, -inf)
        legal_mask = batch["legal_mask"].to(device) #single transfer to device
        #pred_legal = argmax over masked logits (only legal moves)
        masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
        pred_legal = masked_logits.argmax(dim=1)
        #accumulate (pred == policy_target) for both, and B into total
        target = batch["policy_target"].to(device)
        raw_hit = pred_raw == target
        legal_hit = pred_legal == target
        raw_illegal = ~legal_mask.gather(1, pred_raw.unsqueeze(1)).squeeze(1)
        correct_raw += raw_hit.sum().item()
        correct_legal += legal_hit.sum().item()
        illegal_raw += raw_illegal.sum().item()
        total += batch["board"].size(0)

        if by_game:
            outcomes = torch.stack((raw_hit, legal_hit, raw_illegal), dim=1).cpu().tolist()
            for game_id, (raw, legal, illegal) in zip(batch["game_id"], outcomes, strict=True):
                counts = game_counts[game_id]
                counts["n"] += 1
                counts["correct_raw"] += raw
                counts["correct_legal"] += legal
                counts["raw_illegal"] += illegal

    model.train(was_training) #restore
    stats = {"top1_raw": correct_raw / total, "top1_legal": correct_legal / total,
             "raw_illegal_rate": illegal_raw / total, "n": total}
    if by_game:
        stats["by_game"] = game_counts
    return stats

def build_eval_loader(history=HISTORY, batch_size=BATCH_SIZE, n=EVAL_SUBSET_N,
                      seed=EVAL_SEED, *, split="val", return_game_id=False,
                      repetition_features="none", dataset=None):
    if split not in ("val", "test"):
        raise ValueError(f"Unknown evaluation split {split!r}")
    if n is not None and n < 1:
        raise ValueError("Sample size must be positive")

    indices = np.load(PROCESSED_DIR / f"{split}_indices.npy")
    if len(indices) == 0:
        raise ValueError(f"The {split} split is empty")

    if n is not None and n < len(indices):
        rng = np.random.default_rng(seed)
        indices = np.sort(rng.choice(indices, size=n, replace=False))

    if dataset is None:
        ds = TCECPlies(history=history, return_fen=True, return_legal_mask=True,
                       return_game_id=return_game_id, repetition_features=repetition_features)
    else:
        ds = dataset
        if ds.history != history or ds.repetition_features != repetition_features:
            raise ValueError("Shared evaluation dataset has incompatible input settings")
        if not ds.return_legal_mask or (return_game_id and not ds.return_game_id):
            raise ValueError("Shared evaluation dataset lacks required evaluation fields")
    return DataLoader(Subset(ds, indices), batch_size=batch_size, shuffle=False,
                      generator=torch.Generator().manual_seed(seed))

def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--by-game", type=Path, metavar="CSV",
                        help="Write prediction counts for each source TCEC game")
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--sample-size", type=int,
                        help="Position limit. Defaults to 51,200 on val and all positions on test.")
    parser.add_argument("--history", type=int, choices=(1, 8),
                        help="History depth for a checkpoint without history metadata")
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    repetition_features = checkpoint_repetition_features(ckpt)
    history = ckpt.get("history", args.history)
    if history not in (1, 8):
        parser.error("Expected history 1 or 8. For a legacy checkpoint, supply --history.")
    if args.history is not None and args.history != history:
        parser.error("--history conflicts with the history stored in the checkpoint")

    n = args.sample_size
    if n is None and args.split == "val":
        n = EVAL_SUBSET_N

    device = pick_device()
    model = cf_6M_2024()
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    del ckpt

    print(f"checkpoint={args.ckpt}  history={history}  split={args.split}  device={device}  "
          f"repetition_features={repetition_features}")
    by_game = args.by_game is not None
    loader = build_eval_loader(history=history, split=args.split, n=n, return_game_id=by_game,
                               repetition_features=repetition_features)
    stats = evaluate(model, loader, device, by_game=by_game)
    print(f"n={stats['n']:,}  top1_raw={stats['top1_raw']:.4f}  "
          f"top1_legal={stats['top1_legal']:.4f}  raw_illegal_rate={stats['raw_illegal_rate']:.4f}")

    if by_game:
        args.by_game.parent.mkdir(parents=True, exist_ok=True)
        fields = ("checkpoint", "history", "split", "game_id", "n",
                  "correct_raw", "correct_legal", "raw_illegal", "repetition_features")
        with args.by_game.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            for game_id, counts in sorted(stats["by_game"].items()):
                writer.writerow({"checkpoint": str(args.ckpt), "history": history,
                                 "split": args.split, "game_id": game_id, **counts,
                                 "repetition_features": repetition_features})
        print(f"wrote {args.by_game}  ({len(stats['by_game']):,} games)")


if __name__ == "__main__":
    main()
