import argparse
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
import torch.optim as optim
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Subset

from data_pipeline.dataset import TCECPlies
from data_pipeline.repetition_cache import sha256_file
from model.main import cf_6M_2024, compute_loss
from model.eval_model import evaluate, build_eval_loader
from model.repetition_features import REPETITION_MODES

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data")).expanduser()
TRAIN_IDX_PATH = DATA_ROOT / "tcec/processed/train_indices.npy"
EVAL_INTERVAL = 10_000
CHECKPOINT_INTERVAL = 1000

# Same optimizer, batch size and loss as the existing experiments.
BATCH_SIZE = 2048
LR = 5e-4
MIN_LR1 = 1.58e-4
MIN_LR2 = 5e-5
BETA1 = 0.9
BETA2 = 0.98
EPS = 1e-7
GRAD_CLIP = 10
LR_DROP_FRACS = (0.8, 0.9)
AMP_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
PRECISION_KEYS = ("amp", "amp_dtype", "loss_dtype")


def lr_for_step(step, total_steps):
    if step < LR_DROP_FRACS[0] * total_steps:
        return LR
    if step < LR_DROP_FRACS[1] * total_steps:
        return MIN_LR1
    return MIN_LR2


class SeededBatchSampler:
    def __init__(self, size, batch_size, seed):
        if size < 1 or batch_size < 1:
            raise ValueError("Training split and batch size must be nonempty")
        self.size = size
        self.batch_size = batch_size
        self.seed = seed
        self.batches_per_epoch = (size + batch_size - 1) // batch_size
        self.set_step(0)

    def set_step(self, completed_steps):
        self.epoch, self.start_batch = divmod(completed_steps, self.batches_per_epoch)

    def __len__(self):
        return self.batches_per_epoch - self.start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        indices = torch.randperm(self.size, generator=generator).tolist()
        for start in range(self.start_batch * self.batch_size, self.size, self.batch_size):
            yield indices[start:start + self.batch_size]


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=int, default=8, choices=(1, 8))
    parser.add_argument("--unmasked", action="store_true")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--amp-dtype", choices=tuple(AMP_DTYPES), default="float16",
                        help="CUDA autocast dtype; bfloat16 is recommended on an A100")
    parser.add_argument("--allow-precision-change", action="store_true",
                        help="Explicitly allow and record a precision change when resuming")
    parser.add_argument("--steps", type=int, default=2_000_000,
                        help="Stop after this many completed training batches")
    parser.add_argument("--lr-horizon", type=int,
                        help="Fixed LR schedule horizon; defaults to --steps on a fresh run")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Positions per batch; use a separate small-batch run for smoke tests")
    parser.add_argument("--log-every", type=int, default=100,
                        help="Print training progress at this interval, plus first and final step")
    parser.add_argument("--trace-first-step", action="store_true",
                        help="Print and synchronize stages of the first training step")
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repetition-features", choices=REPETITION_MODES, default="none")
    parser.add_argument("--run-name", help="Separate subdirectory under checkpoints/")
    parser.add_argument("--resume", action="store_true",
                        help="Explicitly resume a checkpoint written by this trainer version")
    args = parser.parse_args(argv)
    if args.allow_precision_change and not args.resume:
        parser.error("--allow-precision-change requires --resume")
    if args.lr_horizon is None:
        args.lr_horizon = args.steps
    if args.steps < 1 or args.lr_horizon < 1 or args.workers < 0:
        parser.error("Steps and LR horizon must be positive; workers must be nonnegative")
    if args.batch_size < 1 or args.log_every < 1:
        parser.error("Batch size and logging interval must be positive")
    if not 0 <= args.seed < 2**32:
        parser.error("Seed must be in [0, 2**32)")
    if args.run_name is None:
        mask = "unmasked" if args.unmasked else "masked"
        suffix = "rep" if args.repetition_features == "binary" else "norep"
        args.run_name = f"h{args.history}_{mask}_{suffix}_seed{args.seed}"
    if args.run_name in ("", ".", "..") or any(c in args.run_name for c in "/\\"):
        parser.error("--run-name must be a single directory name")
    return args


def checkpoint_precision(checkpoint):
    """Infer precision for checkpoints written before dtype metadata existed."""
    amp = checkpoint.get("amp", False)
    return {"amp": amp,
            "amp_dtype": checkpoint.get("amp_dtype", "float16" if amp else "float32"),
            "loss_dtype": checkpoint.get("loss_dtype", "float16" if amp else "float32")}


def validate_resume(checkpoint, settings, *, allow_precision_change=False):
    previous = {**checkpoint, **checkpoint_precision(checkpoint)}
    for key, expected in settings.items():
        if key not in previous or previous[key] != expected:
            if key in PRECISION_KEYS:
                if allow_precision_change:
                    continue
                raise ValueError(
                    f"Cannot resume: {key} is {previous.get(key)!r}, expected {expected!r}. "
                    "Use --allow-precision-change to record an intentional precision change."
                )
            raise ValueError(
                f"Cannot resume: {key} is {previous.get(key)!r}, expected {expected!r}. "
                "Use the original settings or a new run name. Legacy checkpoints remain "
                "usable for evaluation, but cannot resume this new training protocol."
            )


def ensure_finite_state(value, name):
    if torch.is_tensor(value):
        if (value.is_floating_point() or value.is_complex()) and not torch.isfinite(value).all():
            raise ValueError(f"Cannot resume: non-finite checkpoint tensor {name}")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Cannot resume: non-finite checkpoint scalar {name}")
    elif isinstance(value, dict):
        for key, item in value.items():
            ensure_finite_state(item, f"{name}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            ensure_finite_state(item, f"{name}[{index}]")


def loss_diagnostics(inputs, logits, policy_target, legal_mask):
    details = {"nonfinite_inputs": int((~torch.isfinite(inputs)).sum().item())}
    for name, tensor in zip(("policy", "promotion", "value"), logits):
        details[f"nonfinite_{name}_logits"] = int((~torch.isfinite(tensor)).sum().item())
    if legal_mask is not None:
        details["empty_legal_masks"] = int((~legal_mask.any(dim=1)).sum().item())
        details["masked_out_targets"] = int((~legal_mask.gather(
            1, policy_target.unsqueeze(1))).sum().item())
    return details


def pick_device():
    return torch.device("cuda" if torch.cuda.is_available() else
                        "mps" if torch.backends.mps.is_available() else "cpu")


def synchronize_device(device):
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path("checkpoints") / args.run_name
    ckpt_files = sorted(run_dir.glob("model_step_*.pt"),
                        key=lambda path: int(path.stem.split("_")[-1]))
    if args.resume and not ckpt_files:
        raise SystemExit("--resume was requested but this run has no checkpoint")
    if not args.resume and run_dir.exists() and any(run_dir.iterdir()):
        raise SystemExit("Run directory is not empty. Use a new --run-name, or explicit --resume")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = pick_device() if args.device == "auto" else torch.device(args.device)
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS is unavailable in this Python/PyTorch environment")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable in this Python/PyTorch environment")
    use_amp = args.amp and device.type == "cuda"
    amp_dtype = AMP_DTYPES[args.amp_dtype]
    if use_amp and amp_dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise SystemExit("This CUDA device does not support bfloat16 autocast")
    masked = not args.unmasked
    print(f"run={args.run_name} history={args.history} repetition_features={args.repetition_features} "
          f"seed={args.seed} steps={args.steps} lr_horizon={args.lr_horizon} device={device} "
          f"batch_size={args.batch_size} amp={use_amp} "
          f"amp_dtype={args.amp_dtype if use_amp else 'float32'} loss_dtype=float32", flush=True)
    if args.amp and not use_amp:
        print("--amp is CUDA-only in this trainer; this run uses float32.", flush=True)

    print("loading training dataset and validating repetition flags...", flush=True)
    ds = TCECPlies(history=args.history, return_legal_mask=masked,
                   repetition_features=args.repetition_features)
    train_indices = np.load(TRAIN_IDX_PATH, allow_pickle=False)
    train_ds = Subset(ds, train_indices)
    print(f"train subset: {len(train_ds):,}; preparing validation loader...", flush=True)
    val_loader = build_eval_loader(history=args.history,
                                   repetition_features=args.repetition_features,
                                   dataset=ds if masked else None)
    sampler = SeededBatchSampler(len(train_ds), args.batch_size, args.seed)
    loader_options = {"num_workers": args.workers, "pin_memory": device.type == "cuda",
                      "generator": torch.Generator().manual_seed(args.seed)}
    if args.workers:
        loader_options.update(persistent_workers=True, prefetch_factor=4)
    dataloader = DataLoader(train_ds, batch_sampler=sampler, **loader_options)

    print(f"creating model on {device}...", flush=True)
    model = cf_6M_2024().to(device)
    print("model created; preparing optimizer and run settings...", flush=True)
    optimizer = optim.NAdam(model.parameters(), betas=(BETA1, BETA2), eps=EPS, lr=LR)
    # bfloat16 has the float32 exponent range and does not need loss scaling.
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    settings = {
        "training_format": "repetition-v1",
        "history": args.history,
        "masked": masked,
        "repetition_features": args.repetition_features,
        "seed": args.seed,
        "lr_horizon": args.lr_horizon,
        "batch_size": args.batch_size,
        "amp": use_amp,
        "amp_dtype": args.amp_dtype if use_amp else "float32",
        "loss_dtype": "float32",
        "plies_sha256": ds.plies_sha256,
        "train_indices_sha256": sha256_file(TRAIN_IDX_PATH),
    }
    precision_history = [{"first_step": 1, **{key: settings[key] for key in PRECISION_KEYS}}]
    global_step = 0
    if args.resume:
        checkpoint = torch.load(ckpt_files[-1], map_location="cpu", weights_only=False)
        validate_resume(checkpoint, settings, allow_precision_change=args.allow_precision_change)
        ensure_finite_state(checkpoint["model_state_dict"], "model_state_dict")
        ensure_finite_state(checkpoint["optimizer_state_dict"], "optimizer_state_dict")
        previous_precision = checkpoint_precision(checkpoint)
        precision_history = list(checkpoint.get("precision_history", [
            {"first_step": 1, **previous_precision}]))
        new_precision = {key: settings[key] for key in PRECISION_KEYS}
        if previous_precision != new_precision:
            precision_history.append({"first_step": int(checkpoint["step"]) + 1,
                                      "resumed_from": ckpt_files[-1].name, **new_precision})
            print(f"precision change: {previous_precision} -> {new_precision}", flush=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # NAdam's scalar state stays on CPU; parameter-sized moments follow weights.
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state = optimizer.state.get(parameter, {})
                for key, value in state.items():
                    if torch.is_tensor(value):
                        state[key] = (value.cpu() if value.numel() == 1 else
                                      value.to(device=parameter.device, dtype=parameter.dtype))
        if scaler.is_enabled() and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        global_step = int(checkpoint["step"])
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        del checkpoint
        print(f"resumed after {global_step:,} completed steps", flush=True)

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(
        json.dumps({**settings, "stop_at_step": args.steps,
                    "precision_history": precision_history}, indent=2) + "\n", encoding="utf-8")
    model.train()
    t0 = time.time()
    start_step = global_step
    if global_step >= args.steps:
        print(f"already at step {global_step}; target is {args.steps}", flush=True)
        return
    print(f"training ready; loading batch for step {global_step + 1}...", flush=True)
    while global_step < args.steps:
        sampler.set_step(global_step)
        for batch in dataloader:
            if global_step >= args.steps:
                break
            trace_step = args.trace_first_step and global_step == start_step
            def trace(message, *, sync=False):
                if trace_step:
                    if sync:
                        synchronize_device(device)
                    print(f"[step {global_step + 1}] {message}", flush=True)

            trace("batch loaded; transferring tensors...")
            inputs = batch["board"].to(device)
            policy_target = batch["policy_target"].to(device)
            value_target = batch["value_target"].to(device)
            legal_mask = batch["legal_mask"].to(device) if masked else None
            trace("transfers complete; forward pass...", sync=True)
            optimizer.zero_grad()
            with autocast(enabled=use_amp, dtype=amp_dtype, device_type=device.type):
                policy_logits, promotion_logits, value_logits = model(inputs)
            trace("forward pass complete; computing loss...", sync=True)
            loss = compute_loss(policy_logits, promotion_logits, value_logits,
                                policy_target, value_target, legal_mask=legal_mask)
            if not torch.isfinite(loss):
                details = {"attempted_step": global_step + 1, "completed_steps": global_step,
                           "loss": str(loss.item()), **settings,
                           **loss_diagnostics(inputs, (policy_logits, promotion_logits, value_logits),
                                              policy_target, legal_mask)}
                report = run_dir / f"nonfinite_loss_step_{global_step + 1}.json"
                report.write_text(json.dumps(details, indent=2) + "\n", encoding="utf-8")
                raise SystemExit(f"Non-finite loss attempting step {global_step + 1} "
                                 f"({global_step} completed); stopping. Diagnostics: {report}")
            trace("loss ready; backward pass...", sync=True)
            scaler.scale(loss).backward()
            trace("backward pass complete; optimizer update...", sync=True)
            scaler.unscale_(optimizer)
            # GradScaler protects fp16 updates. Without it, stop before a
            # non-finite gradient can corrupt model parameters or NAdam moments.
            clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP,
                            error_if_nonfinite=not scaler.is_enabled())
            lr = lr_for_step(global_step, args.lr_horizon)
            for group in optimizer.param_groups:
                group["lr"] = lr
            scaler.step(optimizer)
            scaler.update()
            trace("optimizer update complete", sync=True)
            global_step += 1

            if global_step % EVAL_INTERVAL == 0:
                stats = evaluate(model, val_loader, device)
                print(f"[eval] step {global_step}: top1_raw={stats['top1_raw']:.4f} "
                      f"top1_legal={stats['top1_legal']:.4f} n={stats['n']:,}", flush=True)
                with (run_dir / "eval_log.csv").open("a", encoding="utf-8") as stream:
                    stream.write(f"{global_step},{stats['top1_raw']},{stats['top1_legal']},{stats['n']}\n")
            if global_step == start_step + 1 or global_step % args.log_every == 0 or global_step == args.steps:
                print(f"step {global_step}: loss={loss.item():.4f} time={time.time() - t0:.2f}s lr={lr:.6f}",
                      flush=True)
                t0 = time.time()
                with (run_dir / "train_log.csv").open("a", encoding="utf-8") as stream:
                    stream.write(f"{global_step},{loss.item()},{lr}\n")
            if global_step % CHECKPOINT_INTERVAL == 0 or global_step == args.steps:
                tmp = run_dir / f".tmp_step_{global_step}.pt"
                torch.save({
                    **settings,
                    "precision_history": precision_history,
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scaler_state_dict": scaler.state_dict(),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all() if device.type == "cuda" else None,
                }, tmp)
                tmp.replace(run_dir / f"model_step_{global_step}.pt")
                print(f"saved {run_dir / f'model_step_{global_step}.pt'}", flush=True)


if __name__ == "__main__":
    main()
