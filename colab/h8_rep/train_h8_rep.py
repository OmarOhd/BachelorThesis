import argparse
import os
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
RUN_NAME = "h8_rep"
STEPS = 100_000
# Each learning rate applies up to and including the given update.
SCHEDULE = ((5_000, 5e-4), (50_000, 1e-4), (90_000, 3.16e-5), (STEPS, 1e-5))
DATA_FILES = ("plies.jsonl", "plies.offsets.npy", "train_indices.npy",
              "val_indices.npy", "test_indices.npy")
FLAG_FILES = ("repetition_flags.npy", "repetition_flags.json")


def h8_rep_lr(completed_steps, lr_horizon=None):
    for last_update, lr in SCHEDULE:
        if completed_steps < last_update:
            return lr
    return SCHEDULE[-1][1]


def restore_data(processed, archive):
    if all((processed / name).is_file() for name in DATA_FILES):
        return
    if archive is None:
        raise SystemExit(f"Processed data missing in {processed}. Build it with "
                         "data_pipeline or pass --data-zip")
    print(f"extracting {archive}...", flush=True)
    processed.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=processed.parent) as tmp:
        with zipfile.ZipFile(archive) as bundle:
            bundle.extractall(tmp)
        found = [path for path in Path(tmp).rglob("plies.jsonl") if "__MACOSX" not in path.parts]
        if len(found) != 1:
            raise SystemExit(f"{archive} must contain exactly one plies.jsonl")
        for path in found[0].parent.iterdir():
            shutil.move(str(path), processed / path.name)
    missing = [name for name in DATA_FILES if not (processed / name).is_file()]
    if missing:
        raise SystemExit(f"{archive} is missing {', '.join(missing)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path("/content/data"))
    parser.add_argument("--data-zip", type=Path, help="zip of data/tcec/processed")
    parser.add_argument("--output", type=Path, default=Path("/content/drive/MyDrive/h8_rep"))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    data_root = args.data_root.expanduser().resolve()
    output = args.output.expanduser().resolve()

    os.environ["DATA_ROOT"] = str(data_root)
    os.environ["NVIDIA_TF32_OVERRIDE"] = "0"
    os.environ["TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"] = "0"
    sys.path.insert(0, str(ROOT))
    import torch
    from data_pipeline.build_repetition_flags import build_repetition_flags
    from model import train_model as trainer

    if not torch.cuda.is_available():
        raise SystemExit("No CUDA GPU found")
    if torch.cuda.get_device_properties(0).total_memory < 35 * 2**30:
        raise SystemExit("FP32 at batch size 2,048 needs the 40 GB A100 runtime")
    torch.set_float32_matmul_precision("highest")

    processed = data_root / "tcec/processed"
    restore_data(processed, args.data_zip)
    if not all((processed / name).is_file() for name in FLAG_FILES):
        build_repetition_flags(processed / "plies.jsonl", force=True)

    output.mkdir(parents=True, exist_ok=True)
    os.chdir(output)
    resume = any((Path("checkpoints") / RUN_NAME).glob("model_step_*.pt"))
    trainer.lr_for_step = h8_rep_lr
    trainer.main(["--history", "8", "--repetition-features", "binary", "--seed", "0",
                  "--steps", str(STEPS), "--lr-horizon", str(STEPS), "--device", "cuda",
                  "--workers", str(args.workers), "--run-name", RUN_NAME]
                 + (["--resume"] if resume else []))


if __name__ == "__main__":
    main()
