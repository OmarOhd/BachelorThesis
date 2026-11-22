import hashlib
import json
from pathlib import Path
import numpy as np


CACHE_SCHEMA = "binary-repetition-v1"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_paths(plies_path):
    directory = Path(plies_path).parent
    return directory / "repetition_flags.npy", directory / "repetition_flags.json"


def load_repetition_flags(plies_path, source_sha256, n_rows):
    flags_path, metadata_path = cache_paths(plies_path)
    if not flags_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Missing repetition flags. Run python -m data_pipeline.build_repetition_flags "
            "with the same DATA_ROOT first."
        )
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema") != CACHE_SCHEMA:
        raise ValueError("Unsupported repetition flag cache schema")
    if metadata.get("plies_sha256") != source_sha256 or metadata.get("n_rows") != n_rows:
        raise ValueError("Repetition flags belong to a different plies.jsonl; rebuild them")
    if metadata.get("flags_sha256") != sha256_file(flags_path):
        raise ValueError("Repetition flag file does not match its metadata; rebuild it")
    flags = np.load(flags_path, allow_pickle=False)
    if flags.dtype != np.uint8 or flags.shape != (n_rows,) or not np.isin(flags, [0, 1]).all():
        raise ValueError("Repetition flags must be a uint8 vector with one 0/1 per dataset row")
    return flags
