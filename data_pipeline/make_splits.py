import hashlib
import json
import numpy as np
from pathlib import Path

def bucket_for(game_id) -> int:#
    #returns a bucket number for a given game_id through modulo 10 to split into 10 buckets. Buckets are used to split the dataset into training, validation, and test sets.
    # use a hash function to get a consistent bucket number for each game_id
    hash_value = hashlib.md5(str(game_id).encode()).hexdigest()
    bucket_number = int(hash_value, 16) % 10  # Get a number between 0 and 9
    return bucket_number

def main():
    # Load dataset
    dataset_path = Path("data/tcec/processed/")
    train, val, test = [], [], []
    train_games, val_games, test_games = set(), set(), set()

    with open(dataset_path / "plies.jsonl", "r") as f:
        for i, line in enumerate(f):
            gid = json.loads(line)["game_id"]
            b = bucket_for(gid)
            if b < 7:
                train.append(i); train_games.add(gid)
            elif b < 9:
                val.append(i); val_games.add(gid)
            else:
                test.append(i); test_games.add(gid) 
    np.save(dataset_path / "train_indices.npy", np.asarray(train, dtype=np.int64))
    np.save(dataset_path / "val_indices.npy",   np.asarray(val,   dtype=np.int64))
    np.save(dataset_path / "test_indices.npy",  np.asarray(test,  dtype=np.int64))    
    
    #sanity check
    print(f"Games  — train: {len(train_games)}, val: {len(val_games)}, test: {len(test_games)}")
    assert len(train) + len(val) + len(test) == 7_631_404, "The total number of samples does not equal 7,631,404."

    #print per-split game count, should be 70/20/10 split
    total = len(train) + len(val) + len(test)
    print(f"Ply fractions — train: {len(train)/total:.3f}, val: {len(val)/total:.3f}, test: {len(test)/total:.3f}")
    assert not (train_games & val_games) and not (train_games & test_games) and not (val_games & test_games)

if __name__ == "__main__":
    main()