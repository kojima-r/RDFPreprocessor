import argparse
import json
import os
import numpy as np
import glob
from typing import Iterator, Tuple, List


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def make_output_paths(output_dir: str, input_path: str):
    base_ = os.path.basename(input_path)
    base,_ = os.path.splitext(base_)
    return {
        "train": os.path.join(output_dir, f"{base}.train.tsv"),
        "valid": os.path.join(output_dir, f"{base}.valid.tsv"),
        "test": os.path.join(output_dir, f"{base}.test.tsv"),
    }


def preprocess_file(
    input_path: str,
    output_dir: str,
    train_ratio: float,
    valid_ratio: float,
    seed: int,
):
    rng = np.random.default_rng(seed)
    paths = make_output_paths(output_dir, input_path)
    counts = {"train": 0, "valid": 0, "test": 0}
    with open(paths["train"], "w", encoding="utf-8") as f_train, \
         open(paths["valid"], "w", encoding="utf-8") as f_valid, \
         open(paths["test"], "w", encoding="utf-8") as f_test:

        with open(input_path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                p = rng.uniform()
                if p < train_ratio:
                    split = "train"
                    out_f = f_train
                elif p < train_ratio + valid_ratio:
                    split = "valid"
                    out_f = f_valid
                else:
                    split = "test"
                    out_f = f_test
                out_f.write(f"{line}\n")
    return paths, counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="data10")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--valid_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    data_paths=list(glob.glob("data06_uniq/*.graph.tsv"))
    test_ratio = 1.0 - args.train_ratio - args.valid_ratio
    if args.train_ratio <= 0 or args.valid_ratio < 0 or test_ratio < 0:
        raise ValueError("train_ratio, valid_ratio の指定が不正です。")

    ensure_dir(args.output_dir)
    all_stats = {}
    for i, path in enumerate(data_paths):
        print(f"Processing: {path}")
        paths, counts = preprocess_file(
            input_path=path,
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed + i,
        )
        all_stats[path] = {"output_paths": paths, "counts": counts}

    with open(os.path.join(args.output_dir, "split_stats.json"), "w", encoding="utf-8") as f:
        json.dump(all_stats, f, ensure_ascii=False, indent=2)

    print("Done.")
    print(f"Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
