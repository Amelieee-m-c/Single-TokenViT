"""
Burst-aware (leakage-free) 80/20 split for MangoLeafBD, to measure real
generalization accuracy now that the per-image random split (make_splits.py)
was confirmed to leak near-duplicate photos across train/test: raw filenames
are capture timestamps ("20211008_124249 (Custom).jpg"), and consecutive
photos of the same physical leaf were shot seconds apart in the same
"burst". 5 independent per-image-random-split seeds all landed at exactly
100.00% test accuracy -- see project memory, this is the burst-duplicate
leakage, not genuine perfect generalization.

Groups images into bursts by parsing the timestamp and chaining any two
consecutive (sorted) images that are <= --gap_seconds apart into the same
burst (transitive: A-B<=gap and B-C<=gap groups A,B,C even if A-C>gap).
Then assigns whole bursts (never split across train/test) greedily to hit
an 80/20 split per class as closely as possible.

Usage:
    python make_mango_split_deduped.py --source <mango raw class-folder dir> --output <out_dir>
"""
import argparse
import random
import re
import shutil
from datetime import datetime
from pathlib import Path

TS_RE = re.compile(r"(\d{8}_\d{6})")


def parse_ts(path: Path):
    m = TS_RE.search(path.name)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d_%H%M%S")


def group_bursts(files: list[Path], gap_seconds: int):
    """Sort by timestamp, chain-group consecutive images within gap_seconds."""
    dated = [(parse_ts(f), f) for f in files]
    undated = [f for ts, f in dated if ts is None]
    dated = sorted([(ts, f) for ts, f in dated if ts is not None], key=lambda x: x[0])

    groups = []
    current = []
    prev_ts = None
    for ts, f in dated:
        if prev_ts is not None and (ts - prev_ts).total_seconds() > gap_seconds:
            groups.append(current)
            current = []
        current.append(f)
        prev_ts = ts
    if current:
        groups.append(current)
    # any file with no parseable timestamp is its own singleton group
    groups.extend([[f]] for f in undated)
    return groups


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True, help="dir with one subfolder per class (raw MangoLeafBD)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--gap_seconds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    source = Path(args.source)
    out = Path(args.output)

    total_train = total_test = total_bursts = 0
    for class_dir in sorted(d for d in source.iterdir() if d.is_dir()):
        files = [f for f in class_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        groups = group_bursts(files, args.gap_seconds)
        random.shuffle(groups)

        n_total = len(files)
        n_train_target = round(n_total * args.train_frac)

        train_files, test_files = [], []
        n_train_so_far = 0
        for g in groups:
            # keep assigning whole bursts to train until we'd overshoot the target
            if n_train_so_far < n_train_target:
                train_files.extend(g)
                n_train_so_far += len(g)
            else:
                test_files.extend(g)

        (out / "train" / class_dir.name).mkdir(parents=True, exist_ok=True)
        (out / "test" / class_dir.name).mkdir(parents=True, exist_ok=True)
        for f in train_files:
            shutil.copy2(f, out / "train" / class_dir.name / f.name)
        for f in test_files:
            shutil.copy2(f, out / "test" / class_dir.name / f.name)

        total_train += len(train_files)
        total_test += len(test_files)
        total_bursts += len(groups)
        print(f"  {class_dir.name:20s} total={n_total:5d} bursts={len(groups):4d} "
              f"train={len(train_files):5d} test={len(test_files):5d}")

    print(f"\nTOTAL train={total_train} test={total_test} bursts={total_bursts} "
          f"(grand total={total_train + total_test})")
    print(f"saved to: {out}")


if __name__ == "__main__":
    main()
