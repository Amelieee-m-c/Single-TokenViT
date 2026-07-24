"""
Builds an 80/20 stratified train/test split (class-balanced) from a pooled
source directory, matching the paper's stated protocol: "All datasets were
divided into 80 percent training and 20 percent evaluation sets with class
balance... performed at the image level... prior to training and remained
fixed."

Each of the 5 raw Kaggle downloads needs this differently:
  - corn, banana (OriginalSet), mango: ship as flat class folders, no split
    at all -- pool + split from scratch.
  - tomato: ships with its own train/val, but val is incomplete (6/10
    classes) and the total (10,584) is well under the paper's cited 16,012
    -- likely a newer/smaller version of the same Kaggle dataset than what
    the paper used. We pool just the balanced `train/` folder (10,000
    images, 1000/class) and split 80/20 ourselves, ignoring the incomplete
    `val/`.
  - groundnut: ships with its own train/test (7910/2451, ~76/24) that
    doesn't match the paper's stated 80/20 (8287/2074) -- pool both and
    re-split 80/20 ourselves. Total (10,361) matches the paper exactly.

Usage:
    python make_splits.py --source <pooled_or_mixed_dir> --output <out_dir> --dataset {corn,tomato,banana,mango,groundnut}
"""
import argparse
import random
import shutil
from pathlib import Path


def collect_corn(source: Path):
    return {d.name: list(d.iterdir()) for d in (source / "data").iterdir() if d.is_dir()}


def collect_tomato(source: Path):
    train_dir = source / "tomato" / "train"
    return {d.name: list(d.iterdir()) for d in train_dir.iterdir() if d.is_dir()}


def collect_banana(source: Path):
    orig = source / "BananaLSD" / "OriginalSet"
    return {d.name: list(d.iterdir()) for d in orig.iterdir() if d.is_dir()}


def collect_mango(source: Path):
    return {d.name: list(d.iterdir()) for d in source.iterdir() if d.is_dir()}


def collect_groundnut(source: Path):
    base = source / "Dataset of groundnut plant leaf images for classification and detection" / "Groundnut_Leaf_dataset"
    classes = {}
    for split in ["train", "test"]:
        for d in (base / split).iterdir():
            if d.is_dir():
                classes.setdefault(d.name, []).extend(d.iterdir())
    return classes


COLLECTORS = {
    "corn": collect_corn,
    "tomato": collect_tomato,
    "banana": collect_banana,
    "mango": collect_mango,
    "groundnut": collect_groundnut,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset", required=True, choices=list(COLLECTORS.keys()))
    ap.add_argument("--train_frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    source = Path(args.source)
    out = Path(args.output)

    class_files = COLLECTORS[args.dataset](source)
    class_files = {
        c: [f for f in files if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        for c, files in class_files.items()
    }

    print(f"classes found: {len(class_files)}")
    total_train, total_test = 0, 0
    for c, files in sorted(class_files.items()):
        random.shuffle(files)
        n_train = round(len(files) * args.train_frac)
        train_files, test_files = files[:n_train], files[n_train:]

        (out / "train" / c).mkdir(parents=True, exist_ok=True)
        (out / "test" / c).mkdir(parents=True, exist_ok=True)
        for f in train_files:
            shutil.copy2(f, out / "train" / c / f.name)
        for f in test_files:
            shutil.copy2(f, out / "test" / c / f.name)

        total_train += len(train_files)
        total_test += len(test_files)
        print(f"  {c:30s} total={len(files):5d} train={len(train_files):5d} test={len(test_files):5d}")

    print(f"\nTOTAL train={total_train} test={total_test} (grand total={total_train+total_test})")
    print(f"saved to: {out}")


if __name__ == "__main__":
    main()
