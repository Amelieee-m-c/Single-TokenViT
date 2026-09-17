"""Runs train.py sequentially over all 5 datasets (avoids GPU memory
contention from running them concurrently). Each dataset's --num_classes
matches its split's actual class count."""
import argparse
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train.py"
SPLITS = HERE.parent / "data" / "splits"
RUNS = HERE.parent / "runs"

DATASETS = ["banana", "mango", "corn", "groundnut", "tomato"]  # smallest to largest

ap = argparse.ArgumentParser()
ap.add_argument("--patch_embed", choices=["gap_linear", "depthwise"], default="gap_linear",
                 help="passed through to train.py -- see its --patch_embed help. Non-default "
                      "runs land in runs/<name>_<patch_embed>/ so they don't overwrite the "
                      "existing gap_linear results.")
args = ap.parse_args()

suffix = "" if args.patch_embed == "gap_linear" else f"_{args.patch_embed}"

for name in DATASETS:
    print(f"\n{'='*20} {name}{suffix} {'='*20}", flush=True)
    cmd = [
        sys.executable, str(TRAIN),
        "--data_dir", str(SPLITS / name),
        "--output_dir", str(RUNS / f"{name}{suffix}"),
        "--epochs", "35",
        "--batch_size", "32",
        "--lr", "1e-5",
        "--num_workers", "4",
        "--patch_embed", args.patch_embed,
    ]
    subprocess.run(cmd, check=True)

print("\nAll datasets done.")
