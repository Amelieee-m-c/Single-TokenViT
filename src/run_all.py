"""Runs train.py sequentially over all 5 datasets (avoids GPU memory
contention from running them concurrently). Each dataset's --num_classes
matches its split's actual class count."""
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
TRAIN = HERE / "train.py"
SPLITS = HERE.parent / "data" / "splits"
RUNS = HERE.parent / "runs"

DATASETS = ["banana", "mango", "corn", "groundnut", "tomato"]  # smallest to largest

for name in DATASETS:
    print(f"\n{'='*20} {name} {'='*20}", flush=True)
    cmd = [
        sys.executable, str(TRAIN),
        "--data_dir", str(SPLITS / name),
        "--output_dir", str(RUNS / name),
        "--epochs", "35",
        "--batch_size", "32",
        "--lr", "1e-5",
        "--num_workers", "4",
    ]
    subprocess.run(cmd, check=True)

print("\nAll datasets done.")
