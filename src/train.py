"""
Training script matching the paper's exact protocol (Section III-B, III-E,
Table 3): AdamW, constant lr=1e-5, batch_size=32, 35 epochs, fixed
train/test split (no validation set, no early stopping -- the paper runs a
single fixed-epoch training pass per dataset). Preprocessing: resize to
224x224, ToTensor, normalize with mean=0.5/std=0.5 per RGB channel (maps
pixels to [-1, 1]). Train-only augmentation: random horizontal flip + random
rotation.

Run (once a dataset's 80/20 split exists under data/splits/<name>/{train,test}):
    python train.py --data_dir ../data/splits/corn --output_dir ../runs/corn --num_classes 4
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)

from model import DenseNetSingleTokenViT, count_params, count_all_params


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def build_transforms(img_size: int):
    normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        normalize,
    ])
    eval_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_tf, eval_tf


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss, total_correct, total_n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if train:
                optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            if train:
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_correct += (out.argmax(1) == y).sum().item()
            total_n += x.size(0)
    return total_loss / total_n, total_correct / total_n


@torch.no_grad()
def predict_all(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for x, y in loader:
        x = x.to(device)
        out = model(x)
        y_pred.extend(out.argmax(1).cpu().numpy().tolist())
        y_true.extend(y.numpy().tolist())
    return np.array(y_true), np.array(y_pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, help="dir with train/test subfolders")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=35)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--num_workers", type=int, default=4)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_dir = Path(args.data_dir)
    train_tf, eval_tf = build_transforms(args.img_size)

    train_ds = datasets.ImageFolder(data_dir / "train", transform=train_tf)
    test_ds = datasets.ImageFolder(data_dir / "test", transform=eval_tf)
    assert train_ds.classes == test_ds.classes
    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"classes ({num_classes}): {class_names}")
    print(f"train={len(train_ds)} test={len(test_ds)}")

    persistent = args.num_workers > 0
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, persistent_workers=persistent)

    model = DenseNetSingleTokenViT(num_classes=num_classes).to(device)
    print(f"trainable params: {count_params(model)}  total params: {count_all_params(model)}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)

    history = []
    train_start = time.time()
    for epoch in range(args.epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        test_loss, test_acc = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        history.append({"epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
                         "test_loss": test_loss, "test_acc": test_acc})
        print(f"epoch {epoch+1}/{args.epochs}  train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"test_loss={test_loss:.4f} test_acc={test_acc:.4f}")
    training_time_s = time.time() - train_start

    torch.save(model.state_dict(), out_dir / "final_model.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "class_names.json", "w") as f:
        json.dump(class_names, f, indent=2)

    x_axis = [h["epoch"] for h in history]
    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    axes[0].plot(x_axis, [h["train_acc"] for h in history], "o-", label="train")
    axes[0].plot(x_axis, [h["test_acc"] for h in history], "o-", label="test")
    axes[0].set_ylabel("accuracy"); axes[0].legend(); axes[0].set_title("Accuracy vs epoch")
    axes[1].plot(x_axis, [h["train_loss"] for h in history], ".-", label="train")
    axes[1].plot(x_axis, [h["test_loss"] for h in history], ".-", label="test")
    axes[1].set_ylabel("loss"); axes[1].set_xlabel("epoch"); axes[1].legend(); axes[1].set_title("Loss vs epoch")
    fig.tight_layout()
    fig.savefig(out_dir / "accuracy_loss.png", dpi=150)
    plt.close(fig)

    # ---- final test-set evaluation (single inference pass, also timed) ----
    infer_start = time.time()
    y_true, y_pred = predict_all(model, test_loader, device)
    inference_time_s = (time.time() - infer_start) / len(test_ds)

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)

    print("\n=== Test set results ===")
    print(f"accuracy:  {acc:.4f}")
    print(f"precision (macro): {precision:.4f}")
    print(f"recall (macro): {recall:.4f}")
    print(f"f1 (macro): {f1:.4f}")
    print(f"training_time_s: {training_time_s:.2f}")
    print(f"inference_time_s (per image): {inference_time_s:.4f}")
    print(report)

    fig, ax = plt.subplots(figsize=(max(6, num_classes), max(5, num_classes)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, cmap=plt.cm.Blues, vmin=0, vmax=1)
    ax.set_xticks(range(num_classes)); ax.set_yticks(range(num_classes))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    metrics = {
        "accuracy": acc,
        "precision_macro": precision,
        "recall_macro": recall,
        "f1_macro": f1,
        "confusion_matrix": cm.tolist(),
        "class_names": class_names,
        "classification_report": report,
        "training_time_s": training_time_s,
        "inference_time_s_per_image": inference_time_s,
        "trainable_params": count_params(model),
        "total_params": count_all_params(model),
    }
    with open(out_dir / "test_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nsaved: {out_dir}")


if __name__ == "__main__":
    main()
