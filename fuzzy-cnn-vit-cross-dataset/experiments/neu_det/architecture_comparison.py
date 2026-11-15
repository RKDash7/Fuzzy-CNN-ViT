#!/usr/bin/env python3
"""Architecture ablation study for the NEU-DET Fuzzy CNN-ViT.

Runs nine predefined Transformer configurations over multiple random seeds.
The validation split is used for scheduler/checkpoint/early-stopping decisions;
the held-out test split is evaluated only after the best validation checkpoint
has been restored.

Expected data layout:
  DATA_ROOT/NEU-DET/train/images/<class>/*
  DATA_ROOT/NEU-DET/validation/images/<class>/*

Example (Colab):
  python architecture_comparison.py --data-root /content/data \
      --output-dir /content/architecture_comparison --seeds 42 123 202 456 789

For a quick pipeline check, use --epochs 2 --patience 2 --seeds 42.
"""

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    cohen_kappa_score, matthews_corrcoef,
)


CONFIGS = [
    {"name": "1L-64D-4H", "layers": 1, "dim": 64, "heads": 4},
    {"name": "2L-64D-4H", "layers": 2, "dim": 64, "heads": 4},
    {"name": "4L-64D-4H", "layers": 4, "dim": 64, "heads": 4},
    {"name": "6L-64D-4H", "layers": 6, "dim": 64, "heads": 4},
    {"name": "2L-128D-4H", "layers": 2, "dim": 128, "heads": 4},
    {"name": "2L-128D-8H", "layers": 2, "dim": 128, "heads": 8},
    {"name": "4L-128D-8H", "layers": 4, "dim": 128, "heads": 8},
    {"name": "6L-128D-8H", "layers": 6, "dim": 128, "heads": 8},
    {"name": "6L-256D-8H", "layers": 6, "dim": 256, "heads": 8},
]


# ----------------------------- Reproducibility -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ------------------------------- Model --------------------------------------
class GaussianFuzzyLayer(nn.Module):
    def __init__(self, channels: int = 1, epsilon: float = 1e-5):
        super().__init__()
        self.center = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.raw_sigma = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.epsilon = epsilon

    def forward(self, x):
        sigma = F.softplus(self.raw_sigma) + self.epsilon
        return torch.exp(-((x - self.center) ** 2) / (2 * sigma ** 2))


class CNNFeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim=512, dropout=0.1):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(f"embed_dim={embed_dim} must be divisible by num_heads={num_heads}")
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        z = self.norm1(x)
        attn_out, attn_weights = self.attention(
            z, z, z, need_weights=False,
        )
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class FuzzyCNNViT(nn.Module):
    def __init__(self, image_size, num_classes, embed_dim, num_heads,
                 num_layers, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.fuzzy_layer = GaussianFuzzyLayer(channels=1)
        self.cnn = CNNFeatureExtractor()
        self.patch_embedding = nn.Conv2d(
            in_channels=64, out_channels=embed_dim,
            kernel_size=2, stride=2,
        )
        # Two CNN max-pooling stages and one patch-embedding stride-2 stage.
        patch_grid = image_size // 8
        num_patches = patch_grid ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

    def forward(self, x):
        x = self.fuzzy_layer(x)
        x = self.cnn(x)
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embedding
        for block in self.transformer_blocks:
            x = block(x)
        return self.classifier(self.norm(x[:, 0]))


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ------------------------------- Data ---------------------------------------
def build_loaders(data_root: Path, image_size: int, batch_size: int,
                  num_workers: int, seed: int):
    train_dir = data_root / "NEU-DET" / "train" / "images"
    val_dir = data_root / "NEU-DET" / "validation" / "images"
    if not train_dir.is_dir() or not val_dir.is_dir():
        raise FileNotFoundError(
            "Expected folders: "
            f"{train_dir} and {val_dir}. Check --data-root and dataset extraction."
        )

    base_train = datasets.ImageFolder(str(train_dir))
    base_val = datasets.ImageFolder(str(val_dir))
    if base_train.classes != base_val.classes:
        raise ValueError("Training and validation class folder names/order differ.")
    if len(base_train.classes) != 6:
        raise ValueError(f"Expected six classes, found {len(base_train.classes)}: {base_train.classes}")

    train_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])
    eval_tf = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(), transforms.Normalize([0.5], [0.5]),
    ])

    targets = np.asarray(base_train.targets)
    indices = np.arange(len(base_train))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.20, stratify=targets, random_state=seed,
    )
    train_ds = Subset(datasets.ImageFolder(str(train_dir), transform=train_tf), train_idx)
    test_ds = Subset(datasets.ImageFolder(str(train_dir), transform=eval_tf), test_idx)
    val_ds = datasets.ImageFolder(str(val_dir), transform=eval_tf)

    loader_args = dict(
        batch_size=batch_size, num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_args)
    return train_loader, val_loader, test_loader, base_train.classes, len(train_ds), len(val_ds), len(test_ds)


# ----------------------------- Train/evaluate -------------------------------
def run_train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.size(0)
    return loss_sum / count, correct / count


@torch.inference_mode()
def evaluate_loss_accuracy(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    count = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss_sum += criterion(logits, labels).item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.size(0)
    return loss_sum / count, correct / count


@torch.inference_mode()
def predict(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(logits.argmax(1).cpu().numpy().tolist())
    return np.asarray(y_true), np.asarray(y_pred)


def calculate_metrics(y_true, y_pred):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "Recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "F1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "Kappa": cohen_kappa_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def train_one_run(config, seed, args, device, output_root):
    set_seed(seed)
    train_loader, val_loader, test_loader, class_names, n_train, n_val, n_test = build_loaders(
        Path(args.data_root), args.image_size, args.batch_size,
        args.num_workers, args.split_seed,
    )
    model = FuzzyCNNViT(
        image_size=args.image_size, num_classes=len(class_names),
        embed_dim=config["dim"], num_heads=config["heads"],
        num_layers=config["layers"], mlp_dim=args.mlp_dim,
        dropout=args.dropout,
    ).to(device)
    params = count_trainable_parameters(model)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=args.scheduler_patience,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_state = None
    bad_epochs = 0
    history = []
    run_dir = output_root / config["name"] / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_validation_checkpoint.pth"

    start = time.time()
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = run_train_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = evaluate_loss_accuracy(model, val_loader, criterion, device)
        scheduler.step(va_loss)
        history.append({
            "configuration": config["name"], "seed": seed, "epoch": epoch,
            "train_loss": tr_loss, "train_accuracy": tr_acc,
            "validation_loss": va_loss, "validation_accuracy": va_acc,
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        print(
            f"[{config['name']} | seed {seed}] epoch {epoch:03d}/{args.epochs} "
            f"train_loss={tr_loss:.4f} val_loss={va_loss:.4f} "
            f"val_acc={va_acc:.4f}", flush=True,
        )
        if va_loss < best_val_loss - args.min_delta:
            best_val_loss = va_loss
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            torch.save({
                "model_state_dict": best_state,
                "config": config,
                "seed": seed,
                "class_names": class_names,
                "image_size": args.image_size,
                "best_validation_loss": best_val_loss,
                "best_epoch": best_epoch,
                "trainable_parameters": params,
            }, checkpoint_path)
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= args.patience:
            print(f"Early stopping at epoch {epoch}.", flush=True)
            break

    if best_state is None:
        raise RuntimeError(f"No validation checkpoint was produced for {config['name']} seed {seed}")

    model.load_state_dict(best_state)
    y_true, y_pred = predict(model, test_loader, device)
    metrics = calculate_metrics(y_true, y_pred)
    elapsed = time.time() - start
    pd.DataFrame(history).to_csv(run_dir / "training_history.csv", index=False)
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(
        run_dir / "test_predictions.csv", index=False,
    )
    result = {
        "Configuration": config["name"], "Layers": config["layers"],
        "Embedding": config["dim"], "Heads": config["heads"],
        "Parameters": params, "Seed": seed,
        **metrics, "Best_Epoch": best_epoch,
        "Best_Validation_Loss": best_val_loss,
        "Training_Time_Seconds": elapsed,
        "Train_Images": n_train, "Validation_Images": n_val,
        "Test_Images": n_test,
    }
    del model, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result, history


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=str, default="/content/data")
    parser.add_argument("--output-dir", type=str, default="./results/architecture_comparison")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 202, 456, 789])
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--min-delta", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--mlp-dim", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--split-seed", type=int, default=42,
                        help="Fixed stratified train/test split seed shared by all runs/configurations.")
    parser.add_argument("--configs", nargs="*", default=None,
                        help="Optional subset of configuration names to run.")
    args = parser.parse_args()

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    selected = CONFIGS
    if args.configs:
        unknown = sorted(set(args.configs) - {c["name"] for c in CONFIGS})
        if unknown:
            raise ValueError(f"Unknown configuration(s): {unknown}")
        selected = [c for c in CONFIGS if c["name"] in args.configs]

    with open(output_root / "experiment_configuration.json", "w") as f:
        json.dump({"configs": selected, "seeds": args.seeds, "arguments": vars(args)}, f, indent=2)

    all_results, all_histories = [], []
    for config in selected:
        for seed in args.seeds:
            result, history = train_one_run(config, seed, args, device, output_root)
            all_results.append(result)
            all_histories.extend(history)
            pd.DataFrame(all_results).to_csv(output_root / "architecture_comparison_all_runs.csv", index=False)
            pd.DataFrame(all_histories).to_csv(output_root / "all_training_histories.csv", index=False)

    results_df = pd.DataFrame(all_results)
    metric_cols = ["Accuracy", "Precision", "Recall", "F1", "Kappa", "MCC"]
    summary = results_df.groupby(
        ["Configuration", "Layers", "Embedding", "Heads", "Parameters"],
        as_index=False,
    )[metric_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join(str(part) for part in col if str(part)) if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary = summary.reset_index(drop=True)
    results_df.to_csv(output_root / "architecture_comparison_all_runs.csv", index=False)
    summary.to_csv(output_root / "architecture_comparison_summary.csv", index=False)

    # Paper-friendly mean ± standard deviation table, with values retained as decimals.
    paper_rows = []
    grouped = results_df.groupby("Configuration", sort=False)
    for config in selected:
        group = grouped.get_group(config["name"])
        row = {
            "Configuration": config["name"], "Layers": config["layers"],
            "Embedding": config["dim"], "Heads": config["heads"],
            "Parameters": int(group["Parameters"].iloc[0]),
        }
        for metric in metric_cols:
            row[metric] = f"{group[metric].mean():.4f} ± {group[metric].std(ddof=1):.4f}" if len(group) > 1 else f"{group[metric].mean():.4f} ± 0.0000"
        paper_rows.append(row)
    pd.DataFrame(paper_rows).to_csv(output_root / "architecture_comparison_paper_table.csv", index=False)

    print("\nArchitecture comparison completed.")
    print(f"Results saved in: {output_root}")
    print("NOTE: The script computes new results; it does not hard-code or guarantee the values in a manuscript table.")
    print("Verify the reported parameter counts and mean±SD against these generated CSV files.")


if __name__ == "__main__":
    main()
