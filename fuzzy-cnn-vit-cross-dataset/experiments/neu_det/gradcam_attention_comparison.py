"""Grad-CAM vs Transformer Attention comparison for the NEU-DET Fuzzy CNN-ViT.

Uses the exact single-block FuzzyCNNViT architecture and checkpoint format from
neu_det_fuzzy_cnn_vit.py. Runs on the held-out 20% stratified split of the
original train/images directory (288 images for the supplied dataset).

Attention method: CLS-to-patch attention from the model's single transformer
block, averaged over heads. Attention Rollout with residual identity is
mathematically equivalent to residual-aware rollout for this one-block model.

Perturbation baseline: zero in normalized [-1, 1] input space (mid-gray in
original intensity space). Results are descriptive faithfulness metrics, not
proof of causal explanation.
"""

import os
import time
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc

# --------------------------- Defaults ---------------------------
SEED = 42
IMAGE_SIZE = 128
BATCH_SIZE = 1
TRAIN_DIR = "/content/data/NEU-DET/train/images"
CHECKPOINT = "/content/Fuzzy_CNN_ViT_Results/best_fuzzy_cnn_vit.pth"
OUTPUT_DIR = "/content/Fuzzy_CNN_ViT_Results/explainability_comparison"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------- Exact model definition ------------------
class GaussianFuzzyLayer(nn.Module):
    def __init__(self, channels=1, epsilon=1e-5):
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
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2)
        )

    def forward(self, x):
        return self.conv2(self.conv1(x))


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout)
        )

    def forward(self, x):
        normalized = self.norm1(x)
        attention_output, attention_weights = self.attention(
            normalized, normalized, normalized,
            need_weights=True, average_attn_weights=False
        )
        x = x + attention_output
        x = x + self.mlp(self.norm2(x))
        return x, attention_weights


class FuzzyCNNViT(nn.Module):
    def __init__(self, image_size=128, num_classes=6, embed_dim=128,
                 num_heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()
        self.fuzzy_layer = GaussianFuzzyLayer(channels=1)
        self.cnn = CNNFeatureExtractor()
        self.patch_embedding = nn.Conv2d(64, embed_dim, kernel_size=2, stride=2)
        self.patch_grid = image_size // 8
        self.num_patches = self.patch_grid ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.transformer = TransformerBlock(embed_dim, num_heads, mlp_dim, dropout)
        self.classifier = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, num_classes))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        self.last_attention = None

    def forward(self, x):
        x = self.fuzzy_layer(x)
        x = self.cnn(x)
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x, attention_weights = self.transformer(x)
        self.last_attention = attention_weights
        return self.classifier(x[:, 0])


# ------------------------- Utilities ----------------------------
def normalize_map(x):
    x = np.asarray(x, dtype=np.float32)
    x = x - np.min(x)
    return x / (np.max(x) + 1e-8)


def get_gradcam(model, image, target_layer, class_index):
    activations = {}
    gradients = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output
        output.register_hook(lambda grad: gradients.__setitem__("value", grad))

    handle = target_layer.register_forward_hook(forward_hook)
    model.zero_grad(set_to_none=True)
    logits = model(image)
    score = logits[:, class_index].sum()
    score.backward()
    handle.remove()

    acts = activations["value"].detach()
    grads = gradients["value"].detach()
    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * acts).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=image.shape[-2:], mode="bilinear", align_corners=False)
    return normalize_map(cam[0, 0].cpu().numpy()), logits.detach()


def get_attention_map(model, image):
    # The model has one TransformerBlock. Average attention over heads,
    # add residual identity and row-normalize (one-layer attention rollout).
    with torch.no_grad():
        _ = model(image)
        attention = model.last_attention[0].mean(dim=0)  # [tokens, tokens]
        identity = torch.eye(attention.shape[-1], device=attention.device)
        joint = attention + identity
        joint = joint / (joint.sum(dim=-1, keepdim=True) + 1e-8)
        cls_to_patch = joint[0, 1:]
        grid = int(np.sqrt(cls_to_patch.numel()))
        amap = cls_to_patch.reshape(1, 1, grid, grid)
        amap = F.interpolate(amap, size=image.shape[-2:], mode="bilinear", align_corners=False)
    return normalize_map(amap[0, 0].cpu().numpy())


def confidence_for_class(model, image, class_index):
    with torch.no_grad():
        return torch.softmax(model(image), dim=1)[0, class_index].item()


def perturbation_curve(model, image, saliency, class_index, fractions):
    """Return deletion/insertion confidence curves using gray (normalized zero) baseline."""
    flat = saliency.reshape(-1)
    order = np.argsort(flat)[::-1]
    x = image.detach()
    baseline = torch.zeros_like(x)
    total = flat.size
    deletion, insertion = [], []

    with torch.no_grad():
        for fraction in fractions:
            k = int(round(fraction * total))
            mask = np.zeros(total, dtype=np.float32)
            if k > 0:
                mask[order[:k]] = 1.0
            mask = torch.from_numpy(mask.reshape(1, 1, *saliency.shape)).to(x.device)
            mask = F.interpolate(mask, size=x.shape[-2:], mode="nearest")
            deleted = x * (1.0 - mask) + baseline * mask
            inserted = baseline * (1.0 - mask) + x * mask
            deletion.append(confidence_for_class(model, deleted, class_index))
            insertion.append(confidence_for_class(model, inserted, class_index))
    return np.asarray(deletion), np.asarray(insertion)


def save_overlay(image_tensor, gradcam, attention, true_name, pred_name, path):
    image = image_tensor[0, 0].detach().cpu().numpy()
    fig, axes = plt.subplots(1, 5, figsize=(18, 4))
    axes[0].imshow(image, cmap="gray")
    axes[0].set_title(f"Input\nTrue: {true_name}\nPred: {pred_name}")
    axes[1].imshow(gradcam, cmap="jet")
    axes[1].set_title("Grad-CAM")
    axes[2].imshow(image, cmap="gray")
    axes[2].imshow(gradcam, cmap="jet", alpha=0.45)
    axes[2].set_title("Grad-CAM overlay")
    axes[3].imshow(attention, cmap="jet")
    axes[3].set_title("CLS attention rollout")
    axes[4].imshow(image, cmap="gray")
    axes[4].imshow(attention, cmap="jet", alpha=0.45)
    axes[4].set_title("Attention overlay")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main(args):
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    if not os.path.isdir(args.train_dir):
        raise FileNotFoundError(f"NEU-DET train directory not found: {args.train_dir}")
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])
    dataset = datasets.ImageFolder(args.train_dir, transform=transform)
    indices = np.arange(len(dataset))
    targets = np.asarray(dataset.targets)
    _, test_indices = train_test_split(
        indices, test_size=0.20, stratify=targets, random_state=args.seed
    )
    test_dataset = Subset(dataset, test_indices)
    loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=args.workers)
    class_names = dataset.classes

    model = FuzzyCNNViT(image_size=IMAGE_SIZE, num_classes=len(class_names)).to(DEVICE)
    checkpoint = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state, strict=True)
    model.eval()

    fractions = np.linspace(0.0, 1.0, args.curve_steps)
    rows = []
    gradcam_del_curves, gradcam_ins_curves = [], []
    attn_del_curves, attn_ins_curves = [], []
    gradcam_times, attn_times = [], []
    gradcam_drops, gradcam_increases = [], []
    attn_drops, attn_increases = [], []
    saved_examples = 0

    for sample_idx, (image, label) in enumerate(loader):
        image = image.to(DEVICE)
        label_int = int(label.item())

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # First get original prediction; class-specific explanation targets predicted class.
        with torch.no_grad():
            original_logits = model(image)
            predicted = int(original_logits.argmax(dim=1).item())
            original_conf = float(torch.softmax(original_logits, dim=1)[0, predicted].item())
        gradcam_map, _ = get_gradcam(model, image, model.cnn.conv2[0], predicted)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        gradcam_time = time.perf_counter() - t0
        gradcam_peak_mb = (torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)) if DEVICE.type == "cuda" else np.nan

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        attention_map = get_attention_map(model, image)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()
        attention_time = time.perf_counter() - t1
        attention_peak_mb = (torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 2)) if DEVICE.type == "cuda" else np.nan

        g_del, g_ins = perturbation_curve(model, image, gradcam_map, predicted, fractions)
        a_del, a_ins = perturbation_curve(model, image, attention_map, predicted, fractions)
        gradcam_del_curves.append(g_del); gradcam_ins_curves.append(g_ins)
        attn_del_curves.append(a_del); attn_ins_curves.append(a_ins)

        g_auc_del = float(auc(fractions, g_del))
        g_auc_ins = float(auc(fractions, g_ins))
        a_auc_del = float(auc(fractions, a_del))
        a_auc_ins = float(auc(fractions, a_ins))
        g_keep_conf = float(g_ins[-1])
        a_keep_conf = float(a_ins[-1])
        g_drop = max(0.0, (original_conf - g_keep_conf) / (original_conf + 1e-8))
        a_drop = max(0.0, (original_conf - a_keep_conf) / (original_conf + 1e-8))
        g_increase = float(g_keep_conf > original_conf)
        a_increase = float(a_keep_conf > original_conf)
        gradcam_drops.append(g_drop); attn_drops.append(a_drop)
        gradcam_increases.append(g_increase); attn_increases.append(a_increase)
        gradcam_times.append(gradcam_time); attn_times.append(attention_time)

        rows.append({
            "sample_index": sample_idx,
            "true_label": class_names[label_int],
            "predicted_label": class_names[predicted],
            "correct_prediction": int(predicted == label_int),
            "original_confidence": original_conf,
            "GradCAM_deletion_AUC": g_auc_del,
            "GradCAM_insertion_AUC": g_auc_ins,
            "GradCAM_average_drop": g_drop,
            "GradCAM_average_increase": g_increase,
            "GradCAM_execution_time_sec": gradcam_time,
            "GradCAM_peak_GPU_memory_MB": gradcam_peak_mb,
            "Attention_deletion_AUC": a_auc_del,
            "Attention_insertion_AUC": a_auc_ins,
            "Attention_average_drop": a_drop,
            "Attention_average_increase": a_increase,
            "Attention_execution_time_sec": attention_time,
            "Attention_peak_GPU_memory_MB": attention_peak_mb
        })

        if saved_examples < args.visual_examples:
            save_overlay(
                image, gradcam_map, attention_map,
                class_names[label_int], class_names[predicted],
                os.path.join(args.output_dir, f"explanation_{sample_idx:03d}.png")
            )
            saved_examples += 1

        if (sample_idx + 1) % 25 == 0 or sample_idx + 1 == len(test_dataset):
            print(f"Processed {sample_idx + 1}/{len(test_dataset)} test images")

    per_image = pd.DataFrame(rows)
    per_image.to_csv(os.path.join(args.output_dir, "gradcam_attention_per_image.csv"), index=False)

    summary_rows = []
    for method, del_curves, ins_curves, drops, increases, times in [
        ("Grad-CAM", gradcam_del_curves, gradcam_ins_curves, gradcam_drops, gradcam_increases, gradcam_times),
        ("Attention Rollout", attn_del_curves, attn_ins_curves, attn_drops, attn_increases, attn_times)
    ]:
        d = np.mean(np.stack(del_curves), axis=0)
        ins = np.mean(np.stack(ins_curves), axis=0)
        summary_rows.append({
            "Method": method,
            "Deletion_AUC_mean": float(auc(fractions, d)),
            "Insertion_AUC_mean": float(auc(fractions, ins)),
            "Average_Drop_mean": float(np.mean(drops)),
            "Average_Increase_mean": float(np.mean(increases)),
            "Execution_Time_sec_mean": float(np.mean(times)),
            "Execution_Time_sec_std": float(np.std(times, ddof=1)) if len(times) > 1 else 0.0,
            "Number_of_test_images": len(rows)
        })

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(args.output_dir, "gradcam_attention_summary.csv"), index=False)

    # Mean perturbation curves for the paper.
    plt.figure(figsize=(7, 5))
    plt.plot(fractions, np.mean(np.stack(gradcam_del_curves), axis=0), label="Grad-CAM")
    plt.plot(fractions, np.mean(np.stack(attn_del_curves), axis=0), label="Attention Rollout")
    plt.xlabel("Fraction of most salient pixels removed")
    plt.ylabel("Target-class confidence")
    plt.title("Deletion Curves: Grad-CAM vs Attention Rollout")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "deletion_curve_comparison.png"), dpi=300)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(fractions, np.mean(np.stack(gradcam_ins_curves), axis=0), label="Grad-CAM")
    plt.plot(fractions, np.mean(np.stack(attn_ins_curves), axis=0), label="Attention Rollout")
    plt.xlabel("Fraction of most salient pixels inserted")
    plt.ylabel("Target-class confidence")
    plt.title("Insertion Curves: Grad-CAM vs Attention Rollout")
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "insertion_curve_comparison.png"), dpi=300)
    plt.close()

    print("\nExplainability comparison summary:")
    print(summary.to_string(index=False))
    print("\nSaved results to:", args.output_dir)
    print("Note: lower deletion AUC and higher insertion AUC are generally preferred.")
    print("Average Drop/Increase here use confidence after retaining the full salient region (insertion endpoint).")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", default=TRAIN_DIR)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--curve-steps", type=int, default=21)
    parser.add_argument("--visual-examples", type=int, default=12)
    args = parser.parse_args()
    main(args)
