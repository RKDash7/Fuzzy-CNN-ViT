"""
Grad-CAM vs. CLS-attention explainability evaluation for NEU-DET.

This is an evaluation module, not a model-definition file. Construct your
FuzzyCNNViT model, load the trained checkpoint, and create test_loader using
the SAME split/preprocessing as the training experiment. Then call
run_explainability_evaluation(...).

Important:
- Average Drop is computed from the image retaining salient pixels only.
- Average Increase uses the same retained image.
- Metrics are accumulated per image, so a smaller final batch is weighted
  correctly.
- The attention function below extracts CLS-to-patch attention from the
  model's last transformer layer. If only one layer is used, call it
  "CLS Attention" rather than multi-layer Attention Rollout.
- The default Grad-CAM target layer assumes model.cnn[4], as in the supplied
  implementation. Change target_layer_getter if your architecture differs.
"""

from __future__ import annotations

import os
import random
from typing import Callable, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import auc
from skimage.metrics import structural_similarity as ssim


def set_reproducibility(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def get_confidence(model, images: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the softmax probability assigned to each image's target class."""
    logits = model(images)
    probabilities = torch.softmax(logits, dim=1)
    return probabilities.gather(1, targets.view(-1, 1)).squeeze(1)


def normalize_explanation(explanation: torch.Tensor) -> torch.Tensor:
    """Per-image min-max normalization; returns [B,1,h,w]."""
    if explanation.ndim == 3:
        explanation = explanation.unsqueeze(1)
    if explanation.ndim != 4:
        raise ValueError(f"Expected explanation [B,1,H,W], got {tuple(explanation.shape)}")
    if explanation.shape[1] != 1:
        explanation = explanation.mean(dim=1, keepdim=True)

    explanation = torch.clamp(explanation, min=0)
    batch = explanation.shape[0]
    flat = explanation.reshape(batch, -1)
    minimum = flat.min(dim=1, keepdim=True).values
    maximum = flat.max(dim=1, keepdim=True).values
    flat = (flat - minimum) / (maximum - minimum + 1e-8)
    return flat.reshape(batch, 1, explanation.shape[-2], explanation.shape[-1])


def resize_explanations(explanations: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
    return F.interpolate(
        explanations,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )


@torch.no_grad()
def average_drop_increase_per_image(
    model,
    images: torch.Tensor,
    explanations: torch.Tensor,
    targets: torch.Tensor,
    keep_ratio: float = 0.20,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute standard Average Drop and Average Increase for each image.

    The retained image contains the top-ranked salient pixels; all other
    pixels are replaced by the zero baseline in normalized input space.
    """
    if not 0 < keep_ratio <= 1:
        raise ValueError("keep_ratio must be in (0, 1].")

    batch, _, height, width = images.shape
    explanations = resize_explanations(explanations, images)
    flat = explanations.flatten(1)
    k = max(1, int(height * width * keep_ratio))

    top_indices = torch.topk(flat, k, dim=1).indices
    mask = torch.zeros_like(flat)
    mask.scatter_(1, top_indices, 1.0)
    mask = mask.reshape(batch, 1, height, width)

    original_conf = get_confidence(model, images, targets)
    retained_images = images * mask
    retained_conf = get_confidence(model, retained_images, targets)

    average_drop = (
        torch.clamp(original_conf - retained_conf, min=0)
        / (original_conf + 1e-8)
    ) * 100.0

    average_increase = (retained_conf > original_conf).float() * 100.0

    return (
        average_drop.detach().cpu().numpy(),
        average_increase.detach().cpu().numpy(),
    )


@torch.no_grad()
def deletion_insertion_auc_per_image(
    model,
    images: torch.Tensor,
    explanations: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 20,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute deletion and insertion AUC separately for every image."""
    if steps < 1:
        raise ValueError("steps must be >= 1.")

    batch, _, height, width = images.shape
    explanations = resize_explanations(explanations, images)
    flat = explanations.flatten(1)
    sorted_indices = torch.argsort(flat, dim=1, descending=True)

    total_pixels = height * width
    baseline = torch.zeros_like(images)
    deletion_curve = []
    insertion_curve = []

    for step in range(steps + 1):
        k = int(total_pixels * step / steps)
        mask_flat = torch.zeros(
            batch, total_pixels, device=images.device, dtype=images.dtype
        )
        if k > 0:
            mask_flat.scatter_(1, sorted_indices[:, :k], 1.0)
        mask = mask_flat.reshape(batch, 1, height, width)

        deleted = images * (1.0 - mask) + baseline * mask
        inserted = baseline * (1.0 - mask) + images * mask

        deletion_curve.append(get_confidence(model, deleted, targets))
        insertion_curve.append(get_confidence(model, inserted, targets))

    # [steps+1, batch] -> calculate AUC for each image independently.
    deletion_values = torch.stack(deletion_curve).cpu().numpy()
    insertion_values = torch.stack(insertion_curve).cpu().numpy()
    x = np.linspace(0.0, 1.0, steps + 1)

    deletion_auc = np.array(
        [auc(x, deletion_values[:, i]) for i in range(batch)], dtype=float
    )
    insertion_auc = np.array(
        [auc(x, insertion_values[:, i]) for i in range(batch)], dtype=float
    )
    return deletion_auc, insertion_auc


def explanation_stability_per_image(
    explanation_original: torch.Tensor,
    explanation_perturbed: torch.Tensor,
) -> np.ndarray:
    """SSIM between original and noise-perturbed explanation maps."""
    original = normalize_explanation(explanation_original).detach().cpu().numpy()
    perturbed = normalize_explanation(explanation_perturbed).detach().cpu().numpy()

    scores = []
    for i in range(original.shape[0]):
        a = original[i, 0]
        b = perturbed[i, 0]
        scores.append(ssim(a, b, data_range=1.0))
    return np.asarray(scores, dtype=float)


def gradcam_fn(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    target_layer: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    Grad-CAM implementation.

    By default, uses model.cnn[4], matching the code supplied in the chat.
    Pass target_layer explicitly if your model uses a different final CNN layer.
    """
    if target_layer is None:
        try:
            target_layer = model.cnn[4]
        except Exception as exc:
            raise AttributeError(
                "Could not access model.cnn[4]. Pass the correct target_layer."
            ) from exc

    activations = {}

    def forward_hook(_module, _inputs, output):
        activations["value"] = output

    handle = target_layer.register_forward_hook(forward_hook)
    model.zero_grad(set_to_none=True)

    try:
        with torch.enable_grad():
            logits = model(images)
            scores = logits.gather(1, labels.view(-1, 1)).sum()
            feature_maps = activations.get("value")
            if feature_maps is None:
                raise RuntimeError("Grad-CAM target layer did not produce activations.")
            gradients = torch.autograd.grad(
                outputs=scores,
                inputs=feature_maps,
                retain_graph=False,
                create_graph=False,
            )[0]

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * feature_maps).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        return cam.detach()
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


@torch.no_grad()
def cls_attention_fn(model, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Extract CLS-token-to-patch attention from model.transformer.last_attention.

    This is last-layer CLS attention. It is not multi-layer Attention Rollout
    unless the model's implementation has already combined layer attentions.
    """
    _ = model(images)
    attention = model.transformer.last_attention

    if attention.ndim == 4:       # [B, heads, tokens, tokens]
        attention = attention.mean(dim=1)
    elif attention.ndim != 3:     # [B, tokens, tokens]
        raise ValueError(f"Unexpected attention shape: {tuple(attention.shape)}")

    cls_to_patches = attention[:, 0, 1:]
    batch, patch_count = cls_to_patches.shape
    grid = int(np.sqrt(patch_count))
    if grid * grid != patch_count:
        raise ValueError(
            f"Patch count {patch_count} is not a square; provide a grid-aware reshape."
        )
    return cls_to_patches.reshape(batch, 1, grid, grid)


def _weighted_summary(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    return float(np.mean(arr)), float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def evaluate_explainability(
    model,
    test_loader,
    device: torch.device,
    gradcam_target_layer: Optional[torch.nn.Module] = None,
    keep_ratio: float = 0.20,
    deletion_insertion_steps: int = 20,
    noise_std: float = 0.05,
    seed: int = 42,
    output_dir: str = "explainability_results",
    method_names: tuple[str, str] = ("Grad-CAM", "CLS Attention"),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate both explanation methods and save per-image and summary CSV files.

    Returns:
        per_image_df, summary_df
    """
    os.makedirs(output_dir, exist_ok=True)
    set_reproducibility(seed)
    model.eval()

    methods: list[tuple[str, Callable]] = [
        (
            method_names[0],
            lambda m, x, y: gradcam_fn(
                m, x, y, target_layer=gradcam_target_layer
            ),
        ),
        (method_names[1], cls_attention_fn),
    ]

    per_image_rows = []

    for method_name, explain_fn in methods:
        sample_index = 0

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            explanation = explain_fn(model, images, labels)
            explanation = normalize_explanation(explanation)

            ad, ai = average_drop_increase_per_image(
                model, images, explanation, labels, keep_ratio=keep_ratio
            )
            deletion, insertion = deletion_insertion_auc_per_image(
                model, images, explanation, labels,
                steps=deletion_insertion_steps,
            )

            # Use a reproducible perturbation for each batch. The same test
            # images are traversed in the same order for both methods.
            generator = torch.Generator(device=images.device)
            generator.manual_seed(seed + sample_index)
            noise = torch.randn(
                images.shape,
                generator=generator,
                device=images.device,
                dtype=images.dtype,
            ) * noise_std
            perturbed_images = torch.clamp(images + noise, -1.0, 1.0)

            perturbed_explanation = explain_fn(
                model, perturbed_images, labels
            )
            stability = explanation_stability_per_image(
                explanation, perturbed_explanation
            )

            for j in range(images.shape[0]):
                per_image_rows.append({
                    "Method": method_name,
                    "Test_Image_Index": sample_index + j,
                    "Target_Class": int(labels[j].item()),
                    "Average_Drop_pct": float(ad[j]),
                    "Average_Increase_pct": float(ai[j]),
                    "Deletion_AUC": float(deletion[j]),
                    "Insertion_AUC": float(insertion[j]),
                    "Stability_SSIM": float(stability[j]),
                })

            sample_index += images.shape[0]

    per_image_df = pd.DataFrame(per_image_rows)
    per_image_path = os.path.join(output_dir, "per_image_results.csv")
    per_image_df.to_csv(per_image_path, index=False)

    metric_columns = [
        "Average_Drop_pct",
        "Average_Increase_pct",
        "Deletion_AUC",
        "Insertion_AUC",
        "Stability_SSIM",
    ]
    summary_rows = []
    for method_name in [item[0] for item in methods]:
        subset = per_image_df[per_image_df["Method"] == method_name]
        row = {
            "Method": method_name,
            "Number_of_test_images": len(subset),
        }
        for metric in metric_columns:
            mean, std = _weighted_summary(subset[metric].tolist())
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = std
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(output_dir, "explainability_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\nExplainability summary (mean ± standard deviation)")
    print(summary_df.round(6).to_string(index=False))
    print(f"\nSaved per-image results: {per_image_path}")
    print(f"Saved summary: {summary_path}")

    return per_image_df, summary_df


# ---------------------------------------------------------------------------
# USAGE (run this from your existing NEU-DET experiment after model creation)
# ---------------------------------------------------------------------------
# model = FuzzyCNNViT(num_classes=num_classes).to(device)
# model.load_state_dict(torch.load(checkpoint_path, map_location=device))
# model.eval()
#
# per_image_df, summary_df = evaluate_explainability(
#     model=model,
#     test_loader=test_loader,
#     device=device,
#     gradcam_target_layer=model.cnn[4],  # verify this is the intended final CNN conv layer
#     keep_ratio=0.20,
#     deletion_insertion_steps=20,
#     noise_std=0.05,
#     output_dir="NEU_DET_Explainability_Results",
#     # Use ("Grad-CAM", "Attention Rollout") ONLY if your model's attention
#     # implementation actually performs multi-layer rollout.
#     method_names=("Grad-CAM", "CLS Attention"),
# )
