# Grad-CAM vs Attention Rollout (NEU-DET)

This experiment compares gradient-based Grad-CAM with CLS-to-patch Transformer attention for the supplied single-block Fuzzy CNN-ViT model.

## Run in Colab

After cloning the repository and mounting the NEU-DET data/checkpoint at the default paths:

```bash
%run experiments/neu_det/gradcam_attention_comparison.py
```

Or provide explicit paths:

```bash
!python experiments/neu_det/gradcam_attention_comparison.py \\
  --train-dir /content/data/NEU-DET/train/images \\
  --checkpoint /content/Fuzzy_CNN_ViT_Results/best_fuzzy_cnn_vit.pth \\
  --output-dir /content/Fuzzy_CNN_ViT_Results/explainability_comparison
```

The script recreates the same stratified 80:20 split of `train/images` using seed 42 and evaluates the held-out subset. It does not retrain the classifier. The checkpoint must be the best validation-loss checkpoint produced by the primary NEU-DET training script.

## Outputs

- `gradcam_attention_per_image.csv`: per-image metrics and labels
- `gradcam_attention_summary.csv`: method-level summary
- `deletion_curve_comparison.png`
- `insertion_curve_comparison.png`
- `explanation_*.png`: original image, maps and overlays

## Methods and interpretation

- Grad-CAM uses gradients of the predicted class score with respect to the last convolution (`model.cnn.conv2[0]`).
- Attention uses the mean CLS-to-patch attention across heads. The network has one Transformer block; residual identity is added and rows are normalized. This is a one-layer residual-aware attention rollout, not multi-layer rollout.
- Perturbation uses zero in normalized `[-1, 1]` input space, corresponding to mid-gray in the original intensity space.
- Lower deletion AUC and higher insertion AUC are generally preferred. Average Drop and Average Increase are computed from the confidence at the full-saliency insertion endpoint and should be interpreted with that definition.
- Execution time is measured separately for map generation. GPU memory is hardware-dependent; run on the same GPU and software environment for a fair comparison.

The script reports measurements; it does not guarantee that the resulting values will match previously reported manuscript numbers. Use the generated outputs from the final checkpoint and report the exact evaluation protocol. Attention maps are not automatically equivalent to causal explanations.
