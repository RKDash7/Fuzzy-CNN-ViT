import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import auc
from skimage.metrics import structural_similarity as ssim
import os
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------------------------------------
# 0. Recreate Model, DataLoader and Load Saved Weights
# --------------------------------------------------
IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 2
TRAIN_DIR = "/content/data/NEU-DET/train/images"

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

eval_train = datasets.ImageFolder(TRAIN_DIR, transform=eval_transform)
num_classes = len(eval_train.classes)

all_indices = np.arange(len(eval_train))
all_targets = np.array(eval_train.targets)

_, test_indices = train_test_split(
    all_indices,
    test_size=0.20,
    stratify=all_targets,
    random_state=42
)

test_data = Subset(eval_train, test_indices)
test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

model = FuzzyCNNViT(num_classes=num_classes).to(device)
checkpoint_path = os.path.join(OUTPUT_DIR, "best_model_run_1_seed_42.pth")
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"Successfully loaded best model from: {checkpoint_path}")
else:
    print("Warning: Saved model weights not found. Using randomly initialized model.")
model.eval()

# --------------------------------------------------
# 1. Normalize explanation maps
# --------------------------------------------------
def normalize_explanation(explanation):
    explanation = torch.clamp(explanation, min=0)
    B = explanation.shape[0]
    explanation = explanation.reshape(B, -1)
    minimum = explanation.min(dim=1, keepdim=True)[0]
    maximum = explanation.max(dim=1, keepdim=True)[0]
    explanation = (explanation - minimum) / (maximum - minimum + 1e-8)
    return explanation.reshape(
        B, 1, int(np.sqrt(explanation.shape[1])),
        int(np.sqrt(explanation.shape[1]))
    )

# --------------------------------------------------
# 2. Prediction confidence
# --------------------------------------------------
@torch.no_grad()
def get_confidence(model, images, targets):
    outputs = model(images)
    probabilities = torch.softmax(outputs, dim=1)
    confidence = probabilities.gather(1, targets.view(-1, 1)).squeeze(1)
    return confidence

# --------------------------------------------------
# 3. Average Drop and Average Increase
# --------------------------------------------------
@torch.no_grad()
def average_drop_increase(
    model, images, explanations, targets,
    keep_ratio=0.2
):
    B, C, H, W = images.shape
    original_conf = get_confidence(model, images, targets)
    explanations = F.interpolate(
        explanations,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )
    flat_exp = explanations.flatten(1)
    k = max(1, int(H * W * keep_ratio))
    indices = torch.topk(flat_exp, k, dim=1).indices
    masks = torch.zeros_like(flat_exp)
    masks.scatter_(1, indices, 1)
    masks = masks.reshape(B, 1, H, W)

    retained_images = images * masks
    retained_conf = get_confidence(model, retained_images, targets)

    removed_images = images * (1 - masks)
    removed_conf = get_confidence(model, removed_images, targets)

    average_drop = (
        torch.clamp(original_conf - removed_conf, min=0)
        / (original_conf + 1e-8)
    ).mean().item() * 100

    average_increase = (
        retained_conf > original_conf
    ).float().mean().item() * 100

    return average_drop, average_increase

# --------------------------------------------------
# 4. Deletion and Insertion AUC
# --------------------------------------------------
@torch.no_grad()
def deletion_insertion_auc(
    model, images, explanations, targets,
    steps=20
):
    B, C, H, W = images.shape
    explanations = F.interpolate(
        explanations,
        size=(H, W),
        mode="bilinear",
        align_corners=False
    )
    flat_exp = explanations.flatten(1)
    sorted_indices = torch.argsort(flat_exp, dim=1, descending=True)
    total_pixels = H * W
    baseline = torch.zeros_like(images)

    deletion_curve = []
    insertion_curve = []

    for step in range(steps + 1):
        fraction = step / steps
        k = int(total_pixels * fraction)
        mask = torch.zeros(B, total_pixels, device=images.device)
        if k > 0:
            selected = sorted_indices[:, :k]
            mask.scatter_(1, selected, 1)
        mask = mask.reshape(B, 1, H, W)

        deleted = images * (1 - mask) + baseline * mask
        inserted = baseline * (1 - mask) + images * mask

        deletion_conf = get_confidence(model, deleted, targets).mean().item()
        insertion_conf = get_confidence(model, inserted, targets).mean().item()

        deletion_curve.append(deletion_conf)
        insertion_curve.append(insertion_conf)

    x = np.linspace(0, 1, steps + 1)
    deletion_auc = auc(x, deletion_curve)
    insertion_auc = auc(x, insertion_curve)

    return deletion_auc, insertion_auc

# --------------------------------------------------
# 5. Explanation stability using SSIM
# --------------------------------------------------
def explanation_stability(explanation_original, explanation_perturbed):
    original = explanation_original.detach().cpu().numpy()
    perturbed = explanation_perturbed.detach().cpu().numpy()
    scores = []
    for i in range(original.shape[0]):
        a = original[i].squeeze()
        b = perturbed[i].squeeze()
        score = ssim(a, b, data_range=1.0)
        scores.append(score)
    return np.mean(scores)

# --------------------------------------------------
# 6. Explanation Functions Definition
# --------------------------------------------------
def gradcam_fn(model, images, labels):
    model.zero_grad()
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.cnn[4]
    h1 = target_layer.register_forward_hook(forward_hook)
    h2 = target_layer.register_full_backward_hook(backward_hook)

    outputs = model(images)
    score = outputs.gather(1, labels.view(-1, 1)).squeeze(1)
    score.backward(torch.ones_like(score))

    h1.remove()
    h2.remove()

    acts = activations[0]
    grads = gradients[0]
    weights = torch.mean(grads, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * acts, dim=1, keepdim=True)
    return F.relu(cam)

def attention_fn(model, images, labels):
    with torch.no_grad():
        _ = model(images)
    attn = model.transformer.last_attention
    if attn.dim() == 4:
        attn = attn.mean(dim=1)
    cls_attn = attn[:, 0, 1:]
    B, num_patches = cls_attn.shape
    grid_size = int(np.sqrt(num_patches))
    return cls_attn.reshape(B, 1, grid_size, grid_size)

# --------------------------------------------------
# 7. Complete evaluation
# --------------------------------------------------
def evaluate_explainability(model, test_loader, gradcam_fn, attention_fn):
    results = []
    for method, explain_fn in [
        ("Grad-CAM", gradcam_fn),
        ("Attention Rollout", attention_fn)
    ]:
        all_ad = []
        all_ai = []
        all_deletion = []
        all_insertion = []
        all_stability = []

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            explanation = explain_fn(model, images, labels)
            explanation = normalize_explanation(explanation)

            ad, ai = average_drop_increase(model, images, explanation, labels)
            all_ad.append(ad)
            all_ai.append(ai)

            d_auc, i_auc = deletion_insertion_auc(model, images, explanation, labels)
            all_deletion.append(d_auc)
            all_insertion.append(i_auc)

            perturbed_images = torch.clamp(
                images + torch.randn_like(images) * 0.05,
                -1, 1
            )
            with torch.enable_grad():
                perturbed_explanation = explain_fn(model, perturbed_images, labels)
            perturbed_explanation = normalize_explanation(perturbed_explanation)

            stability = explanation_stability(explanation, perturbed_explanation)
            all_stability.append(stability)

        results.append({
            "Method": method,
            "Average Drop (%)": np.mean(all_ad),
            "Average Increase (%)": np.mean(all_ai),
            "Deletion AUC": np.mean(all_deletion),
            "Insertion AUC": np.mean(all_insertion),
            "Stability (SSIM)": np.mean(all_stability)
        })
    return pd.DataFrame(results)

results_df = evaluate_explainability(
    model,
    test_loader,
    gradcam_fn,
    attention_fn
)

print("\nExplainability Evaluation Results")
print(results_df.round(4))

results_df.to_csv(
    "NEU_DET_Explainability_Metrics.csv",
    index=False
)
print("\nResults saved successfully.")
