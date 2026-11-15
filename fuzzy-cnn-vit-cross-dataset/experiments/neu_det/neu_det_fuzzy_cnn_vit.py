"""Fuzzy CNN-ViT for NEU-DET steel surface defect classification.

Dataset: NEU-DET (six classes; 1,800 images total).
Split: 1,152 training + 360 validation + 288 testing images.
Run in Google Colab after placing the dataset in the configured directories.
"""


# ================================================================
# FUZZY CNN-ViT FOR NEU-DET STEEL SURFACE DEFECT CLASSIFICATION
# Complete implementation in one cell
# Total dataset: 1800 images
# Training: 1152 | Validation: 360 | Testing: 288
# ================================================================

# ========================= 1. IMPORTS ============================

import os
import copy
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    matthews_corrcoef,
    cohen_kappa_score
)

# ========================= 2. CONFIGURATION ======================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# Dataset paths
train_dir = "/content/data/NEU-DET/train/images"
val_dir = "/content/data/NEU-DET/validation/images"

# Output directory
output_dir = "/content/Fuzzy_CNN_ViT_Results"
os.makedirs(output_dir, exist_ok=True)

# Hyperparameters
IMAGE_SIZE = 128
BATCH_SIZE = 32
NUM_WORKERS = 2

EPOCHS = 200
PATIENCE = 20

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

NUM_CLASSES = 6

# ========================= 3. CHECK DATASET =====================

assert os.path.isdir(train_dir), "Training directory not found."
assert os.path.isdir(val_dir), "Validation directory not found."

base_train = datasets.ImageFolder(train_dir)
base_val = datasets.ImageFolder(val_dir)

class_names = base_train.classes
num_classes = len(class_names)

print("\nClasses:", class_names)
print("Original training images:", len(base_train))
print("Original validation images:", len(base_val))

assert num_classes == 6, "Expected six classes."
assert class_names == base_val.classes, "Class directories mismatch."
assert len(base_train) == 1440, "Expected 1440 training images."
assert len(base_val) == 360, "Expected 360 validation images."

# ========================= 4. TRANSFORMATIONS ====================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=10),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

eval_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# ========================= 5. STRATIFIED SPLITTING ===============

all_indices = np.arange(len(base_train))
all_targets = np.array(base_train.targets)

# 80% of original training folder for training
# 20% of original training folder for testing

train_indices, test_indices = train_test_split(
    all_indices,
    test_size=0.20,
    stratify=all_targets,
    random_state=SEED
)

# Separate dataset objects preserve separate transforms
train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

test_dataset = datasets.ImageFolder(
    train_dir,
    transform=eval_transform
)

# Existing validation directory
val_dataset = datasets.ImageFolder(
    val_dir,
    transform=eval_transform
)

train_data = Subset(train_dataset, train_indices)
test_data = Subset(test_dataset, test_indices)
val_data = val_dataset

# Data loaders
train_loader = DataLoader(
    train_data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

val_loader = DataLoader(
    val_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

test_loader = DataLoader(
    test_data,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

print("\n========== DATASET SUMMARY ==========")
print("Training:", len(train_data))
print("Validation:", len(val_data))
print("Testing:", len(test_data))
print("Total:", len(train_data) + len(val_data) + len(test_data))

assert len(train_data) + len(val_data) + len(test_data) == 1800

# ========================= 6. CLASS DISTRIBUTION =================

distribution = {}

for split_name, indices in [
    ("Training", train_indices),
    ("Validation", np.arange(len(base_val))),
    ("Testing", test_indices)
]:

    if split_name == "Validation":
        labels = np.array(base_val.targets)
    else:
        labels = all_targets[indices]

    distribution[split_name] = np.bincount(
        labels,
        minlength=num_classes
    )

distribution_df = pd.DataFrame(
    distribution,
    index=class_names
)

print("\nClass distribution:")
print(distribution_df)

distribution_df.to_csv(
    os.path.join(output_dir, "class_distribution.csv")
)

distribution_df.plot(
    kind="bar",
    figsize=(10, 5)
)

plt.title("NEU-DET Class Distribution")
plt.xlabel("Defect Class")
plt.ylabel("Number of Images")
plt.xticks(rotation=30)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(
    os.path.join(output_dir, "class_distribution.png"),
    dpi=300
)
plt.show()

# ========================= 7. FUZZY MEMBERSHIP LAYER =============

class GaussianFuzzyLayer(nn.Module):

    def __init__(self, channels=1, epsilon=1e-5):

        super().__init__()

        self.center = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

        self.raw_sigma = nn.Parameter(
            torch.zeros(1, channels, 1, 1)
        )

        self.epsilon = epsilon

    def forward(self, x):

        sigma = F.softplus(self.raw_sigma) + self.epsilon

        membership = torch.exp(
            -((x - self.center) ** 2) /
            (2 * sigma ** 2)
        )

        return membership

# ========================= 8. CNN FEATURE EXTRACTOR ==============

class CNNFeatureExtractor(nn.Module):

    def __init__(self):

        super().__init__()

        self.conv1 = nn.Sequential(

            nn.Conv2d(
                1, 32,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.conv2 = nn.Sequential(

            nn.Conv2d(
                32, 64,
                kernel_size=3,
                padding=1
            ),

            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

    def forward(self, x):

        x = self.conv1(x)
        x = self.conv2(x)

        return x

# ========================= 9. TRANSFORMER BLOCK ==================

class TransformerBlock(nn.Module):

    def __init__(
        self,
        embed_dim=128,
        num_heads=4,
        mlp_dim=512,
        dropout=0.1
    ):

        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)

        self.mlp = nn.Sequential(

            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        normalized = self.norm1(x)

        attention_output, attention_weights = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=True,
            average_attn_weights=False
        )

        x = x + attention_output

        x = x + self.mlp(self.norm2(x))

        return x, attention_weights

# ========================= 10. FUZZY CNN-ViT MODEL ===============

class FuzzyCNNViT(nn.Module):

    def __init__(
        self,
        image_size=128,
        num_classes=6,
        embed_dim=128,
        num_heads=4,
        mlp_dim=512,
        dropout=0.1
    ):

        super().__init__()

        self.fuzzy_layer = GaussianFuzzyLayer(
            channels=1
        )

        self.cnn = CNNFeatureExtractor()

        # 128 -> 64 -> 32 -> 16
        self.patch_embedding = nn.Conv2d(
            in_channels=64,
            out_channels=embed_dim,
            kernel_size=2,
            stride=2
        )

        self.patch_grid = image_size // 8
        self.num_patches = self.patch_grid ** 2

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches + 1,
                embed_dim
            )
        )

        self.transformer = TransformerBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_dim=mlp_dim,
            dropout=dropout
        )

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes)
        )

        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        self.last_attention = None

    def forward(self, x):

        # Fuzzy membership transformation
        x = self.fuzzy_layer(x)

        # CNN feature extraction
        x = self.cnn(x)

        # Patch embedding
        x = self.patch_embedding(x)

        # Convert feature maps into tokens
        x = x.flatten(2).transpose(1, 2)

        batch_size = x.shape[0]

        # CLS token
        cls_tokens = self.cls_token.expand(
            batch_size, -1, -1
        )

        x = torch.cat(
            (cls_tokens, x),
            dim=1
        )

        # Positional embedding
        x = x + self.pos_embedding

        # Transformer
        x, attention_weights = self.transformer(x)

        self.last_attention = attention_weights.detach()

        # Classification
        cls_output = x[:, 0]

        output = self.classifier(cls_output)

        return output

# ========================= 11. MODEL INITIALIZATION ==============

model = FuzzyCNNViT(
    image_size=IMAGE_SIZE,
    num_classes=num_classes,
    embed_dim=128,
    num_heads=4,
    mlp_dim=512,
    dropout=0.1
).to(device)

total_params = sum(
    p.numel() for p in model.parameters()
)

trainable_params = sum(
    p.numel() for p in model.parameters()
    if p.requires_grad
)

print("\nModel initialized.")
print("Total parameters:", total_params)
print("Trainable parameters:", trainable_params)

# Verify output dimensions
sample_images, sample_labels = next(iter(train_loader))

with torch.no_grad():
    sample_output = model(sample_images.to(device))

print("Sample input shape:", sample_images.shape)
print("Sample output shape:", sample_output.shape)

assert sample_output.shape[1] == num_classes

# ========================= 12. LOSS AND OPTIMIZER ================

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=5
)

best_val_loss = float("inf")
best_model_weights = copy.deepcopy(model.state_dict())

early_stop_counter = 0

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": []
}

# ========================= 13. TRAINING FUNCTION =================

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

# ========================= 14. EVALUATION FUNCTION ===============

def evaluate(model, loader, criterion, device):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.inference_mode():

        for images, labels in loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    return (
        running_loss / total,
        correct / total,
        np.array(all_labels),
        np.array(all_predictions)
    )

# ========================= 15. TRAINING LOOP =====================

for epoch in range(EPOCHS):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, val_acc, _, _ = evaluate(
        model,
        val_loader,
        criterion,
        device
    )

    scheduler.step(val_loss)

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    current_lr = optimizer.param_groups[0]["lr"]

    print(
        f"Epoch [{epoch+1}/{EPOCHS}] "
        f"| Train Loss: {train_loss:.4f} "
        f"| Train Acc: {train_acc:.4f} "
        f"| Val Loss: {val_loss:.4f} "
        f"| Val Acc: {val_acc:.4f} "
        f"| LR: {current_lr:.7f}"
    )

    # Best validation checkpoint
    if val_loss < best_val_loss:

        best_val_loss = val_loss

        best_model_weights = copy.deepcopy(
            model.state_dict()
        )

        torch.save(
            {
                "model_state_dict": best_model_weights,
                "class_names": class_names,
                "image_size": IMAGE_SIZE,
                "best_val_loss": best_val_loss
            },
            os.path.join(
                output_dir,
                "best_fuzzy_cnn_vit.pth"
            )
        )

        early_stop_counter = 0

    else:

        early_stop_counter += 1

    if early_stop_counter >= PATIENCE:

        print("\nEarly stopping activated.")
        break

# Restore best model
model.load_state_dict(best_model_weights)

print("\nTraining completed.")
print("Best validation loss:", best_val_loss)

# ========================= 16. TRAINING CURVES ===================

epochs_range = range(1, len(history["train_loss"]) + 1)

# Loss
plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    history["train_loss"],
    label="Training Loss"
)

plt.plot(
    epochs_range,
    history["val_loss"],
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "loss_curve.png"),
    dpi=300
)

plt.show()

# Accuracy
plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    history["train_acc"],
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    history["val_acc"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "accuracy_curve.png"),
    dpi=300
)

plt.show()

# ========================= 17. FINAL TESTING =====================

checkpoint = torch.load(
    os.path.join(
        output_dir,
        "best_fuzzy_cnn_vit.pth"
    ),
    map_location=device,
    weights_only=False
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

test_loss, test_accuracy, y_true, y_pred = evaluate(
    model,
    test_loader,
    criterion,
    device
)

precision = precision_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    average="macro",
    zero_division=0
)

mcc = matthews_corrcoef(y_true, y_pred)

kappa = cohen_kappa_score(y_true, y_pred)

print("\n========================================")
print("          FINAL TEST RESULTS")
print("========================================")

print(f"Test images : {len(y_true)}")
print(f"Test Loss   : {test_loss:.4f}")
print(f"Accuracy    : {test_accuracy:.4f}")
print(f"Precision   : {precision:.4f}")
print(f"Recall      : {recall:.4f}")
print(f"F1-score    : {f1:.4f}")
print(f"MCC         : {mcc:.4f}")
print(f"Cohen Kappa : {kappa:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)

# ========================= 18. CONFUSION MATRIX ==================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes)
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Fuzzy CNN-ViT Confusion Matrix")

plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "confusion_matrix.png"),
    dpi=300
)

plt.show()

# ========================= 19. NORMALIZED CONFUSION MATRIX =======

cm_normalized = confusion_matrix(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    normalize="true"
)

plt.figure(figsize=(9, 7))

sns.heatmap(
    cm_normalized,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    xticklabels=class_names,
    yticklabels=class_names,
    vmin=0,
    vmax=1
)

plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.title("Normalized Confusion Matrix")

plt.xticks(rotation=35, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(output_dir, "normalized_confusion_matrix.png"),
    dpi=300
)

plt.show()

# ========================= 20. ATTENTION VISUALIZATION ===========

def visualize_attention(
    model,
    dataset,
    class_names,
    device,
    save_path
):

    model.eval()

    image, label = dataset[0]

    input_tensor = image.unsqueeze(0).to(device)

    with torch.inference_mode():

        _ = model(input_tensor)

        attention = model.last_attention

    # [batch, heads, tokens, tokens]
    attention = attention[0]

    # Average attention heads
    attention = attention.mean(dim=0)

    # CLS token attention to patches
    cls_attention = attention[0, 1:]

    patch_grid = int(np.sqrt(cls_attention.numel()))

    attention_map = cls_attention.reshape(
        patch_grid,
        patch_grid
    ).cpu().numpy()

    attention_map = (
        attention_map - attention_map.min()
    ) / (
        attention_map.max() -
        attention_map.min() + 1e-8
    )

    image_np = image.squeeze().cpu().numpy()

    fig, axes = plt.subplots(
        1, 3,
        figsize=(14, 4)
    )

    axes[0].imshow(
        image_np,
        cmap="gray"
    )

    axes[0].set_title(
        f"Original Image\nTrue: {class_names[label]}"
    )

    axes[1].imshow(
        attention_map,
        cmap="jet",
        interpolation="bilinear"
    )

    axes[1].set_title(
        "Transformer Attention"
    )

    axes[2].imshow(
        image_np,
        cmap="gray"
    )

    axes[2].imshow(
        attention_map,
        cmap="jet",
        alpha=0.45,
        interpolation="bilinear"
    )

    axes[2].set_title(
        "Attention Overlay"
    )

    for ax in axes:
        ax.axis("off")

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=300
    )

    plt.show()


visualize_attention(
    model,
    test_dataset,
    class_names,
    device,
    os.path.join(
        output_dir,
        "attention_visualization.png"
    )
)

# ========================= 21. SAVE RESULTS ======================

# Metrics
results = {
    "Test_Loss": test_loss,
    "Accuracy": test_accuracy,
    "Precision_Macro": precision,
    "Recall_Macro": recall,
    "F1_Macro": f1,
    "MCC": mcc,
    "Cohen_Kappa": kappa
}

pd.DataFrame([results]).to_csv(
    os.path.join(
        output_dir,
        "final_test_metrics.csv"
    ),
    index=False
)

# Training history
pd.DataFrame(history).to_csv(
    os.path.join(
        output_dir,
        "training_history.csv"
    ),
    index=False
)

# Classification report
report = classification_report(
    y_true,
    y_pred,
    labels=np.arange(num_classes),
    target_names=class_names,
    output_dict=True,
    zero_division=0
)

pd.DataFrame(report).transpose().to_csv(
    os.path.join(
        output_dir,
        "classification_report.csv"
    )
)

# Predictions
pd.DataFrame({
    "True_Label": [class_names[i] for i in y_true],
    "Predicted_Label": [class_names[i] for i in y_pred]
}).to_csv(
    os.path.join(
        output_dir,
        "test_predictions.csv"
    ),
    index=False
)

# Save final model
torch.save(
    model.state_dict(),
    os.path.join(
        output_dir,
        "final_fuzzy_cnn_vit.pth"
    )
)

print("\n========================================")
print("       EXPERIMENT COMPLETED")
print("========================================")

print("Total dataset images:", 1800)
print("Training images:", len(train_data))
print("Validation images:", len(val_data))
print("Testing images:", len(test_data))

print("\nSaved files:")

for filename in sorted(os.listdir(output_dir)):
    print(filename)

print("\nResults directory:", output_dir)



# ================================================================
# 22. EXPLAINABILITY: GRAD-CAM vs TRANSFORMER CLS ATTENTION
# ================================================================
# This section is integrated for the exact FuzzyCNNViT architecture
# defined above. It uses:
#   Grad-CAM target: model.cnn.conv2[0]
#   Attention tensor: model.last_attention
#
# IMPORTANT:
# The supplied architecture contains ONE TransformerBlock.
# Therefore, its attention explanation is single-layer CLS attention,
# not multi-layer attention rollout.

from skimage.metrics import structural_similarity as structural_ssim
from sklearn.metrics import auc as sklearn_auc

EXPLAINABILITY_DIR = os.path.join(
    output_dir, "Explainability_Results"
)
os.makedirs(EXPLAINABILITY_DIR, exist_ok=True)


def normalize_explanation_map(explanation):
    """Convert explanation to per-image normalized [B,1,H,W] maps."""
    if explanation.ndim == 3:
        explanation = explanation.unsqueeze(1)

    if explanation.ndim != 4:
        raise ValueError(
            f"Expected explanation [B,1,H,W], got {tuple(explanation.shape)}"
        )

    if explanation.shape[1] != 1:
        explanation = explanation.mean(dim=1, keepdim=True)

    explanation = torch.relu(explanation)
    batch_size = explanation.shape[0]
    flat = explanation.reshape(batch_size, -1)

    minimum = flat.min(dim=1, keepdim=True).values
    maximum = flat.max(dim=1, keepdim=True).values

    flat = (flat - minimum) / (maximum - minimum + 1e-8)

    return flat.reshape(
        batch_size, 1, explanation.shape[-2], explanation.shape[-1]
    )


def resize_explanation_map(explanation, images):
    return F.interpolate(
        explanation,
        size=images.shape[-2:],
        mode="bilinear",
        align_corners=False
    )


@torch.no_grad()
def target_class_confidence(model, images, labels):
    logits = model(images)
    probabilities = torch.softmax(logits, dim=1)
    return probabilities.gather(
        1, labels.view(-1, 1)
    ).squeeze(1)


def generate_gradcam(model, images, labels, target_layer):
    """Grad-CAM for the selected convolutional layer."""
    captured = {}

    def forward_hook(module, inputs, output):
        captured["activation"] = output

    handle = target_layer.register_forward_hook(forward_hook)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits = model(images)
            activations = captured.get("activation")

            if activations is None:
                raise RuntimeError(
                    "The selected Grad-CAM layer did not produce activations."
                )

            selected_scores = logits.gather(
                1, labels.view(-1, 1)
            ).sum()

            gradients = torch.autograd.grad(
                selected_scores,
                activations,
                retain_graph=False,
                create_graph=False
            )[0]

            channel_weights = gradients.mean(
                dim=(2, 3), keepdim=True
            )

            cam = torch.relu(
                (channel_weights * activations).sum(dim=1, keepdim=True)
            )

        return cam.detach()

    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)


@torch.no_grad()
def generate_transformer_cls_attention(model, images):
    """
    Extract CLS-token-to-patch attention from the supplied single
    TransformerBlock. Expected stored tensor: [B, heads, tokens, tokens].
    """
    _ = model(images)

    attention = model.last_attention

    if attention is None:
        raise RuntimeError(
            "model.last_attention is None. The model forward pass "
            "did not store attention weights."
        )

    if attention.ndim != 4:
        raise ValueError(
            "Expected model.last_attention shape [B, heads, tokens, tokens], "
            f"but received {tuple(attention.shape)}."
        )

    # Average attention across heads: [B, tokens, tokens]
    attention = attention.mean(dim=1)

    # CLS token attending to all image patch tokens
    cls_to_patches = attention[:, 0, 1:]

    patch_count = cls_to_patches.shape[1]
    grid_size = int(np.sqrt(patch_count))

    if grid_size * grid_size != patch_count:
        raise ValueError(
            f"{patch_count} patch tokens cannot be reshaped into a square grid."
        )

    attention_map = cls_to_patches.reshape(
        images.shape[0], 1, grid_size, grid_size
    )

    return attention_map


@torch.no_grad()
def calculate_average_drop_increase(
    model, images, explanations, labels, keep_ratio=0.20
):
    explanations = resize_explanation_map(explanations, images)

    batch_size, _, height, width = images.shape
    flat = explanations.flatten(1)
    total_pixels = height * width
    keep_count = max(1, int(total_pixels * keep_ratio))

    top_indices = torch.topk(
        flat, keep_count, dim=1
    ).indices

    mask = torch.zeros_like(flat)
    mask.scatter_(1, top_indices, 1.0)
    mask = mask.reshape(batch_size, 1, height, width)

    original_confidence = target_class_confidence(
        model, images, labels
    )

    # Keep only the most salient pixels; remaining pixels use zero baseline.
    retained_images = images * mask

    retained_confidence = target_class_confidence(
        model, retained_images, labels
    )

    average_drop = (
        torch.clamp(
            original_confidence - retained_confidence, min=0
        ) / (original_confidence + 1e-8)
    ) * 100.0

    average_increase = (
        retained_confidence > original_confidence
    ).float() * 100.0

    return (
        average_drop.cpu().numpy(),
        average_increase.cpu().numpy()
    )


@torch.no_grad()
def calculate_deletion_insertion_auc(
    model, images, explanations, labels, steps=20
):
    explanations = resize_explanation_map(explanations, images)

    batch_size, _, height, width = images.shape
    total_pixels = height * width

    sorted_indices = torch.argsort(
        explanations.flatten(1),
        dim=1,
        descending=True
    )

    deletion_curve = []
    insertion_curve = []

    # Zero is the baseline in the model's normalized input space.
    baseline = torch.zeros_like(images)

    for step in range(steps + 1):
        number_to_change = int(total_pixels * step / steps)

        mask_flat = torch.zeros(
            batch_size,
            total_pixels,
            device=images.device,
            dtype=images.dtype
        )

        if number_to_change > 0:
            mask_flat.scatter_(
                1,
                sorted_indices[:, :number_to_change],
                1.0
            )

        mask = mask_flat.reshape(
            batch_size, 1, height, width
        )

        # Deletion: remove salient pixels progressively.
        deleted_images = images * (1.0 - mask) + baseline * mask

        # Insertion: restore salient pixels progressively.
        inserted_images = baseline * (1.0 - mask) + images * mask

        deletion_confidence = target_class_confidence(
            model, deleted_images, labels
        )
        insertion_confidence = target_class_confidence(
            model, inserted_images, labels
        )

        deletion_curve.append(deletion_confidence.cpu().numpy())
        insertion_curve.append(insertion_confidence.cpu().numpy())

    deletion_curve = np.stack(deletion_curve, axis=0)
    insertion_curve = np.stack(insertion_curve, axis=0)

    x_axis = np.linspace(0.0, 1.0, steps + 1)

    deletion_auc = np.array([
        sklearn_auc(x_axis, deletion_curve[:, i])
        for i in range(batch_size)
    ])

    insertion_auc = np.array([
        sklearn_auc(x_axis, insertion_curve[:, i])
        for i in range(batch_size)
    ])

    return deletion_auc, insertion_auc


def calculate_stability_ssim(
    original_explanation, perturbed_explanation
):
    original = normalize_explanation_map(
        original_explanation
    ).detach().cpu().numpy()

    perturbed = normalize_explanation_map(
        perturbed_explanation
    ).detach().cpu().numpy()

    scores = []

    for index in range(original.shape[0]):
        scores.append(
            structural_ssim(
                original[index, 0],
                perturbed[index, 0],
                data_range=1.0
            )
        )

    return np.asarray(scores, dtype=float)


def evaluate_explainability_comparison(
    model,
    loader,
    device,
    output_directory,
    keep_ratio=0.20,
    deletion_steps=20,
    noise_std=0.05,
    seed=42
):
    os.makedirs(output_directory, exist_ok=True)

    # Exact layer name from the uploaded CNNFeatureExtractor class.
    gradcam_target_layer = model.cnn.conv2[0]

    if not isinstance(gradcam_target_layer, nn.Conv2d):
        raise TypeError(
            "model.cnn.conv2[0] is not nn.Conv2d. "
            "Check the CNNFeatureExtractor definition."
        )

    print("Grad-CAM target layer:", gradcam_target_layer)
    print("Stored attention tensor:", "model.last_attention")

    methods = [
        (
            "Grad-CAM",
            lambda x, y: generate_gradcam(
                model, x, y, gradcam_target_layer
            )
        ),
        (
            "Transformer CLS Attention",
            lambda x, y: generate_transformer_cls_attention(
                model, x
            )
        )
    ]

    result_rows = []
    global_image_index = 0

    model.eval()

    for method_name, explanation_function in methods:
        print(f"\nEvaluating {method_name} ...")
        image_index = 0

        for batch_index, (images, labels) in enumerate(loader):
            images = images.to(device)
            labels = labels.to(device)

            # Generate and normalize explanation maps.
            explanations = explanation_function(images, labels)
            explanations = normalize_explanation_map(explanations)

            # Faithfulness metrics.
            average_drop, average_increase = (
                calculate_average_drop_increase(
                    model,
                    images,
                    explanations,
                    labels,
                    keep_ratio=keep_ratio
                )
            )

            deletion_auc, insertion_auc = (
                calculate_deletion_insertion_auc(
                    model,
                    images,
                    explanations,
                    labels,
                    steps=deletion_steps
                )
            )

            # Perturb inputs with deterministic Gaussian noise.
            generator = torch.Generator(device=device)
            generator.manual_seed(seed + batch_index)

            noise = torch.randn(
                images.shape,
                generator=generator,
                device=device,
                dtype=images.dtype
            ) * noise_std

            perturbed_images = torch.clamp(
                images + noise, min=-1.0, max=1.0
            )

            perturbed_explanations = explanation_function(
                perturbed_images, labels
            )

            stability = calculate_stability_ssim(
                explanations,
                perturbed_explanations
            )

            for i in range(images.shape[0]):
                result_rows.append({
                    "Method": method_name,
                    "Image_Index": image_index + i,
                    "True_Class_Index": int(labels[i].item()),
                    "True_Class": class_names[int(labels[i].item())],
                    "Average_Drop_pct": float(average_drop[i]),
                    "Average_Increase_pct": float(average_increase[i]),
                    "Deletion_AUC": float(deletion_auc[i]),
                    "Insertion_AUC": float(insertion_auc[i]),
                    "Stability_SSIM": float(stability[i])
                })

            image_index += images.shape[0]

            print(
                f"  Batch {batch_index + 1}/{len(loader)} "
                f"completed ({image_index} images)"
            )

        if image_index == 0:
            raise RuntimeError("The test DataLoader returned no images.")

        if global_image_index == 0:
            global_image_index = image_index
        elif image_index != global_image_index:
            raise RuntimeError(
                "The two methods were not evaluated on the same number of images."
            )

    per_image_df = pd.DataFrame(result_rows)

    per_image_file = os.path.join(
        output_directory, "explainability_per_image.csv"
    )
    per_image_df.to_csv(per_image_file, index=False)

    metrics = [
        "Average_Drop_pct",
        "Average_Increase_pct",
        "Deletion_AUC",
        "Insertion_AUC",
        "Stability_SSIM"
    ]

    summary_rows = []

    for method_name in per_image_df["Method"].unique():
        subset = per_image_df[
            per_image_df["Method"] == method_name
        ]

        row = {
            "Method": method_name,
            "Number_of_images": len(subset)
        }

        for metric in metrics:
            row[metric + "_Mean"] = subset[metric].mean()
            row[metric + "_Std"] = subset[metric].std(ddof=1)

        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)

    summary_file = os.path.join(
        output_directory, "explainability_summary.csv"
    )
    summary_df.to_csv(summary_file, index=False)

    print("\n========== EXPLAINABILITY RESULTS ==========")
    print(summary_df.round(6).to_string(index=False))

    print("\nFiles saved:")
    print(per_image_file)
    print(summary_file)

    for path in [per_image_file, summary_file]:
        if not os.path.isfile(path) or os.path.getsize(path) == 0:
            raise RuntimeError(f"Output file was not created correctly: {path}")

    print("\nVerified output files:")
    for filename in sorted(os.listdir(output_directory)):
        print(filename)

    return per_image_df, summary_df


# ================================================================
# 23. RUN EXPLAINABILITY EVALUATION
# ================================================================

# The original script has already:
#   1. Created test_loader.
#   2. Loaded best_fuzzy_cnn_vit.pth.
#   3. Restored the best model weights.
#
# Run the evaluation on the 288-image test set.

explainability_per_image, explainability_summary = (
    evaluate_explainability_comparison(
        model=model,
        loader=test_loader,
        device=device,
        output_directory=EXPLAINABILITY_DIR,
        keep_ratio=0.20,
        deletion_steps=20,
        noise_std=0.05,
        seed=SEED
    )
)

# Display results in the notebook.
display(explainability_summary)
display(explainability_per_image.head())

print("\nExplainability output directory:", EXPLAINABILITY_DIR)
