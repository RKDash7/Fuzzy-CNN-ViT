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
        num_heads=8,
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
        num_heads=8,
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
    num_heads=8,
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
