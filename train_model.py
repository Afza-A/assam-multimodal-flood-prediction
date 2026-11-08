
import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.optim import Adam

from sar_encoder import SAREncoder
from dem_encoder import DEMEncoder
from rainfall_encoder import RainfallEncoder

from fusion_decoder import MultimodalFloodModel
from pre_flood_dataset import get_dataloaders


# =========================================================
# 1. CONFIGURATION
# =========================================================

SEED = 42

BATCH_SIZE = 4
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 8

CHECKPOINT_PATH = "best_flood_model.pth"
HISTORY_PATH = "training_history.csv"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# =========================================================
# 2. REPRODUCIBILITY
# =========================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("=" * 55)
print("MULTIMODAL FLOOD MODEL TRAINING")
print("=" * 55)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3), 2
        ),
        "GB"
    )


# =========================================================
# 3. LOSS FUNCTION: BCE + DICE
# =========================================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probabilities = torch.sigmoid(logits)

        probabilities = probabilities.reshape(-1)
        targets = targets.reshape(-1)

        intersection = (probabilities * targets).sum()

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            probabilities.sum()
            + targets.sum()
            + self.smooth
        )

        return 1.0 - dice


class CombinedLoss(nn.Module):

    def __init__(self):
        super().__init__()

        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        total_loss = 0.5 * bce_loss + 0.5 * dice_loss

        return total_loss


# =========================================================
# 4. METRICS
# =========================================================

def calculate_metrics(
    total_tp,
    total_fp,
    total_fn,
    epsilon=1e-8
):

    precision = total_tp / (
        total_tp + total_fp + epsilon
    )

    recall = total_tp / (
        total_tp + total_fn + epsilon
    )

    dice = (2 * total_tp) / (
        2 * total_tp + total_fp + total_fn + epsilon
    )

    iou = total_tp / (
        total_tp + total_fp + total_fn + epsilon
    )

    return {
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou
    }


# =========================================================
# 5. TRAINING FUNCTION
# =========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for batch in loader:

        sar = batch["sar"].to(DEVICE)
        dem = batch["dem"].to(DEVICE)
        rainfall = batch["rainfall"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        optimizer.zero_grad(set_to_none=True)

        outputs = model(
            sar,
            dem,
            rainfall
        )

        logits = outputs["logits"]

        loss = criterion(logits, mask)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        # Training metrics at threshold 0.5.
        predictions = (
            torch.sigmoid(logits) >= 0.5
        ).float()

        total_tp += (
            predictions * mask
        ).sum().item()

        total_fp += (
            predictions * (1 - mask)
        ).sum().item()

        total_fn += (
            (1 - predictions) * mask
        ).sum().item()

    average_loss = running_loss / len(loader)

    metrics = calculate_metrics(
        total_tp,
        total_fp,
        total_fn
    )

    return average_loss, metrics


# =========================================================
# 6. VALIDATION FUNCTION
# =========================================================

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0.0

    total_tp = 0
    total_fp = 0
    total_fn = 0

    with torch.no_grad():

        for batch in loader:

            sar = batch["sar"].to(DEVICE)
            dem = batch["dem"].to(DEVICE)
            rainfall = batch["rainfall"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            outputs = model(
                sar,
                dem,
                rainfall
            )

            logits = outputs["logits"]

            loss = criterion(logits, mask)

            running_loss += loss.item()

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).float()

            total_tp += (
                predictions * mask
            ).sum().item()

            total_fp += (
                predictions * (1 - mask)
            ).sum().item()

            total_fn += (
                (1 - predictions) * mask
            ).sum().item()

    average_loss = running_loss / len(loader)

    metrics = calculate_metrics(
        total_tp,
        total_fp,
        total_fn
    )

    return average_loss, metrics


# =========================================================
# 7. LOAD DATA
# =========================================================

print("\nLoading datasets...")

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=BATCH_SIZE
)

print("\nTraining events:", len(train_loader.dataset))
print("Validation events:", len(val_loader.dataset))
print("Test events reserved:", len(test_loader.dataset))

# IMPORTANT:
# test_loader is intentionally not used during training.


# =========================================================
# 8. INITIALIZE MODEL
# =========================================================

sar_encoder = SAREncoder()
dem_encoder = DEMEncoder()
rainfall_encoder = RainfallEncoder()

model = MultimodalFloodModel(
    sar_encoder,
    dem_encoder,
    rainfall_encoder
).to(DEVICE)

print("\nModel initialized on:", DEVICE)


# =========================================================
# 9. OPTIMIZER AND LOSS
# =========================================================

criterion = CombinedLoss()

optimizer = Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# =========================================================
# 10. TRAINING LOOP
# =========================================================

best_val_dice = -1.0
epochs_without_improvement = 0

history = []

if DEVICE.type == "cuda":
    torch.cuda.reset_peak_memory_stats(DEVICE)

print("\nStarting joint end-to-end training...")
print("=" * 55)

for epoch in range(EPOCHS):

    train_loss, train_metrics = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer
    )

    val_loss, val_metrics = validate(
        model,
        val_loader,
        criterion
    )

    current_val_dice = val_metrics["dice"]

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Dice: {train_metrics['dice']:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Dice: {val_metrics['dice']:.4f} | "
        f"Val IoU: {val_metrics['iou']:.4f}"
    )

    print(
        f"Val Precision: {val_metrics['precision']:.4f} | "
        f"Val Recall: {val_metrics['recall']:.4f}"
    )

    # Save epoch history.
    history.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_dice": train_metrics["dice"],
        "val_loss": val_loss,
        "val_dice": val_metrics["dice"],
        "val_iou": val_metrics["iou"],
        "val_precision": val_metrics["precision"],
        "val_recall": val_metrics["recall"]
    })

    # -----------------------------------------------------
    # SAVE BEST CHECKPOINT
    # -----------------------------------------------------

    if current_val_dice > best_val_dice:

        best_val_dice = current_val_dice

        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_dice": best_val_dice,
                "normalization_file": "normalization_stats.json"
            },
            CHECKPOINT_PATH
        )

        print(
            "Best model checkpoint saved!",
            f"Validation Dice: {best_val_dice:.4f}"
        )

    else:

        epochs_without_improvement += 1

        print(
            "No validation improvement:",
            epochs_without_improvement,
            "/", PATIENCE
        )

    # -----------------------------------------------------
    # EARLY STOPPING
    # -----------------------------------------------------

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# =========================================================
# 11. SAVE TRAINING HISTORY
# =========================================================

history_df = pd.DataFrame(history)

history_df.to_csv(
    HISTORY_PATH,
    index=False
)


# =========================================================
# 12. TRAINING SUMMARY
# =========================================================

print("\n" + "=" * 55)
print("TRAINING COMPLETED")
print("=" * 55)

print("Epochs completed:", len(history))
print("Best validation Dice:", round(best_val_dice, 4))

print("Checkpoint:", CHECKPOINT_PATH)
print("Training history:", HISTORY_PATH)

if DEVICE.type == "cuda":

    peak_memory = (
        torch.cuda.max_memory_allocated(DEVICE)
        / (1024 ** 3)
    )

    print(
        "Peak GPU memory allocated:",
        round(peak_memory, 3),
        "GB"
    )

print("\n2025 test data was NOT used during training.")
print("Proceed to validation-based threshold selection next.")
