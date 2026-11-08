
import json
import numpy as np
import torch

from sar_encoder import SAREncoder
from dem_encoder import DEMEncoder
from rainfall_encoder import RainfallEncoder

from fusion_decoder import MultimodalFloodModel
from pre_flood_dataset import get_dataloaders


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_PATH = "best_flood_model.pth"

# Candidate thresholds.
THRESHOLDS = np.arange(0.10, 0.91, 0.05)


# ---------------------------------------------------------
# 2. LOAD VALIDATION DATA ONLY
# ---------------------------------------------------------

_, val_loader, _ = get_dataloaders(batch_size=4)


# ---------------------------------------------------------
# 3. LOAD BEST MODEL CHECKPOINT
# ---------------------------------------------------------

model = MultimodalFloodModel(
    SAREncoder(),
    DEMEncoder(),
    RainfallEncoder()
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nLoaded checkpoint from epoch:", checkpoint["epoch"])
print("Best validation Dice during training:",
      checkpoint["best_val_dice"])


# ---------------------------------------------------------
# 4. COLLECT VALIDATION PROBABILITIES
# ---------------------------------------------------------

all_probabilities = []
all_targets = []

with torch.no_grad():

    for batch in val_loader:

        sar = batch["sar"].to(DEVICE)
        dem = batch["dem"].to(DEVICE)
        rainfall = batch["rainfall"].to(DEVICE)
        mask = batch["mask"].to(DEVICE)

        outputs = model(sar, dem, rainfall)

        all_probabilities.append(
            outputs["probabilities"].cpu().numpy()
        )

        all_targets.append(
            mask.cpu().numpy()
        )

probabilities = np.concatenate(
    all_probabilities, axis=0
).reshape(-1)

targets = np.concatenate(
    all_targets, axis=0
).reshape(-1)

print("\nValidation pixels:", len(targets))
print("Actual flooded percentage:",
      round(targets.mean() * 100, 2))


# ---------------------------------------------------------
# 5. EVALUATE CANDIDATE THRESHOLDS
# ---------------------------------------------------------

results = []

for threshold in THRESHOLDS:

    predictions = (
        probabilities >= threshold
    ).astype(np.uint8)

    tp = np.sum((predictions == 1) & (targets == 1))
    fp = np.sum((predictions == 1) & (targets == 0))
    fn = np.sum((predictions == 0) & (targets == 1))
    tn = np.sum((predictions == 0) & (targets == 0))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    dice = (
        2 * tp
    ) / (
        2 * tp + fp + fn + 1e-8
    )

    iou = tp / (
        tp + fp + fn + 1e-8
    )

    results.append({
        "threshold": round(float(threshold), 2),
        "precision": precision,
        "recall": recall,
        "dice": dice,
        "iou": iou,
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn)
    })


# ---------------------------------------------------------
# 6. SELECT BEST VALIDATION THRESHOLD
# ---------------------------------------------------------

best_result = max(
    results,
    key=lambda x: x["dice"]
)

print("\nTHRESHOLD COMPARISON")
print("-" * 75)

print(
    f"{'Threshold':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'Dice':<12}"
    f"{'IoU':<12}"
)

for result in results:

    print(
        f"{result['threshold']:<12.2f}"
        f"{result['precision']:<12.4f}"
        f"{result['recall']:<12.4f}"
        f"{result['dice']:<12.4f}"
        f"{result['iou']:<12.4f}"
    )


# ---------------------------------------------------------
# 7. SAVE SELECTED THRESHOLD
# ---------------------------------------------------------

selected_threshold = best_result["threshold"]

with open("best_threshold.json", "w") as f:

    json.dump({
        "threshold": selected_threshold,
        "selection_metric": "validation_dice",
        "validation_dice": float(best_result["dice"]),
        "validation_iou": float(best_result["iou"]),
        "validation_precision": float(best_result["precision"]),
        "validation_recall": float(best_result["recall"]),
        "checkpoint": CHECKPOINT_PATH,
        "validation_year": 2024
    }, f, indent=4)


print("\nBEST VALIDATION THRESHOLD")
print("Threshold:", selected_threshold)
print("Dice:", round(best_result["dice"], 4))
print("IoU:", round(best_result["iou"], 4))
print("Precision:", round(best_result["precision"], 4))
print("Recall:", round(best_result["recall"], 4))

print("\nSaved: best_threshold.json")
print("\n2025 TEST DATA WAS NOT USED.")
