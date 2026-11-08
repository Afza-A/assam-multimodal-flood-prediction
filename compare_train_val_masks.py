
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from pre_flood_dataset import AssamFloodDataset
from sar_encoder import SAREncoder
from dem_encoder import DEMEncoder
from rainfall_encoder import RainfallEncoder
from fusion_decoder import MultimodalFloodModel


# ============================================================
# 1. CONFIGURATION
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

BATCH_SIZE = 4

OUTPUT_DIR = Path("train_val_mask_comparison")
OUTPUT_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = "best_flood_model.pth"
THRESHOLD_PATH = "best_threshold.json"

print("=" * 65)
print("TRAINING AND VALIDATION FLOOD MASK COMPARISON")
print("=" * 65)


# ============================================================
# 2. LOAD SELECTED THRESHOLD
# ============================================================

with open(THRESHOLD_PATH, "r") as file:
    threshold_data = json.load(file)

if "best_threshold" in threshold_data:
    THRESHOLD = float(threshold_data["best_threshold"])
elif "threshold" in threshold_data:
    THRESHOLD = float(threshold_data["threshold"])
else:
    raise KeyError(
        f"Threshold key not found: {list(threshold_data.keys())}"
    )

print("Selected threshold:", THRESHOLD)


# ============================================================
# 3. INITIALIZE MODEL
# ============================================================

sar_encoder = SAREncoder()
dem_encoder = DEMEncoder()
rainfall_encoder = RainfallEncoder()

model = MultimodalFloodModel(
    sar_encoder,
    dem_encoder,
    rainfall_encoder
).to(DEVICE)


# ============================================================
# 4. LOAD TRAINED CHECKPOINT
# ============================================================

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Model loaded successfully.")
print("Checkpoint epoch:", checkpoint.get("epoch", "Unknown"))
print("Device:", DEVICE)


# ============================================================
# 5. CALCULATE EVENT-LEVEL METRICS
# ============================================================

def calculate_metrics(prediction, reference):

    prediction = prediction.astype(bool)
    reference = reference.astype(bool)

    TP = int(np.logical_and(prediction, reference).sum())
    FP = int(np.logical_and(prediction, ~reference).sum())
    TN = int(np.logical_and(~prediction, ~reference).sum())
    FN = int(np.logical_and(~prediction, reference).sum())

    epsilon = 1e-8

    precision = TP / (TP + FP + epsilon)

    recall = TP / (TP + FN + epsilon)

    dice = (
        2 * TP
    ) / (
        2 * TP + FP + FN + epsilon
    )

    iou = TP / (
        TP + FP + FN + epsilon
    )

    accuracy = (
        TP + TN
    ) / (
        TP + TN + FP + FN + epsilon
    )

    return {
        "TP": TP,
        "FP": FP,
        "TN": TN,
        "FN": FN,
        "Precision": precision,
        "Recall": recall,
        "Dice": dice,
        "IoU": iou,
        "Accuracy": accuracy
    }


# ============================================================
# 6. EVALUATE ALL EVENTS IN A DATASET SPLIT
# ============================================================

def evaluate_split(split_name, dataset_split):

    print(f"\nEvaluating {split_name}...")

    dataset = AssamFloodDataset(dataset_split)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    event_results = []

    total_TP = 0
    total_FP = 0
    total_TN = 0
    total_FN = 0

    with torch.no_grad():

        for batch in loader:

            sar = batch["sar"].to(DEVICE)
            dem = batch["dem"].to(DEVICE)
            rainfall = batch["rainfall"].to(DEVICE)

            reference_masks = (
                batch["mask"].cpu().numpy()
            )

            outputs = model(
                sar,
                dem,
                rainfall
            )

            probabilities = (
                outputs["probabilities"]
                .detach()
                .cpu()
                .numpy()
            )

            predictions = (
                probabilities >= THRESHOLD
            ).astype(np.uint8)

            current_batch_size = sar.shape[0]

            for i in range(current_batch_size):

                year = int(batch["year"][i].item())

                district = batch["district"][i]

                event_id = batch["event_id"][i]

                prediction = predictions[i, 0]

                reference = (
                    reference_masks[i, 0] >= 0.5
                ).astype(np.uint8)

                metrics = calculate_metrics(
                    prediction,
                    reference
                )

                result = {
                    "Split": split_name,
                    "Year": year,
                    "District": district,
                    "Event_ID": event_id,
                    **metrics
                }

                event_results.append(result)

                total_TP += metrics["TP"]
                total_FP += metrics["FP"]
                total_TN += metrics["TN"]
                total_FN += metrics["FN"]

    # Overall pixel-level metrics across all events
    epsilon = 1e-8

    micro_metrics = {

        "Split": split_name,

        "Events": len(event_results),

        "TP": total_TP,
        "FP": total_FP,
        "TN": total_TN,
        "FN": total_FN,

        "Precision": total_TP / (
            total_TP + total_FP + epsilon
        ),

        "Recall": total_TP / (
            total_TP + total_FN + epsilon
        ),

        "Dice": (2 * total_TP) / (
            2 * total_TP + total_FP + total_FN + epsilon
        ),

        "IoU": total_TP / (
            total_TP + total_FP + total_FN + epsilon
        ),

        "Accuracy": (total_TP + total_TN) / (
            total_TP + total_TN + total_FP + total_FN + epsilon
        )
    }

    return event_results, micro_metrics


# ============================================================
# 7. EVALUATE TRAINING AND VALIDATION
# ============================================================

train_results, train_micro = evaluate_split(
    "Training",
    "train"
)

val_results, val_micro = evaluate_split(
    "Validation",
    "val"
)

all_results = train_results + val_results

event_df = pd.DataFrame(all_results)


# ============================================================
# 8. SAVE EVENT-LEVEL METRICS
# ============================================================

event_path = OUTPUT_DIR / "event_level_metrics.csv"

event_df.to_csv(
    event_path,
    index=False
)

print("\nSaved event-level metrics:", event_path)


# ============================================================
# 9. CALCULATE YEAR-WISE AVERAGE METRICS
# ============================================================

metric_columns = [
    "Precision",
    "Recall",
    "Dice",
    "IoU",
    "Accuracy"
]

year_summary = (
    event_df
    .groupby(["Split", "Year"])[metric_columns]
    .mean()
    .reset_index()
)

year_summary["Events"] = (
    event_df
    .groupby(["Split", "Year"])
    .size()
    .values
)

year_path = OUTPUT_DIR / "year_wise_metrics.csv"

year_summary.to_csv(
    year_path,
    index=False
)


# ============================================================
# 10. SAVE OVERALL PIXEL-LEVEL METRICS
# ============================================================

micro_df = pd.DataFrame([
    train_micro,
    val_micro
])

micro_path = OUTPUT_DIR / "overall_pixel_metrics.csv"

micro_df.to_csv(
    micro_path,
    index=False
)


# ============================================================
# 11. DISPLAY YEAR-WISE RESULTS
# ============================================================

print("\n" + "=" * 65)
print("YEAR-WISE EVENT-AVERAGED RESULTS")
print("=" * 65)

print(
    year_summary.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 12. DISPLAY OVERALL RESULTS
# ============================================================

print("\n" + "=" * 65)
print("OVERALL PIXEL-LEVEL RESULTS")
print("=" * 65)

print(
    micro_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# ============================================================
# 13. VALIDATION SIMILARITY INTERPRETATION
# ============================================================

validation_dice = val_micro["Dice"]
validation_iou = val_micro["IoU"]

print("\n" + "=" * 65)
print("VALIDATION MASK SIMILARITY")
print("=" * 65)

print(f"Validation Dice: {validation_dice:.4f}")
print(f"Validation IoU:  {validation_iou:.4f}")

print(
    "\nDice and IoU measure the overlap between "
    "predicted and reference flood masks."
)

print(
    "Use the validation results and year-wise results "
    "to decide whether the model is ready to freeze."
)


# ============================================================
# 14. COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("MASK COMPARISON COMPLETED")
print("=" * 65)

print("Training events:", len(train_results))
print("Validation events:", len(val_results))

print("\nSaved files:")
print(event_path)
print(year_path)
print(micro_path)

print("\n2025 TEST DATA WAS NOT USED.")
