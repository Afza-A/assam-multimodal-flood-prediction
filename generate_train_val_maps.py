
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap, BoundaryNorm
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

YEARS = [2021, 2022, 2023, 2024]

CHECKPOINT_PATH = "best_flood_model.pth"
THRESHOLD_PATH = "best_threshold.json"

OUTPUT_DIR = Path("train_val_map_results") / "annual_composite_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print("ANNUAL AGGREGATED FLOOD MAP GENERATION")
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
        f"Threshold key not found. Available keys: "
        f"{list(threshold_data.keys())}"
    )

print("Selected threshold:", THRESHOLD)


# ============================================================
# 3. LOAD TRAINING AND VALIDATION DATA ONLY
# ============================================================

print("\nLoading training dataset...")

train_dataset = AssamFloodDataset("train")

print("\nLoading validation dataset...")

val_dataset = AssamFloodDataset("val")

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("\nTraining events:", len(train_dataset))
print("Validation events:", len(val_dataset))


# ============================================================
# 4. INITIALIZE AND LOAD TRAINED MODEL
# ============================================================

sar_encoder = SAREncoder()
dem_encoder = DEMEncoder()
rainfall_encoder = RainfallEncoder()

model = MultimodalFloodModel(
    sar_encoder,
    dem_encoder,
    rainfall_encoder
).to(DEVICE)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("\nTrained model loaded successfully.")
print("Best checkpoint epoch:", checkpoint.get("epoch", "Unknown"))
print("Device:", DEVICE)


# ============================================================
# 5. INITIALIZE YEAR-WISE ACCUMULATORS
# ============================================================

annual_data = {}

for year in YEARS:

    annual_data[year] = {

        "probability_sum": np.zeros((10, 10), dtype=np.float64),

        "reference_sum": np.zeros((10, 10), dtype=np.float64),

        "event_count": 0

    }


# ============================================================
# 6. COLLECT PREDICTIONS AND REFERENCES
# ============================================================

def process_loader(loader, split_name):

    print(f"\nProcessing {split_name} data...")

    with torch.no_grad():

        for batch in loader:

            sar = batch["sar"].to(DEVICE)
            dem = batch["dem"].to(DEVICE)
            rainfall = batch["rainfall"].to(DEVICE)

            reference_masks = batch["mask"].cpu().numpy()

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

            batch_size = sar.shape[0]

            for i in range(batch_size):

                year = int(batch["year"][i].item())

                if year not in YEARS:
                    continue

                probability_map = probabilities[i, 0]

                reference_mask = (
                    reference_masks[i, 0] >= 0.5
                ).astype(np.float64)

                annual_data[year]["probability_sum"] += (
                    probability_map
                )

                annual_data[year]["reference_sum"] += (
                    reference_mask
                )

                annual_data[year]["event_count"] += 1

    print(f"{split_name} data processed successfully.")


# Training years: 2021, 2022, 2023
process_loader(
    train_loader,
    "TRAINING"
)

# Validation year: 2024
process_loader(
    val_loader,
    "VALIDATION"
)


# ============================================================
# 7. CALCULATE ANNUAL AGGREGATED GRIDS
# ============================================================

for year in YEARS:

    count = annual_data[year]["event_count"]

    if count != 60:

        raise ValueError(
            f"Expected 60 events in {year}, but found {count}."
        )

    # Average predicted probability across all events
    annual_probability = (
        annual_data[year]["probability_sum"] / count
    )

    # Average reference flood occurrence at each pixel
    annual_reference_frequency = (
        annual_data[year]["reference_sum"] / count
    )

    # Binary predicted map using validation-selected threshold
    annual_prediction = (
        annual_probability >= THRESHOLD
    ).astype(np.uint8)

    # Majority-vote independent reference mask
    annual_reference = (
        annual_reference_frequency >= 0.50
    ).astype(np.uint8)

    annual_data[year]["probability"] = annual_probability

    annual_data[year]["prediction"] = annual_prediction

    annual_data[year]["reference_frequency"] = (
        annual_reference_frequency
    )

    annual_data[year]["reference"] = annual_reference

    print(
        f"\n{year}: {count} events aggregated successfully."
    )


# ============================================================
# 8. FLOOD INTENSITY COLOUR SCALE
# ============================================================

# Light blue  = Least flooded
# Medium blue = Moderately flooded
# Dark blue   = Highly flooded

probability_cmap = ListedColormap([
    "#deebf7",
    "#6baed6",
    "#08519c"
])

probability_norm = BoundaryNorm(
    [0.0, 0.33, 0.67, 1.000001],
    probability_cmap.N
)

binary_cmap = ListedColormap([
    "white",
    "black"
])


# ============================================================
# 9. COMMON PIXEL GRID SETTINGS
# ============================================================

def configure_grid(ax):

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    ax.set_xticklabels(range(10), fontsize=11)
    ax.set_yticklabels(range(10), fontsize=11)

    ax.set_xlabel("Pixel X", fontsize=13)
    ax.set_ylabel("Pixel Y", fontsize=13)

    # Separate every pixel using grid lines
    ax.set_xticks(
        np.arange(-0.5, 10, 1),
        minor=True
    )

    ax.set_yticks(
        np.arange(-0.5, 10, 1),
        minor=True
    )

    ax.grid(
        which="minor",
        color="black",
        linewidth=0.8
    )

    ax.tick_params(
        which="minor",
        bottom=False,
        left=False
    )

    ax.set_xlim(-0.5, 9.5)
    ax.set_ylim(9.5, -0.5)


# ============================================================
# 10. GENERATE PROBABILITY MAP
# ============================================================

def generate_probability_map(year, probability):

    fig, ax = plt.subplots(figsize=(9, 8))

    image = ax.imshow(
        probability,
        cmap=probability_cmap,
        norm=probability_norm,
        interpolation="nearest"
    )

    ax.set_title(
        f"{year} | Annual Aggregated Predicted Flood Intensity",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    # Display averaged probability in every pixel
    for y in range(10):

        for x in range(10):

            value = probability[y, x]

            text_color = (
                "white" if value >= 0.50 else "black"
            )

            ax.text(
                x,
                y,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=9,
                color=text_color
            )

    configure_grid(ax)

    colorbar = fig.colorbar(
        image,
        ax=ax,
        fraction=0.046,
        pad=0.04,
        ticks=[0.165, 0.50, 0.835]
    )

    colorbar.ax.set_yticklabels([
        "Least flooded",
        "Moderately flooded",
        "Highly flooded"
    ])

    colorbar.set_label(
        "Predicted Flood Intensity",
        fontsize=11
    )

    fig.tight_layout()

    output_path = (
        OUTPUT_DIR /
        f"{year}_probability_map.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)


# ============================================================
# 11. GENERATE PREDICTED BINARY MASK
# ============================================================

def generate_binary_map(year, prediction):

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.imshow(
        prediction,
        cmap=binary_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax.set_title(
        f"{year} | Annual Aggregated Predicted Flood Mask\n"
        f"Threshold = {THRESHOLD:.2f}",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    for y in range(10):

        for x in range(10):

            value = int(prediction[y, x])

            text_color = (
                "white" if value == 1 else "black"
            )

            ax.text(
                x,
                y,
                str(value),
                ha="center",
                va="center",
                fontsize=12,
                color=text_color
            )

    configure_grid(ax)

    fig.text(
        0.5,
        0.04,
        "0 = Non-flooded     |     1 = Flooded",
        ha="center",
        fontsize=12
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    output_path = (
        OUTPUT_DIR /
        f"{year}_predicted_binary_mask.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)


# ============================================================
# 12. GENERATE INDEPENDENT REFERENCE MASK
# ============================================================

def generate_reference_map(year, reference):

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.imshow(
        reference,
        cmap=binary_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest"
    )

    ax.set_title(
        f"{year} | Annual Aggregated Reference Flood Mask\n"
        "Majority Vote Across 60 Events",
        fontsize=15,
        fontweight="bold",
        pad=15
    )

    for y in range(10):

        for x in range(10):

            value = int(reference[y, x])

            text_color = (
                "white" if value == 1 else "black"
            )

            ax.text(
                x,
                y,
                str(value),
                ha="center",
                va="center",
                fontsize=12,
                color=text_color
            )

    configure_grid(ax)

    fig.text(
        0.5,
        0.04,
        "0 = Non-flooded     |     1 = Flooded",
        ha="center",
        fontsize=12
    )

    fig.tight_layout(rect=[0, 0.08, 1, 1])

    output_path = (
        OUTPUT_DIR /
        f"{year}_reference_mask.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)

    print("Saved:", output_path)


# ============================================================
# 13. GENERATE THREE MAPS FOR EACH YEAR
# ============================================================

print("\n" + "=" * 65)
print("GENERATING ANNUAL AGGREGATED MAPS")
print("=" * 65)

for year in YEARS:

    print(f"\nGenerating maps for {year}...")

    probability = annual_data[year]["probability"]

    prediction = annual_data[year]["prediction"]

    reference = annual_data[year]["reference"]

    generate_probability_map(
        year,
        probability
    )

    generate_binary_map(
        year,
        prediction
    )

    generate_reference_map(
        year,
        reference
    )


# ============================================================
# 14. COMPLETION
# ============================================================

print("\n" + "=" * 65)
print("ANNUAL MAP GENERATION COMPLETED")
print("=" * 65)

print("Training years: 2021, 2022, 2023")
print("Validation year: 2024")
print("Events aggregated per year: 60")
print("Maps per year: 3")
print("Total images generated: 12")

print("\n2025 TEST DATA WAS NOT USED.")

print(
    "\nOutput folder:",
    OUTPUT_DIR.resolve()
)
