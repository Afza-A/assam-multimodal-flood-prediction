
import json
import shutil
import hashlib
from pathlib import Path
from datetime import datetime

# Files that define the selected model and its preprocessing
FILES_TO_FREEZE = [
    "best_flood_model.pth",
    "best_threshold.json",
    "normalization_stats.json",
    "fusion_decoder.py",
    "sar_encoder.py",
    "dem_encoder.py",
    "rainfall_encoder.py",
    "pre_flood_dataset.py"
]

SOURCE_DIR = Path(".")
FROZEN_DIR = Path("frozen_model")

FROZEN_DIR.mkdir(exist_ok=True)

# Verify all required files exist before copying anything
missing = [
    filename
    for filename in FILES_TO_FREEZE
    if not (SOURCE_DIR / filename).is_file()
]

if missing:
    raise FileNotFoundError(
        f"Cannot freeze model. Missing files: {missing}"
    )

# Copy selected files
for filename in FILES_TO_FREEZE:

    source = SOURCE_DIR / filename
    destination = FROZEN_DIR / filename

    shutil.copy2(source, destination)

    print("Frozen:", filename)


# Read selected threshold
with open(FROZEN_DIR / "best_threshold.json", "r") as file:
    threshold_data = json.load(file)

if "best_threshold" in threshold_data:
    threshold = float(threshold_data["best_threshold"])
else:
    threshold = float(threshold_data["threshold"])


# Read checkpoint information
import torch

checkpoint = torch.load(
    FROZEN_DIR / "best_flood_model.pth",
    map_location="cpu",
    weights_only=False
)

# Calculate SHA256 checksums for integrity verification
checksums = {}

for filename in FILES_TO_FREEZE:

    file_path = FROZEN_DIR / filename

    sha256 = hashlib.sha256(
        file_path.read_bytes()
    ).hexdigest()

    checksums[filename] = sha256


# Save freeze manifest
manifest = {

    "freeze_date": datetime.now().isoformat(),

    "checkpoint_epoch": checkpoint.get("epoch"),

    "validation_best_dice": checkpoint.get("best_val_dice"),

    "selected_threshold": threshold,

    "training_years": [2021, 2022, 2023],

    "validation_year": 2024,

    "reserved_test_year": 2025,

    "normalization_file": "normalization_stats.json",

    "model_frozen": True,

    "checksums_sha256": checksums

}

with open(FROZEN_DIR / "freeze_manifest.json", "w") as file:

    json.dump(
        manifest,
        file,
        indent=4
    )

print("\n" + "=" * 55)
print("MODEL FREEZING COMPLETED")
print("=" * 55)

print("Checkpoint epoch:", checkpoint.get("epoch"))
print("Selected threshold:", threshold)
print("Validation best Dice:", checkpoint.get("best_val_dice"))
print("Frozen files:", len(FILES_TO_FREEZE))
print("Location:", FROZEN_DIR.resolve())

print("\n2025 TEST DATA WAS NOT USED.")
