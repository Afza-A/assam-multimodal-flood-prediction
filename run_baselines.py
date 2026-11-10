"""Train and compare reduced-modality CNN/GRU models and a Random Forest.

Training: 2021-2023. Hyperparameter selection: 2024 only. Final test: 2025,
after all configurations have been selected and saved. The frozen full model
and its threshold are loaded read-only and never modified by this script.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, Dataset


PROJECT_DIR = Path(__file__).resolve().parent
FROZEN_DIR = PROJECT_DIR / "frozen_model"
OUTPUT_DIR = PROJECT_DIR / "baseline_experiments"
SEED = 42
BATCH_SIZE = 4
MAX_EPOCHS = 50
PATIENCE = 8
LEARNING_RATES = [0.001, 0.0003]
THRESHOLD = 0.50
EPSILON = 1e-8
METRIC_COLUMNS = ["Precision", "Recall", "Dice", "IoU", "Accuracy"]

CONFIGS = {
    "SAR-only": ("sar",),
    "DEM-only": ("dem",),
    "Rainfall-only": ("rainfall",),
    "SAR+DEM": ("sar", "dem"),
    "SAR+Rainfall": ("sar", "rainfall"),
    "DEM+Rainfall": ("dem", "rainfall"),
}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen() -> tuple[dict, float]:
    manifest_path = FROZEN_DIR / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    errors = []
    for filename, expected in manifest["checksums_sha256"].items():
        path = FROZEN_DIR / filename
        if not path.is_file():
            errors.append(f"missing {filename}")
        elif sha256(path).lower() != expected.lower():
            errors.append(f"hash mismatch {filename}")
    if errors:
        raise RuntimeError("Frozen package integrity check failed: " + "; ".join(errors))

    threshold_data = json.loads(
        (FROZEN_DIR / "best_threshold.json").read_text(encoding="utf-8-sig")
    )
    threshold = float(threshold_data.get("best_threshold", threshold_data.get("threshold")))
    if threshold != 0.50 or float(manifest["selected_threshold"]) != 0.50:
        raise RuntimeError("Full-model frozen threshold must remain 0.50.")
    root_norm = PROJECT_DIR / "normalization_stats.json"
    if not root_norm.is_file() or sha256(root_norm) != sha256(FROZEN_DIR / "normalization_stats.json"):
        raise RuntimeError("Project and frozen normalization files differ.")
    return manifest, threshold


def metrics_from_counts(tp: int, fp: int, tn: int, fn: int) -> dict:
    return {
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "Precision": tp / (tp + fp + EPSILON),
        "Recall": tp / (tp + fn + EPSILON),
        "Dice": (2 * tp) / (2 * tp + fp + fn + EPSILON),
        "IoU": tp / (tp + fp + fn + EPSILON),
        "Accuracy": (tp + tn) / (tp + tn + fp + fn + EPSILON),
    }


def count_pixels(prediction: np.ndarray, reference: np.ndarray) -> tuple[int, int, int, int]:
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    return (
        int(np.logical_and(pred, ref).sum()),
        int(np.logical_and(pred, ~ref).sum()),
        int(np.logical_and(~pred, ~ref).sum()),
        int(np.logical_and(~pred, ref).sum()),
    )


class CachedSplit(Dataset):
    """Cache the project's aligned, normalized event tensors once per split."""

    def __init__(self, dataset_class, split: str):
        self.source = dataset_class(split)
        self.samples = [self.source[index] for index in range(len(self.source))]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        return self.samples[index]


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, value):
        return self.block(value)


class ReducedModalityModel(nn.Module):
    """Subset encoders plus the same one-level U-Net-style decoder pattern."""

    def __init__(self, modalities, encoder_classes):
        super().__init__()
        self.modalities = tuple(modalities)
        for modality in self.modalities:
            setattr(self, f"{modality}_encoder", encoder_classes[modality]())
        self.encoder_block = ConvBlock(32 * len(self.modalities), 64)
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = ConvBlock(128, 128)
        self.upsample = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder_block = ConvBlock(128, 32)
        self.output_layer = nn.Conv2d(32, 1, 1)

    def forward(self, batch):
        features = [
            getattr(self, f"{name}_encoder")(batch[name])
            for name in self.modalities
        ]
        fused = torch.cat(features, dim=1)
        skip = self.encoder_block(fused)
        value = self.bottleneck(self.downsample(skip))
        value = self.upsample(value)
        if value.shape[-2:] != skip.shape[-2:]:
            value = F.interpolate(value, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        value = torch.cat([value, skip], dim=1)
        return self.output_layer(self.decoder_block(value))


def combined_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    probability = torch.sigmoid(logits).reshape(-1)
    labels = target.reshape(-1)
    dice_loss = 1 - (2 * (probability * labels).sum() + 1) / (
        probability.sum() + labels.sum() + 1
    )
    return 0.5 * bce + 0.5 * dice_loss


def evaluate_neural(model, loader, device, threshold=THRESHOLD, keep_events=False):
    model.eval()
    totals = [0, 0, 0, 0]
    event_rows = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {name: batch[name].to(device) for name in model.modalities}
            probabilities = torch.sigmoid(model(inputs)).cpu().numpy()
            references = batch["mask"].cpu().numpy()
            for index in range(len(probabilities)):
                pred = (probabilities[index, 0] >= threshold).astype(np.uint8)
                ref = (references[index, 0] >= 0.5).astype(np.uint8)
                counts = count_pixels(pred, ref)
                totals = [left + right for left, right in zip(totals, counts)]
                if keep_events:
                    event = {
                        "Year": int(batch["year"][index].item()),
                        "District": str(batch["district"][index]),
                        "Event_ID": str(batch["event_id"][index]),
                        **metrics_from_counts(*counts),
                    }
                    event_rows.append(event)
    return metrics_from_counts(*totals), event_rows


def train_candidate(model, train_loader, val_loader, device, learning_rate):
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_state = None
    best_score = -1.0
    best_epoch = 0
    stale_epochs = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for batch in train_loader:
            inputs = {name: batch[name].to(device) for name in model.modalities}
            target = batch["mask"].to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = combined_loss(model(inputs), target)
            loss.backward()
            optimizer.step()

        validation_metrics, _ = evaluate_neural(model, val_loader, device)
        score = validation_metrics["Dice"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= PATIENCE:
                break

    return best_state, best_score, best_epoch


def pixel_feature_matrix(samples, modalities):
    matrices = []
    targets = []
    for sample in samples:
        parts = []
        if "sar" in modalities:
            parts.append(sample["sar"].numpy().reshape(100, 1))
        if "dem" in modalities:
            parts.append(sample["dem"].numpy().reshape(100, 1))
        if "rainfall" in modalities:
            parts.append(sample["rainfall"].numpy().reshape(100, 10))
        matrices.append(np.concatenate(parts, axis=1).astype(np.float32))
        targets.append(sample["mask"].numpy().reshape(100).astype(np.uint8))
    return np.concatenate(matrices), np.concatenate(targets)


def rf_validation_metrics(model, x, y):
    probability = model.predict_proba(x)[:, 1]
    prediction = probability >= THRESHOLD
    return metrics_from_counts(*count_pixels(prediction, y))


def evaluate_rf_events(model, samples, modalities, threshold=THRESHOLD):
    rows = []
    total = [0, 0, 0, 0]
    for sample in samples:
        x, y = pixel_feature_matrix([sample], modalities)
        probability = model.predict_proba(x)[:, 1].reshape(10, 10)
        pred = (probability >= threshold).astype(np.uint8)
        ref = y.reshape(10, 10)
        counts = count_pixels(pred, ref)
        total = [a + b for a, b in zip(total, counts)]
        rows.append({
            "Year": int(sample["year"]),
            "District": str(sample["district"]),
            "Event_ID": str(sample["event_id"]),
            **metrics_from_counts(*counts),
        })
    return metrics_from_counts(*total), rows


def main():
    seed_everything()
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"{OUTPUT_DIR} already has results. Rename it before starting a fresh run."
        )
    frozen_manifest, full_threshold = verify_frozen()

    sys.path.insert(0, str(FROZEN_DIR))
    os.chdir(PROJECT_DIR)
    from dem_encoder import DEMEncoder
    from fusion_decoder import MultimodalFloodModel
    from pre_flood_dataset import AssamFloodDataset
    from rainfall_encoder import RainfallEncoder
    from sar_encoder import SAREncoder

    encoder_classes = {"sar": SAREncoder, "dem": DEMEncoder, "rainfall": RainfallEncoder}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Cache only train and validation data for candidate selection.
    train_data = CachedSplit(AssamFloodDataset, "train")
    val_data = CachedSplit(AssamFloodDataset, "val")
    if len(train_data) != 180 or len(val_data) != 60:
        raise RuntimeError("Expected 180 training events and 60 validation events.")
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_dir = OUTPUT_DIR / "selected_models"
    model_dir.mkdir()

    selected = {}
    validation_rows = []
    for label, modalities in CONFIGS.items():
        best_for_config = None
        for learning_rate in LEARNING_RATES:
            seed_everything()
            model = ReducedModalityModel(modalities, encoder_classes).to(device)
            state, score, epoch = train_candidate(
                model, train_loader, val_loader, device, learning_rate
            )
            validation_rows.append({
                "Model": label, "Modalities": "+".join(modalities),
                "Learning_Rate": learning_rate, "Best_Epoch": epoch,
                "Validation_Dice_at_0.50": score,
            })
            if best_for_config is None or score > best_for_config["validation_dice"]:
                best_for_config = {
                    "model": copy.deepcopy(model).cpu(),
                    "state": state,
                    "learning_rate": learning_rate,
                    "best_epoch": epoch,
                    "validation_dice": score,
                    "modalities": modalities,
                }
        label_slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
        checkpoint_path = model_dir / f"{label_slug}.pth"
        torch.save({
            "model_state_dict": best_for_config["state"],
            "modalities": list(best_for_config["modalities"]),
            "learning_rate": best_for_config["learning_rate"],
            "best_epoch": best_for_config["best_epoch"],
            "validation_dice": best_for_config["validation_dice"],
            "threshold": THRESHOLD,
            "seed": SEED,
        }, checkpoint_path)
        selected[label] = {
            "modalities": list(modalities),
            "learning_rate": best_for_config["learning_rate"],
            "best_epoch": best_for_config["best_epoch"],
            "validation_dice": best_for_config["validation_dice"],
            "threshold": THRESHOLD,
            "checkpoint": str(checkpoint_path.relative_to(OUTPUT_DIR)),
        }
        print(f"Selected {label}: lr={best_for_config['learning_rate']}, val Dice={best_for_config['validation_dice']:.4f}")

    pd.DataFrame(validation_rows).to_csv(OUTPUT_DIR / "validation_hyperparameter_grid.csv", index=False)

    # Random Forest candidates: features are the normalized per-pixel SAR/DEM
    # values and the ten chronological normalized rainfall values.
    x_train, y_train = pixel_feature_matrix(train_data.samples, ("sar", "dem", "rainfall"))
    x_val, y_val = pixel_feature_matrix(val_data.samples, ("sar", "dem", "rainfall"))
    rf_candidates = [
        {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 2},
        {"n_estimators": 200, "max_depth": 16, "min_samples_leaf": 1},
    ]
    rf_rows = []
    rf_best = None
    for config in rf_candidates:
        rf = RandomForestClassifier(
            **config, random_state=SEED, n_jobs=-1, class_weight=None
        )
        rf.fit(x_train, y_train)
        score = rf_validation_metrics(rf, x_val, y_val)["Dice"]
        rf_rows.append({**config, "Validation_Dice_at_0.50": score})
        if rf_best is None or score > rf_best["validation_dice"]:
            rf_best = {"model": rf, "config": config, "validation_dice": score}
    import joblib

    joblib.dump(rf_best["model"], model_dir / "random_forest.joblib")
    selected["Random Forest"] = {
        **rf_best["config"],
        "feature_modalities": ["SAR VV", "DEM", "Rainfall Days 1-10"],
        "feature_count": int(x_train.shape[1]),
        "validation_dice": rf_best["validation_dice"],
        "threshold": THRESHOLD,
        "checkpoint": "selected_models/random_forest.joblib",
    }
    pd.DataFrame(rf_rows).to_csv(OUTPUT_DIR / "random_forest_validation_grid.csv", index=False)

    # Save every validation-selected configuration before opening test data.
    config_record = {
        "seed": SEED,
        "train_years": [2021, 2022, 2023],
        "validation_year": 2024,
        "test_year": 2025,
        "fixed_threshold_for_all_models": THRESHOLD,
        "candidate_learning_rates": LEARNING_RATES,
        "max_epochs": MAX_EPOCHS,
        "early_stopping_patience": PATIENCE,
        "selected_models": selected,
        "full_model_checkpoint_sha256": sha256(FROZEN_DIR / "best_flood_model.pth"),
        "freeze_manifest_sha256": sha256(FROZEN_DIR / "freeze_manifest.json"),
        "saved_before_2025_test": True,
    }
    (OUTPUT_DIR / "selected_configurations.json").write_text(
        json.dumps(config_record, indent=2), encoding="utf-8"
    )

    # Only now load the held-out test split. No tuning occurs below this point.
    test_data = CachedSplit(AssamFloodDataset, "test")
    test_years = {int(sample["year"]) for sample in test_data.samples}
    if len(test_data) != 60 or test_years != {2025}:
        raise RuntimeError("Expected exactly 60 held-out test events from 2025.")
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    test_overall_rows = []
    test_event_rows = []
    for label, modalities in CONFIGS.items():
        model = ReducedModalityModel(modalities, encoder_classes).to(device)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", label).strip("_").lower()
        saved = torch.load(model_dir / f"{slug}.pth", map_location="cpu", weights_only=False)
        model.load_state_dict(saved["model_state_dict"], strict=True)
        model.to(device)
        overall, events = evaluate_neural(model, test_loader, device, THRESHOLD, keep_events=True)
        test_overall_rows.append({"Model": label, "Threshold": THRESHOLD, "Events": 60, **overall})
        test_event_rows.extend({"Model": label, **row} for row in events)

    rf_test_overall, rf_test_events = evaluate_rf_events(
        rf_best["model"], test_data.samples, ("sar", "dem", "rainfall"), THRESHOLD
    )
    test_overall_rows.append({"Model": "Random Forest", "Threshold": THRESHOLD, "Events": 60, **rf_test_overall})
    test_event_rows.extend({"Model": "Random Forest", **row} for row in rf_test_events)

    overall_df = pd.DataFrame(test_overall_rows)
    event_df = pd.DataFrame(test_event_rows)
    overall_df.to_csv(OUTPUT_DIR / "baseline_test_overall_metrics.csv", index=False)
    event_df.to_csv(OUTPUT_DIR / "baseline_test_event_metrics.csv", index=False)

    proposed_path = PROJECT_DIR / "test_results_2025" / "overall_metrics_2025.csv"
    if not proposed_path.is_file():
        raise FileNotFoundError("Frozen full-model 2025 results are missing; run evaluate_test.py first.")
    proposed = pd.read_csv(proposed_path).iloc[0]
    proposed_row = {
        "Model": "Proposed full three-modality model",
        "Threshold": float(proposed["threshold"]),
        "Events": int(proposed["number_of_events"]),
        "TP": int(proposed["TP"]), "FP": int(proposed["FP"]),
        "TN": int(proposed["TN"]), "FN": int(proposed["FN"]),
        "Precision": float(proposed["precision"]),
        "Recall": float(proposed["recall"]), "Dice": float(proposed["dice"]),
        "IoU": float(proposed["iou"]), "Accuracy": float(proposed["accuracy"]),
    }
    comparison = pd.concat([pd.DataFrame([proposed_row]), overall_df], ignore_index=True)
    comparison.to_csv(OUTPUT_DIR / "baseline_vs_proposed_2025.csv", index=False)

    ax = comparison.set_index("Model")[METRIC_COLUMNS].plot(
        kind="bar", figsize=(13, 6), ylim=(0, 1), width=0.82
    )
    ax.set_ylabel("Pooled pixel score")
    ax.set_xlabel("")
    ax.set_title("Synthetic 2025 test comparison at threshold 0.50")
    ax.legend(loc="lower right")
    ax.tick_params(axis="x", labelrotation=25)
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "baseline_vs_proposed_2025.png", dpi=180)
    plt.close()

    manifest = {
        "experiment": "Reduced-modality baselines and Random Forest",
        "created_at": datetime.now().astimezone().isoformat(),
        "data_status": "Fully synthetic; simulation-based results only",
        "training_years": [2021, 2022, 2023],
        "validation_year": 2024,
        "test_year": 2025,
        "test_events": 60,
        "threshold": THRESHOLD,
        "full_model_was_not_modified": True,
        "full_model_threshold_was_not_modified": True,
        "baseline_hyperparameters_selected_on_validation_only": True,
        "full_model_frozen_manifest_sha256": sha256(FROZEN_DIR / "freeze_manifest.json"),
        "outputs": sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file()),
    }
    (OUTPUT_DIR / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("Baseline experiment complete. Test results were not used for selection.")
    print(comparison[["Model", *METRIC_COLUMNS]].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
