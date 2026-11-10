"""Evaluate the frozen flood model once on the reserved synthetic 2025 split.

Run this file from the project root. It never trains, tunes a threshold, or
updates normalization. Results are written to a new test_results_2025 folder.
"""

from __future__ import annotations

import hashlib
import json
import os
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
from torch.utils.data import DataLoader


PROJECT_DIR = Path(__file__).resolve().parent
FROZEN_DIR = PROJECT_DIR / "frozen_model"
OUTPUT_DIR = PROJECT_DIR / "test_results_2025"
BATCH_SIZE = 4
EPSILON = 1e-8
METRIC_COLUMNS = ["Precision", "Recall", "Dice", "IoU", "Accuracy"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen_package() -> tuple[dict, float]:
    manifest_path = FROZEN_DIR / "freeze_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Frozen manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    mismatches = []
    for filename, expected in manifest["checksums_sha256"].items():
        path = FROZEN_DIR / filename
        if not path.is_file():
            mismatches.append(f"{filename}: missing")
            continue
        actual = sha256(path)
        if actual.lower() != expected.lower():
            mismatches.append(
                f"{filename}: expected {expected}, got {actual}"
            )
    if mismatches:
        raise RuntimeError("Frozen package hash check failed:\n" + "\n".join(mismatches))

    threshold_data = json.loads(
        (FROZEN_DIR / "best_threshold.json").read_text(encoding="utf-8-sig")
    )
    threshold = float(
        threshold_data.get("best_threshold", threshold_data.get("threshold"))
    )
    if not np.isclose(threshold, float(manifest["selected_threshold"])):
        raise RuntimeError("Threshold file does not match freeze manifest.")
    if int(manifest["checkpoint_epoch"]) != 4:
        raise RuntimeError("Manifest does not identify Epoch 4 checkpoint.")
    if not np.isclose(threshold, 0.50):
        raise RuntimeError(f"Expected frozen threshold 0.50, found {threshold}.")

    # The dataset loader reads this file from the project root. Require an exact
    # byte-for-byte match with the frozen training-derived statistics.
    root_norm = PROJECT_DIR / "normalization_stats.json"
    frozen_norm = FROZEN_DIR / "normalization_stats.json"
    if not root_norm.is_file() or sha256(root_norm) != sha256(frozen_norm):
        raise RuntimeError(
            "Project-root normalization_stats.json differs from the frozen copy."
        )
    return manifest, threshold


def pixel_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict:
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    tp = int(np.logical_and(pred, ref).sum())
    fp = int(np.logical_and(pred, ~ref).sum())
    tn = int(np.logical_and(~pred, ~ref).sum())
    fn = int(np.logical_and(~pred, ref).sum())
    return {
        "TP": tp,
        "FP": fp,
        "TN": tn,
        "FN": fn,
        "Precision": tp / (tp + fp + EPSILON),
        "Recall": tp / (tp + fn + EPSILON),
        "Dice": (2 * tp) / (2 * tp + fp + fn + EPSILON),
        "IoU": tp / (tp + fp + fn + EPSILON),
        "Accuracy": (tp + tn) / (tp + tn + fp + fn + EPSILON),
    }


def plot_event_maps(
    probability: np.ndarray,
    prediction: np.ndarray,
    reference: np.ndarray,
    title: str,
    output_path: Path,
    threshold: float,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), constrained_layout=True)
    im = axes[0].imshow(probability, cmap="Blues", vmin=0, vmax=1)
    axes[0].set_title("Predicted probability")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    axes[1].imshow(prediction, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Predicted mask (threshold {threshold:.2f})")
    axes[2].imshow(reference, cmap="Blues", vmin=0, vmax=1, interpolation="nearest")
    axes[2].set_title("Independent reference mask")
    for axis in axes:
        axis.set_xlabel("Pixel X")
        axis.set_ylabel("Pixel Y")
        axis.set_xticks(range(10))
        axis.set_yticks(range(10))
    fig.suptitle(title)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main() -> None:
    if not FROZEN_DIR.is_dir():
        raise FileNotFoundError(f"Frozen model folder not found: {FROZEN_DIR}")
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(
            f"Output folder already contains files: {OUTPUT_DIR}. "
            "Move or rename it before starting another test run."
        )

    manifest, threshold = verify_frozen_package()

    # Make Python import only the frozen model/data-loader definitions.
    sys.path.insert(0, str(FROZEN_DIR))
    os.chdir(PROJECT_DIR)
    from dem_encoder import DEMEncoder
    from fusion_decoder import MultimodalFloodModel
    from pre_flood_dataset import AssamFloodDataset
    from rainfall_encoder import RainfallEncoder
    from sar_encoder import SAREncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultimodalFloodModel(SAREncoder(), DEMEncoder(), RainfallEncoder()).to(device)
    checkpoint_path = FROZEN_DIR / "best_flood_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    if int(checkpoint.get("epoch", -1)) != int(manifest["checkpoint_epoch"]):
        raise RuntimeError("Checkpoint epoch does not match freeze manifest.")
    model.eval()

    dataset = AssamFloodDataset("test")
    event_keys = getattr(dataset, "events", [])
    if len(dataset) != 60 or {int(key[1]) for key in event_keys} != {2025}:
        raise RuntimeError(
            f"Expected exactly 60 events from 2025; got {len(dataset)} events "
            f"with years {sorted({int(key[1]) for key in event_keys})}."
        )
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if OUTPUT_DIR.exists() and not any(OUTPUT_DIR.iterdir()):
        pass
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    maps_dir = OUTPUT_DIR / "prediction_reference_maps"
    maps_dir.mkdir()

    event_rows = []
    pixel_rows = []
    total = {key: 0 for key in ("TP", "FP", "TN", "FN")}

    with torch.inference_mode():
        for batch in loader:
            sar = batch["sar"].to(device)
            dem = batch["dem"].to(device)
            rainfall = batch["rainfall"].to(device)
            probabilities = model(sar, dem, rainfall)["probabilities"].cpu().numpy()
            references = batch["mask"].cpu().numpy()

            for index in range(sar.shape[0]):
                year = int(batch["year"][index].item())
                if year != 2025:
                    raise RuntimeError(f"Unexpected non-test year in batch: {year}")
                district = str(batch["district"][index])
                event_id = str(batch["event_id"][index])
                prob = probabilities[index, 0].astype(np.float32)
                ref = (references[index, 0] >= 0.5).astype(np.uint8)
                pred = (prob >= threshold).astype(np.uint8)
                metrics = pixel_metrics(pred, ref)
                total = {key: total[key] + metrics[key] for key in total}
                event_rows.append(
                    {"Split": "Test", "Year": year, "District": district,
                     "Event_ID": event_id, **metrics}
                )

                safe_district = re.sub(r"[^A-Za-z0-9_-]", "_", district)
                safe_event = re.sub(r"[^A-Za-z0-9_-]", "_", event_id)
                map_name = f"{safe_district}_{year}_{safe_event}"
                plot_event_maps(
                    prob, pred, ref,
                    f"{district} | {year} | {event_id}",
                    maps_dir / f"{map_name}.png", threshold,
                )
                for y in range(prob.shape[0]):
                    for x in range(prob.shape[1]):
                        pixel_rows.append(
                            {"District": district, "Year": year, "Event_ID": event_id,
                             "Pixel_X": x, "Pixel_Y": y,
                             "Flood_Probability": float(prob[y, x]),
                             "Predicted_Label": int(pred[y, x]),
                             "Actual_Label": int(ref[y, x])}
                        )

    if len(event_rows) != 60 or len(pixel_rows) != 6000:
        raise RuntimeError(
            f"Unexpected output size: {len(event_rows)} events, {len(pixel_rows)} pixels."
        )

    event_df = pd.DataFrame(event_rows)
    pixel_df = pd.DataFrame(pixel_rows)
    event_df.to_csv(OUTPUT_DIR / "event_metrics_2025.csv", index=False)
    pixel_df.to_csv(OUTPUT_DIR / "prediction_pixels_2025.csv", index=False)

    pooled = pixel_metrics(
        pixel_df["Predicted_Label"].to_numpy(),
        pixel_df["Actual_Label"].to_numpy(),
    )
    overall = {
        "test_year": 2025,
        "number_of_events": len(event_df),
        "threshold": threshold,
        "accuracy": pooled["Accuracy"],
        "precision": pooled["Precision"],
        "recall": pooled["Recall"],
        "dice": pooled["Dice"],
        "iou": pooled["IoU"],
        "TP": pooled["TP"], "FP": pooled["FP"],
        "TN": pooled["TN"], "FN": pooled["FN"],
    }
    pd.DataFrame([overall]).to_csv(OUTPUT_DIR / "overall_metrics_2025.csv", index=False)

    event_summary = (
        event_df.groupby(["Year"], as_index=False)[METRIC_COLUMNS]
        .mean()
    )
    event_summary["Split"] = "Test"
    event_summary["Events"] = event_df.groupby("Year").size().to_numpy()
    comparison_event_path = PROJECT_DIR / "train_val_mask_comparison" / "year_wise_metrics.csv"
    if comparison_event_path.is_file():
        previous = pd.read_csv(comparison_event_path)
        combined = pd.concat([previous, event_summary[previous.columns]], ignore_index=True)
        combined.to_csv(OUTPUT_DIR / "comparison_event_averaged_2021_2025.csv", index=False)

    pooled_path = PROJECT_DIR / "train_val_mask_comparison" / "overall_pixel_metrics.csv"
    if pooled_path.is_file():
        previous = pd.read_csv(pooled_path)
        test_row = {column: np.nan for column in previous.columns}
        test_row.update({"Split": "Test", "Events": len(event_df), **pooled})
        pd.concat([previous, pd.DataFrame([test_row])[previous.columns]], ignore_index=True).to_csv(
            OUTPUT_DIR / "comparison_pooled_pixels_2021_2025.csv", index=False
        )

    metadata = {
        "experiment": "Held-out synthetic temporal test",
        "test_year": 2025,
        "events": len(event_df),
        "pixels_per_event": 100,
        "input_window": "Pre-Flood SAR VV; rainfall Days 1-10; static DEM",
        "threshold": threshold,
        "checkpoint_epoch": int(checkpoint.get("epoch")),
        "checkpoint_sha256": sha256(checkpoint_path),
        "freeze_manifest_sha256": sha256(FROZEN_DIR / "freeze_manifest.json"),
        "device": str(device),
        "evaluated_at": datetime.now().astimezone().isoformat(),
        "interpretation": "Simulation-based results; synthetic inputs and reference masks, not real-world validation.",
    }
    (OUTPUT_DIR / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    print("2025 frozen-model evaluation complete.")
    print(f"Events: {len(event_df)} | Pixels: {len(pixel_df)} | Threshold: {threshold:.2f}")
    print("Pooled pixel metrics:")
    for name in ("Precision", "Recall", "Dice", "IoU", "Accuracy"):
        print(f"  {name}: {pooled[name]:.4f}")
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
