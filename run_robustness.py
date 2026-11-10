"""Inference-only missing/noisy-modality tests for the frozen model on 2025."""

from __future__ import annotations

import hashlib
import json
import os
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
OUTPUT_DIR = PROJECT_DIR / "robustness_experiments"
BATCH_SIZE = 4
THRESHOLD = 0.50
NOISE_SIGMAS = [0.25, 0.50, 1.00]
NOISE_SEED = 24051
EPSILON = 1e-8
METRICS = ["Precision", "Recall", "Dice", "IoU", "Accuracy"]
MODALITIES = ["sar", "dem", "rainfall"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_frozen():
    manifest_path = FROZEN_DIR / "freeze_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    problems = []
    for filename, expected in manifest["checksums_sha256"].items():
        path = FROZEN_DIR / filename
        if not path.is_file():
            problems.append(f"missing {filename}")
        elif sha256(path).lower() != expected.lower():
            problems.append(f"hash mismatch {filename}")
    if problems:
        raise RuntimeError("Frozen model integrity check failed: " + "; ".join(problems))
    threshold_data = json.loads(
        (FROZEN_DIR / "best_threshold.json").read_text(encoding="utf-8-sig")
    )
    threshold = float(threshold_data.get("best_threshold", threshold_data.get("threshold")))
    if threshold != THRESHOLD or float(manifest["selected_threshold"]) != THRESHOLD:
        raise RuntimeError("Frozen threshold must remain 0.50.")
    root_norm = PROJECT_DIR / "normalization_stats.json"
    if not root_norm.is_file() or sha256(root_norm) != sha256(FROZEN_DIR / "normalization_stats.json"):
        raise RuntimeError("Project and frozen normalization files differ.")
    return manifest


def counts(prediction, reference):
    pred = np.asarray(prediction, dtype=bool)
    ref = np.asarray(reference, dtype=bool)
    return (
        int(np.logical_and(pred, ref).sum()),
        int(np.logical_and(pred, ~ref).sum()),
        int(np.logical_and(~pred, ~ref).sum()),
        int(np.logical_and(~pred, ref).sum()),
    )


def metrics(tp, fp, tn, fn):
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "Precision": tp / (tp + fp + EPSILON),
        "Recall": tp / (tp + fn + EPSILON),
        "Dice": (2 * tp) / (2 * tp + fp + fn + EPSILON),
        "IoU": tp / (tp + fp + fn + EPSILON),
        "Accuracy": (tp + tn) / (tp + tn + fp + fn + EPSILON),
    }


def add_noise(value, sigma, generator):
    noise = torch.randn(
        value.shape,
        dtype=value.dtype,
        device=value.device,
        generator=generator,
    ) * sigma
    return value + noise


def run_condition(
    model, loader, device, condition, missing_modality=None,
    noise_modality=None, sigma=None, seed=None,
):
    model.eval()
    totals = [0, 0, 0, 0]
    event_rows = []
    generator = None
    if noise_modality:
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)

    with torch.inference_mode():
        for batch in loader:
            inputs = {name: batch[name].to(device) for name in MODALITIES}
            if missing_modality:
                # Inputs are normalized first, so zero means the training mean.
                inputs[missing_modality] = torch.zeros_like(inputs[missing_modality])
            elif noise_modality:
                inputs[noise_modality] = add_noise(
                    inputs[noise_modality], sigma, generator
                )

            probability_batch = model(
                inputs["sar"], inputs["dem"], inputs["rainfall"]
            )["probabilities"].cpu().numpy()
            reference_batch = batch["mask"].cpu().numpy()
            for index in range(len(probability_batch)):
                prediction = probability_batch[index, 0] >= THRESHOLD
                reference = reference_batch[index, 0] >= 0.5
                event_counts = counts(prediction, reference)
                totals = [left + right for left, right in zip(totals, event_counts)]
                event_rows.append({
                    "Condition": condition,
                    "Year": int(batch["year"][index].item()),
                    "District": str(batch["district"][index]),
                    "Event_ID": str(batch["event_id"][index]),
                    **metrics(*event_counts),
                })
    return metrics(*totals), event_rows


def main():
    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        raise FileExistsError(f"{OUTPUT_DIR} already contains results; move it before a new run.")
    frozen_manifest = verify_frozen()

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
    if int(checkpoint.get("epoch", -1)) != int(frozen_manifest["checkpoint_epoch"]):
        raise RuntimeError("Loaded checkpoint epoch does not match the manifest.")
    model.eval()

    dataset = AssamFloodDataset("test")
    if len(dataset) != 60 or {int(key[1]) for key in dataset.events} != {2025}:
        raise RuntimeError("Expected exactly 60 test events from 2025.")
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Condition names make the kept modalities explicit in the output table.
    conditions = [
        ("Full model", None, None, None),
        ("SAR+DEM only (rainfall missing)", "Missing Rainfall", None, None),
        ("SAR+Rainfall only (DEM missing)", "Missing DEM", None, None),
        ("DEM+Rainfall only (SAR missing)", "Missing SAR", None, None),
    ]
    for modality in MODALITIES:
        for sigma in NOISE_SIGMAS:
            label = f"{modality.upper()} Gaussian noise sigma={sigma:.2f}"
            conditions.append((label, None, modality, sigma))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    overall_rows = []
    event_rows = []
    baseline_full = None
    for index, (label, missing_condition, noise_modality, sigma) in enumerate(conditions):
        condition_seed = NOISE_SEED + index
        pooled, events = run_condition(
            model, loader, device, label,
            missing_modality=(missing_condition.removeprefix("Missing ").lower()
                              if missing_condition else None),
            noise_modality=noise_modality, sigma=sigma, seed=condition_seed,
        )
        row = {
            "Condition": label,
            "Type": "full reference" if label == "Full model" else (
                "missing modality" if missing_condition else "noisy modality"
            ),
            "Noise_Modality": noise_modality or "",
            "Noise_Sigma_Training_Std_Units": sigma if sigma is not None else "",
            "Noise_Seed": condition_seed if noise_modality else "",
            "Threshold": THRESHOLD,
            "Events": len(events),
            **pooled,
        }
        overall_rows.append(row)
        event_rows.extend(events)
        if label == "Full model":
            baseline_full = pooled
        print(f"{label}: Dice={pooled['Dice']:.4f}, IoU={pooled['IoU']:.4f}")

    overall = pd.DataFrame(overall_rows)
    for metric in METRICS:
        overall[f"Delta_{metric}_vs_Full"] = overall[metric] - baseline_full[metric]
    overall.to_csv(OUTPUT_DIR / "robustness_overall_metrics.csv", index=False)
    pd.DataFrame(event_rows).to_csv(OUTPUT_DIR / "robustness_event_metrics.csv", index=False)

    # Cross-check the unperturbed condition against the frozen Stage 2 result.
    official_path = PROJECT_DIR / "test_results_2025" / "overall_metrics_2025.csv"
    if official_path.is_file():
        official = pd.read_csv(official_path).iloc[0]
        if not np.isclose(baseline_full["Dice"], float(official["dice"]), atol=1e-7):
            raise RuntimeError("Full-model robustness reference does not match Stage 2 test Dice.")

    chart = overall.set_index("Condition")[METRICS].plot(
        kind="bar", figsize=(15, 7), ylim=(0, 1), width=0.82
    )
    chart.set_ylabel("Pooled pixel score")
    chart.set_xlabel("")
    chart.set_title("Frozen model robustness on synthetic 2025 test events")
    chart.tick_params(axis="x", labelrotation=35)
    chart.grid(axis="y", alpha=0.25)
    chart.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "robustness_comparison.png", dpi=180)
    plt.close()

    manifest = {
        "experiment": "Inference-only missing/noisy modality robustness",
        "created_at": datetime.now().astimezone().isoformat(),
        "dataset": "Fully synthetic held-out 2025 split",
        "test_events": 60,
        "frozen_checkpoint_epoch": int(checkpoint["epoch"]),
        "threshold": THRESHOLD,
        "missing_modality_strategy": "Set normalized modality tensor to zero (training-mean input); do not retrain.",
        "noise": "Additive zero-mean Gaussian noise in normalized training-standard-deviation units.",
        "noise_sigmas": NOISE_SIGMAS,
        "noise_seed_base": NOISE_SEED,
        "no_retraining_or_threshold_tuning": True,
        "frozen_manifest_sha256": sha256(FROZEN_DIR / "freeze_manifest.json"),
        "frozen_checkpoint_sha256": sha256(checkpoint_path),
        "outputs": sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_file()),
    }
    (OUTPUT_DIR / "robustness_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Robustness experiment complete. Results: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
