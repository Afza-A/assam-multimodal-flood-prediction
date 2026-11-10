import json
from pathlib import Path

folder = Path("frozen_model")

manifest_path = folder / "freeze_manifest.json"

if not manifest_path.exists():
    print("ERROR: Freeze manifest not found!")
else:
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    print("\n========== FREEZE MANIFEST ==========")

    print(json.dumps(manifest, indent=4))

    print("\n========== FILE CHECK ==========")

    for filename in [
        "best_flood_model.pth",
        "best_threshold.json",
        "normalization_stats.json"
    ]:
        path = folder / filename
        print(
            filename,
            "EXISTS" if path.exists() else "MISSING"
        )