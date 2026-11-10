import json
import hashlib
from pathlib import Path

folder = Path("frozen_model")

with open(folder / "freeze_manifest.json") as f:
    manifest = json.load(f)

checksums = manifest["checksums_sha256"]

print("\n========== SHA-256 VERIFICATION ==========")

for filename, expected in checksums.items():

    path = folder / filename

    if not path.exists():
        print(f"MISSING: {filename}")
        continue

    actual = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()

    if actual == expected:
        print(f"PASS: {filename}")
    else:
        print(f"FAIL: {filename}")

print("\nVerification completed.")