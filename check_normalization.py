import json

with open(
    "frozen_model/normalization_stats.json", "r"
) as f:
    stats = json.load(f)

print("Frozen normalization statistics:")
print(json.dumps(stats, indent=4))