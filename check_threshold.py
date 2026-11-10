import json

with open(
    "frozen_model/best_threshold.json", "r"
) as f:
    threshold_data = json.load(f)

print("Threshold information:")
print(threshold_data)
