
import json
import numpy as np
import pandas as pd

sar = pd.read_csv("synthetic_assam_sar_dataset_10x10.csv")
dem = pd.read_csv("synthetic_assam_dem.csv")
rain = pd.read_csv("synthetic_assam_rainfall_june.csv")

TRAIN_YEARS = [2021, 2022, 2023]

# Training SAR: pre-flood observations only.
sar_train = sar[
    (sar["Year"].isin(TRAIN_YEARS))
    & (sar["Phase"] == "Pre-Flood")
]["SAR_VV_dB"]

# DEM is static and shared across all years.
dem_train = dem["Elevation_m"]

# Training rainfall: days 1-10 only.
rain_train = rain[
    (rain["Year"].isin(TRAIN_YEARS))
    & (rain["Day"].between(1, 10))
]["Rainfall_mm"]

def statistics(values):
    values = values.to_numpy(dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std())
    }

stats = {
    "sar": statistics(sar_train),
    "dem": statistics(dem_train),
    "rainfall": statistics(rain_train)
}

for name, values in stats.items():
    if values["std"] <= 0:
        raise ValueError(f"Invalid standard deviation for {name}")

with open("normalization_stats.json", "w") as f:
    json.dump(stats, f, indent=4)

print("\nTRAINING-ONLY NORMALIZATION STATISTICS")
print(json.dumps(stats, indent=4))
print("\nSaved: normalization_stats.json")
