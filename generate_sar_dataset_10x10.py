"""
Generate a synthetic 10x10 SAR dataset aligned with the synthetic
rainfall and DEM grids in this project.

IMPORTANT:
- This produces SYNTHETIC test data, not real Sentinel-1 observations.
- It writes a NEW output filename and does not overwrite the original CSV.
- Spatial coordinates and district anchors intentionally match the DEM/rainfall scripts.
"""
import numpy as np
import pandas as pd

DISTRICTS = {
    "Nagaon": {"lat": 26.34, "lon": 92.68},
    "Hojai": {"lat": 26.00, "lon": 92.85},
    "Morigaon": {"lat": 26.24, "lon": 92.34},
}
YEARS = [2021, 2022, 2023, 2024, 2025]
PHASES = ["Pre-Flood", "Peak-Flood", "Post-Flood"]
GRID_SIZE = 10
PIXEL_RESOLUTION = 0.001

rows = []
rng = np.random.default_rng(42)

for district, center in DISTRICTS.items():
    for x_idx in range(GRID_SIZE):
        for y_idx in range(GRID_SIZE):
            lat = center["lat"] + (y_idx - GRID_SIZE // 2) * PIXEL_RESOLUTION
            lon = center["lon"] + (x_idx - GRID_SIZE // 2) * PIXEL_RESOLUTION

            # Synthetic river corridor: two diagonal pixel lines.
            river_channel = (x_idx == y_idx) or (x_idx == y_idx + 1)
            distance_to_river = abs(x_idx - y_idx)

            for year in YEARS:
                for phase in PHASES:
                    if phase == "Pre-Flood":
                        flooded = int(river_channel)
                    elif phase == "Peak-Flood":
                        flooded = int(distance_to_river <= 4)
                    else:  # Post-Flood
                        flooded = int(distance_to_river <= 1)

                    if flooded:
                        vv = -23.5 + rng.normal(0, 0.8)
                        vh = -29.0 + rng.normal(0, 1.1)
                    else:
                        vv = -11.0 + rng.normal(0, 1.4)
                        vh = -17.5 + rng.normal(0, 1.6)

                    rows.append({
                        "Year": year,
                        "Phase": phase,
                        "District": district,
                        "Pixel_X": x_idx,
                        "Pixel_Y": y_idx,
                        "Latitude": round(lat, 5),
                        "Longitude": round(lon, 5),
                        "SAR_VV_dB": round(vv, 2),
                        "SAR_VH_dB": round(vh, 2),
                        "Flooded_Label": flooded,
                    })

df = pd.DataFrame(rows)
output_file = "synthetic_assam_sar_dataset_10x10.csv"
df.to_csv(output_file, index=False)

print(f"Saved {output_file}")
print(f"Rows: {len(df):,}")
print(f"Grid: {GRID_SIZE}x{GRID_SIZE} per district/year/phase")
print("Reminder: these are synthetic SAR-like values, not satellite measurements.")
