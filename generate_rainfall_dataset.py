import os
import numpy as np
import pandas as pd

# 1. Define matching spatial anchors from your editor window
DISTRICTS = {
    "Nagaon": {"lat": 26.34, "lon": 92.68},
    "Hojai": {"lat": 26.00, "lon": 92.85},
    "Morigaon": {"lat": 26.24, "lon": 92.34}
}

# 2. Structural dimensions matching the SAR grid perfectly
years = [2021, 2022, 2023, 2024, 2025]
days_in_june = 30
grid_size = 10  # 10x10 structural image grid matching SAR dimensions
pixel_resolution = 0.001 

rows = []
np.random.seed(101)  # Fixed seed for repeatable meteorological features

print("Generating perfectly aligned spatial & temporal rainfall cubes...")

# Outer Loops: Establish Space (Static Lat/Lon Framework)
for district_name, coords in DISTRICTS.items():
    center_lat = coords["lat"]
    center_lon = coords["lon"]
    
    for x_idx in range(grid_size):
        for y_idx in range(grid_size):
            # Compute exact pixel coordinate alignment
            pixel_lat = center_lat + (y_idx - grid_size // 2) * pixel_resolution
            pixel_lon = center_lon + (x_idx - grid_size // 2) * pixel_resolution
            
            # Spatial topography variation index near simulated Kopili line
            topography_bonus = (x_idx + y_idx) * 0.4
            
            # Inner Loops: Establish Time (Years and Days progression)
            for year in years:
                for day in range(1, days_in_june + 1):
                    
                    # Distribute rainfall values based on phase segments
                    if day <= 10:
                        phase = "Pre-Flood"
                        base_rain = 15.0     # Monsoon setup showers
                        noise_scale = 4.0
                    elif day <= 20:
                        phase = "Peak-Flood"
                        base_rain = 60.0     # Cloudburst events causing river swelling
                        noise_scale = 10.0
                    else:
                        phase = "Post-Flood"
                        base_rain = 10.0     # Post-flood runoff showers
                        noise_scale = 3.0
                    
                    # Calculate final value
                    daily_rain = max(0.0, base_rain + topography_bonus + np.random.normal(0, noise_scale))
                    
                    rows.append({
                        "Year": year,
                        "Month": "June",
                        "Day": day,
                        "Phase": phase,
                        "District": district_name,
                        "Pixel_X": x_idx,
                        "Pixel_Y": y_idx,
                        "Latitude": round(pixel_lat, 5),
                        "Longitude": round(pixel_lon, 5),
                        "Rainfall_mm": round(daily_rain, 2)
                    })

# 3. Export to CSV structured exactly like your SAR pipeline
df = pd.DataFrame(rows)
output_csv = "synthetic_assam_rainfall_june.csv"
df.to_csv(output_csv, index=False)

print(f"Data matrix created successfully! Saved as '{output_csv}'")
print(f"Total structured multi-dimensional rows: {len(df)}")
