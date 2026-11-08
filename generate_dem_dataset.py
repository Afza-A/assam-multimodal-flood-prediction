import os
import numpy as np
import pandas as pd

# 1. Define matching spatial anchors from your editor window
DISTRICTS = {
    "Nagaon": {"lat": 26.34, "lon": 92.68},
    "Hojai": {"lat": 26.00, "lon": 92.85},
    "Morigaon": {"lat": 26.24, "lon": 92.34}
}

# 2. Match the exact spatial grid structure from your previous data
grid_size = 10  # 10x10 matrix simulating 1 image grid layout per district
pixel_resolution = 0.001 

rows = []
np.random.seed(42)  # Controlled seed for realistic micro-roughness anomalies

print("Modeling Kopili River basin digital elevation profiles...")

# Establish matching spatial framework
for district_name, coords in DISTRICTS.items():
    center_lat = coords["lat"]
    center_lon = coords["lon"]
    
    for x_idx in range(grid_size):
        for y_idx in range(grid_size):
            # Compute exact pixel coordinate alignment
            pixel_lat = center_lat + (y_idx - grid_size // 2) * pixel_resolution
            pixel_lon = center_lon + (x_idx - grid_size // 2) * pixel_resolution
            
            # Simulate a river valley corridor crossing diagonally (where x_idx == y_idx)
            distance_to_river = abs(x_idx - y_idx)
            
            # Low elevation baseline along the channel rising up the slopes
            base_elevation = 55.0 + (distance_to_river * 4.5)
            
            # Add micro-terrain roughness noise
            roughness = np.random.normal(0, 0.3)
            final_elevation = max(0.0, base_elevation + roughness)
            
            rows.append({
                "District": district_name,
                "Pixel_X": x_idx,
                "Pixel_Y": y_idx,
                "Latitude": round(pixel_lat, 5),
                "Longitude": round(pixel_lon, 5),
                "Elevation_m": round(final_elevation, 2)
            })

# 3. Export to CSV pipeline
df = pd.DataFrame(rows)
output_csv = "synthetic_assam_dem.csv"
df.to_csv(output_csv, index=False)

print(f"DEM Matrix created successfully! Saved as '{output_csv}'")
print(f"Generated {len(df)} total aligned spatial elevation points.")
