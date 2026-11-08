
"""
Integrated synthetic flood-event generator
Study region: Assam-inspired synthetic grids labelled Nagaon, Hojai, Morigaon.

IMPORTANT:
- This is a simplified synthetic simulator, NOT a calibrated hydrological model.
- The 10x10 grids are synthetic and are not verified geographic rasters.
- The primary flood reference mask is derived from simulated water depth.
- SAR VV is generated as an observation of the simulated surface state.
- No VH channel or SAR-derived flood label is generated.
"""

from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

SEED = 42
GRID_SIZE = 10
YEARS = [2021, 2022, 2023, 2024, 2025]
EVENTS_PER_YEAR = 20
DAYS_PER_EVENT = 30

DISTRICTS = {
    "Nagaon": {"lat": 26.34, "lon": 92.68, "base_elevation": 58},
    "Hojai": {"lat": 26.00, "lon": 92.85, "base_elevation": 65},
    "Morigaon": {"lat": 26.24, "lon": 92.34, "base_elevation": 54},
}

OUTPUT_DIR = Path(__file__).resolve().parent

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------
# 2. CREATE STATIC TERRAIN / DEM
# ---------------------------------------------------------

def create_dem():
    dem_rows = []
    terrain = {}

    for district, info in DISTRICTS.items():
        grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)

        # A gently curved synthetic river valley.
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                river_x = 4.5 + 0.20 * y
                river_distance = abs(x - river_x)

                # Elevation rises gradually away from the river.
                elevation = (
                    info["base_elevation"]
                    + 3.0 * river_distance
                    + 0.8 * y
                    + rng.normal(0, 0.35)
                )

                grid[y, x] = elevation

        # Approximate river proximity and relative elevation.
        river_distance_grid = np.zeros_like(grid)

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                river_x = 4.5 + 0.20 * y
                river_distance_grid[y, x] = abs(x - river_x)

        low = grid.min()
        high = grid.max()
        elevation_norm = (grid - low) / (high - low + 1e-8)

        # Higher susceptibility near the valley and at lower elevations.
        river_proximity = np.exp(-river_distance_grid / 2.2)

        susceptibility = (
            0.35
            + 0.90 * river_proximity
            + 0.35 * (1.0 - elevation_norm)
        )

        terrain[district] = {
            "elevation": grid,
            "susceptibility": susceptibility,
            "river_distance": river_distance_grid,
        }

        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                dem_rows.append({
                    "District": district,
                    "Pixel_X": x,
                    "Pixel_Y": y,
                    "Latitude": info["lat"] + (y - 4.5) * 0.001,
                    "Longitude": info["lon"] + (x - 4.5) * 0.001,
                    "Elevation_m": round(float(grid[y, x]), 3),
                })

    return pd.DataFrame(dem_rows), terrain


# ---------------------------------------------------------
# 3. GENERATE EVENT-SPECIFIC DAILY RAINFALL
# ---------------------------------------------------------


def generate_rainfall(event_severity, susceptibility):
    """
    Generate 30 days of rainfall with independently varying
    pre-flood, peak-flood and post-flood rainfall intensities.
    """

    rainfall = np.zeros(
        (DAYS_PER_EVENT, GRID_SIZE, GRID_SIZE),
        dtype=np.float32
    )

    # Independent daily patterns for each phase.
    pre_pattern = rng.gamma(shape=2.0, scale=1.0, size=10)
    peak_pattern = rng.gamma(shape=2.2, scale=1.0, size=10)
    post_pattern = rng.gamma(shape=1.5, scale=1.0, size=10)

    pre_pattern /= pre_pattern.mean() + 1e-8
    peak_pattern /= peak_pattern.mean() + 1e-8
    post_pattern /= post_pattern.mean() + 1e-8

    # Each phase has its own intensity variation.
    pre_severity = np.clip(
        event_severity + rng.normal(0, 0.20), 0.15, 1.30
    )
    peak_severity = np.clip(
        event_severity + rng.normal(0, 0.30), 0.15, 1.40
    )
    post_severity = np.clip(
        event_severity + rng.normal(0, 0.25), 0.10, 1.30
    )

    pre_base = 5.0 + 9.0 * pre_severity
    peak_base = 9.0 + 19.0 * peak_severity
    post_base = 2.0 + 4.0 * post_severity

    for day in range(30):

        if day < 10:
            daily_base = pre_base * pre_pattern[day]

        elif day < 20:
            daily_base = peak_base * peak_pattern[day - 10]

        else:
            daily_base = post_base * post_pattern[day - 20]

        spatial_noise = rng.normal(
            loc=1.0,
            scale=0.08,
            size=(GRID_SIZE, GRID_SIZE)
        )

        local_rainfall = (
            daily_base
            * spatial_noise
            * (0.95 + 0.08 * susceptibility)
        )

        rainfall[day] = np.clip(local_rainfall, 0, None)

    return rainfall



# ---------------------------------------------------------
# 4. SIMULATE WATER DEPTH AND FLOOD REFERENCE MASK
# ---------------------------------------------------------

def simulate_water_depth(rainfall, susceptibility):
    """
    Simplified rainfall-runoff and terrain-susceptibility model.

    Water depth is a synthetic simulation variable. It is not
    calibrated against observed flood depths.
    """

    pre_rain_total = rainfall[0:10].sum(axis=0)
    peak_rain_total = rainfall[10:20].sum(axis=0)
    post_rain_total = rainfall[20:30].sum(axis=0)

    # Pre-event surface wetness.
    pre_depth = np.maximum(
        0.0,
        (pre_rain_total * 0.0015 * susceptibility) - 0.035
    )

    # Peak water depth depends on antecedent and peak rainfall,
    # modified by terrain susceptibility.
    runoff_index = peak_rain_total + 0.25 * pre_rain_total

    peak_depth = np.maximum(
        0.0,
        (runoff_index * 0.0015 * susceptibility) - 0.15
    )

    # Approximate recession after the peak event.
    post_depth = np.maximum(
        0.0,
        0.35 * peak_depth
        + (post_rain_total * 0.0008 * susceptibility)
        - 0.025
    )

    return pre_depth, peak_depth, post_depth


# ---------------------------------------------------------
# 5. GENERATE SAR VV OBSERVATIONS
# ---------------------------------------------------------

def generate_sar_vv(water_depth):
    """
    Generate a single-channel Sentinel-1-like VV observation.

    Water-covered pixels generally receive lower backscatter
    in this simplified simulation. Noise is added independently.
    """

    flooded = water_depth >= 0.15

    vv = np.where(
        flooded,
        rng.normal(-21.5, 1.5, size=water_depth.shape),
        rng.normal(-10.5, 2.0, size=water_depth.shape)
    )

    # Slight wet-soil effect in non-flooded pixels.
    vv = np.where(
        (~flooded) & (water_depth > 0.02),
        vv - 1.0,
        vv
    )

    return np.clip(vv, -30, 0)


# ---------------------------------------------------------
# 6. GENERATE ALL ALIGNED DATASETS
# ---------------------------------------------------------

def main():
    dem_df, terrain = create_dem()

    rainfall_rows = []
    sar_rows = []
    reference_rows = []

    total_events = len(DISTRICTS) * len(YEARS) * EVENTS_PER_YEAR

    print(f"Generating {total_events} synthetic flood events...")

    for district in DISTRICTS:
        susceptibility = terrain[district]["susceptibility"]

        for year in YEARS:
            for event_number in range(1, EVENTS_PER_YEAR + 1):

                event_id = f"E{event_number:03d}"

                # Each event gets a different rainfall severity.
                # 2025 uses a shifted severity range as a
                # synthetic distribution-shift stress test.
                if year == 2025:
                    event_severity = rng.uniform(0.65, 1.15)
                else:
                    event_severity = rng.uniform(0.35, 1.00)

                rainfall = generate_rainfall(
                    event_severity,
                    susceptibility
                )

                pre_depth, peak_depth, post_depth = (
                    simulate_water_depth(rainfall, susceptibility)
                )

                pre_vv = generate_sar_vv(pre_depth)
                peak_vv = generate_sar_vv(peak_depth)
                post_vv = generate_sar_vv(post_depth)

                phase_data = [
                    ("Pre-Flood", pre_vv, pre_depth),
                    ("Peak-Flood", peak_vv, peak_depth),
                    ("Post-Flood", post_vv, post_depth),
                ]

                # Rainfall CSV: one record per day and pixel.
                for day in range(DAYS_PER_EVENT):
                    phase = (
                        "Pre-Flood" if day < 10
                        else "Peak-Flood" if day < 20
                        else "Post-Flood"
                    )

                    for y in range(GRID_SIZE):
                        for x in range(GRID_SIZE):
                            rainfall_rows.append({
                                "District": district,
                                "Year": year,
                                "Event_ID": event_id,
                                "Day": day + 1,
                                "Phase": phase,
                                "Pixel_X": x,
                                "Pixel_Y": y,
                                "Rainfall_mm": round(
                                    float(rainfall[day, y, x]), 3
                                ),
                            })

                # SAR CSV: one VV observation per phase and pixel.
                for phase, vv_grid, depth_grid in phase_data:
                    for y in range(GRID_SIZE):
                        for x in range(GRID_SIZE):
                            sar_rows.append({
                                "District": district,
                                "Year": year,
                                "Event_ID": event_id,
                                "Phase": phase,
                                "Pixel_X": x,
                                "Pixel_Y": y,
                                "SAR_VV_dB": round(
                                    float(vv_grid[y, x]), 3
                                ),
                            })

                # Independent reference: peak water depth and mask.
                for y in range(GRID_SIZE):
                    for x in range(GRID_SIZE):
                        depth = float(peak_depth[y, x])

                        reference_rows.append({
                            "District": district,
                            "Year": year,
                            "Event_ID": event_id,
                            "Pixel_X": x,
                            "Pixel_Y": y,
                            "Peak_Water_Depth_m": round(depth, 4),
                            "Flooded_Label": int(depth >= 0.15),
                        })

    rainfall_df = pd.DataFrame(rainfall_rows)
    sar_df = pd.DataFrame(sar_rows)
    reference_df = pd.DataFrame(reference_rows)

    # Save all files in the same folder as this Python script.
    dem_df.to_csv(
        OUTPUT_DIR / "synthetic_assam_dem.csv", index=False
    )
    rainfall_df.to_csv(
        OUTPUT_DIR / "synthetic_assam_rainfall_june.csv", index=False
    )
    sar_df.to_csv(
        OUTPUT_DIR / "synthetic_assam_sar_dataset_10x10.csv", index=False
    )
    reference_df.to_csv(
        OUTPUT_DIR / "synthetic_assam_flood_reference.csv", index=False
    )

    # -----------------------------------------------------
    # 7. BASIC DATASET VERIFICATION
    # -----------------------------------------------------

    print("\nDATA GENERATION COMPLETED")
    print("------------------------------------")
    print("DEM rows:", len(dem_df))
    print("Rainfall rows:", len(rainfall_df))
    print("SAR rows:", len(sar_df))
    print("Reference rows:", len(reference_df))

    print("\nUnique events:", rainfall_df[
        ["District", "Year", "Event_ID"]
    ].drop_duplicates().shape[0])

    print("\nFlooded pixel percentage:")
    print(
        round(reference_df["Flooded_Label"].mean() * 100, 2),
        "%"
    )

    print("\nFlooded pixel percentage by year:")
    print(
        reference_df.groupby("Year")["Flooded_Label"]
        .mean()
        .mul(100)
        .round(2)
    )

    print("\nFiles saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
