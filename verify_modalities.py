"""Verify structural/spatial/temporal compatibility of the three synthetic CSVs."""
from pathlib import Path
import pandas as pd

HERE = Path(__file__).resolve().parent
sar = pd.read_csv(HERE / "synthetic_assam_sar_dataset_10x10.csv")
rain = pd.read_csv(HERE / "synthetic_assam_rainfall_june.csv")
dem = pd.read_csv(HERE / "synthetic_assam_dem.csv")

def report(name, df):
    print(f"\n{name}: {len(df):,} rows; missing={df.isna().sum().sum()}; duplicates={df.duplicated().sum()}")

for name, df in [("SAR", sar), ("Rainfall", rain), ("DEM", dem)]:
    report(name, df)

spatial_keys = ["District", "Pixel_X", "Pixel_Y"]
coord_cols = spatial_keys + ["Latitude", "Longitude"]

def coordinate_table(df):
    return (df[coord_cols].drop_duplicates()
            .sort_values(spatial_keys).reset_index(drop=True))

sar_coords = coordinate_table(sar)
rain_coords = coordinate_table(rain)
dem_coords = coordinate_table(dem)

print("\nSPATIAL CHECKS")
print("SAR vs rainfall coordinates:", sar_coords.equals(rain_coords))
print("SAR vs DEM coordinates:", sar_coords.equals(dem_coords))
print("Rainfall vs DEM coordinates:", rain_coords.equals(dem_coords))

print("\nGRID CHECKS (each district/year/phase should contain 100 SAR pixels)")
sar_grid = sar.groupby(["District", "Year", "Phase"]).size()
print("SAR group sizes:", sorted(sar_grid.unique().tolist()))
print("SAR groups all have 100 pixels:", bool((sar_grid == 100).all()))

print("\nTEMPORAL CHECKS (each rainfall phase should have 10 daily records per pixel)")
rain_group = rain.groupby(["District", "Year", "Phase", "Pixel_X", "Pixel_Y"])["Day"].agg(["count", "min", "max"])
print("Rainfall groups all contain 10 days:", bool((rain_group["count"] == 10).all()))
print("Rainfall day ranges by phase:")
print(rain.groupby("Phase")["Day"].agg(["min", "max", "nunique"]))

print("\nYEAR CHECK")
print("SAR years:", sorted(sar["Year"].unique().tolist()))
print("Rainfall years:", sorted(rain["Year"].unique().tolist()))
print("Years match:", set(sar["Year"]) == set(rain["Year"]))

print("\nLABEL CHECK")
print(sar.groupby("Phase")["Flooded_Label"].agg(["sum", "count", "mean"]))

all_ok = (
    sar_coords.equals(rain_coords)
    and sar_coords.equals(dem_coords)
    and bool((sar_grid == 100).all())
    and bool((rain_group["count"] == 10).all())
    and set(sar["Year"]) == set(rain["Year"])
    and all(df.isna().sum().sum() == 0 for df in [sar, rain, dem])
)
print("\nSTRUCTURAL VERIFICATION:", "PASS" if all_ok else "CHECK FAILED")
print("This validates generated table structure, not real-world data authenticity or event accuracy.")
