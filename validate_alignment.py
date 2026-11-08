
import pandas as pd

dem = pd.read_csv("synthetic_assam_dem.csv")
rain = pd.read_csv("synthetic_assam_rainfall_june.csv")
sar = pd.read_csv("synthetic_assam_sar_dataset_10x10.csv")
ref = pd.read_csv("synthetic_assam_flood_reference.csv")

event_keys = ["District", "Year", "Event_ID"]
pixel_keys = event_keys + ["Pixel_X", "Pixel_Y"]

# 1. Filter forecasting inputs.
pre_sar = sar[sar["Phase"] == "Pre-Flood"]
pre_rain = rain[rain["Day"].between(1, 10)]

print("\n1. ROW COUNTS")
print("DEM:", len(dem))
print("Pre-flood SAR:", len(pre_sar))
print("Pre-flood rainfall:", len(pre_rain))
print("Reference masks:", len(ref))

# 2. Check pixels per event.
for name, df in [
    ("Pre-flood SAR", pre_sar),
    ("Pre-flood rainfall", pre_rain),
    ("Reference", ref)
]:
    counts = df.groupby(event_keys).size()
    print(f"\n{name}:")
    print("Events:", len(counts))
    print("Minimum rows per event:", counts.min())
    print("Maximum rows per event:", counts.max())

# 3. Check rainfall days per event.
rain_days = pre_rain.groupby(event_keys)["Day"].nunique()
print("\n2. PRE-FLOOD RAINFALL DAYS")
print("Minimum:", rain_days.min())
print("Maximum:", rain_days.max())

# 4. Check pixel alignment between SAR and reference.
sar_pixels = set(map(tuple, pre_sar[pixel_keys].drop_duplicates().values))
ref_pixels = set(map(tuple, ref[pixel_keys].drop_duplicates().values))

print("\n3. SAR-REFERENCE PIXEL ALIGNMENT")
print("Matching:", sar_pixels == ref_pixels)
print("SAR-only pixels:", len(sar_pixels - ref_pixels))
print("Reference-only pixels:", len(ref_pixels - sar_pixels))

# 5. Check split separation.
train_years = {2021, 2022, 2023}
validation_years = {2024}
test_years = {2025}

print("\n4. TEMPORAL SPLIT")
print("Training events:", len(ref[ref["Year"].isin(train_years)][event_keys].drop_duplicates()))
print("Validation events:", len(ref[ref["Year"].isin(validation_years)][event_keys].drop_duplicates()))
print("Test events:", len(ref[ref["Year"].isin(test_years)][event_keys].drop_duplicates()))

print("\n5. CHECK FOR INVALID YEARS")
print("Years present:", sorted(ref["Year"].unique()))

print("\nVALIDATION SCRIPT COMPLETED")
