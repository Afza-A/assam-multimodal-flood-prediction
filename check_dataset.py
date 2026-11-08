
import pandas as pd

base = "."

rain = pd.read_csv(f"{base}/synthetic_assam_rainfall_june.csv")
ref = pd.read_csv(f"{base}/synthetic_assam_flood_reference.csv")

# Calculate flood extent for each independent event.
event_flood = (
    ref.groupby(["District", "Year", "Event_ID"])["Flooded_Label"]
    .mean()
    .reset_index(name="Flood_Percentage")
)

event_flood["Flood_Percentage"] *= 100

# Calculate rainfall totals for each event.
event_rain = (
    rain.groupby(["District", "Year", "Event_ID", "Day"])["Rainfall_mm"]
    .mean()
    .reset_index()
)

rainfall_summary = (
    event_rain.groupby(["District", "Year", "Event_ID"])
    .agg(
        Pre_Rainfall=("Rainfall_mm", lambda x: x.iloc[:10].sum()),
        Peak_Rainfall=("Rainfall_mm", lambda x: x.iloc[10:20].sum()),
        Post_Rainfall=("Rainfall_mm", lambda x: x.iloc[20:30].sum())
    )
    .reset_index()
)

combined = event_flood.merge(
    rainfall_summary,
    on=["District", "Year", "Event_ID"]
)

print("\n1. EVENT FLOOD EXTENT STATISTICS")
print(combined["Flood_Percentage"].describe().round(2))

print("\n2. FLOOD EXTENT BY YEAR")
print(
    combined.groupby("Year")["Flood_Percentage"]
    .agg(["mean", "min", "max", "std"])
    .round(2)
)

print("\n3. RAINFALL AND FLOOD EXTENT CORRELATION")
print(
    combined[
        ["Pre_Rainfall", "Peak_Rainfall", "Flood_Percentage"]
    ].corr().round(3)
)

print("\n4. SAMPLE EVENTS")
print(combined.head(10).round(2).to_string(index=False))

print("\n5. CHECK EVENT COUNT")
print("Total events:", len(combined))
print("Expected events: 300")
