
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader


# =========================================================
# 1. CONFIGURATION
# =========================================================

DEM_FILE = "synthetic_assam_dem.csv"
RAINFALL_FILE = "synthetic_assam_rainfall_june.csv"
SAR_FILE = "synthetic_assam_sar_dataset_10x10.csv"
REFERENCE_FILE = "synthetic_assam_flood_reference.csv"
NORMALIZATION_FILE = "normalization_stats.json"

GRID_SIZE = 10
PRE_FLOOD_DAYS = 10

TRAIN_YEARS = [2021, 2022, 2023]
VAL_YEARS = [2024]
TEST_YEARS = [2025]

EVENT_KEYS = ["District", "Year", "Event_ID"]
PIXEL_KEYS = ["Pixel_Y", "Pixel_X"]


# =========================================================
# 2. DATASET CLASS
# =========================================================

class AssamFloodDataset(Dataset):

    def __init__(self, split="train"):

        self.split = split

        if split == "train":
            selected_years = TRAIN_YEARS
        elif split == "val":
            selected_years = VAL_YEARS
        elif split == "test":
            selected_years = TEST_YEARS
        else:
            raise ValueError(
                "split must be train, val, or test"
            )

        # Load datasets.
        dem = pd.read_csv(DEM_FILE)
        rainfall = pd.read_csv(RAINFALL_FILE)
        sar = pd.read_csv(SAR_FILE)
        reference = pd.read_csv(REFERENCE_FILE)

        # Load training-only normalization statistics.
        with open(NORMALIZATION_FILE, "r") as f:
            self.norm_stats = json.load(f)

        # =================================================
        # 3. FILTER MODEL INPUTS
        # =================================================

        # Only pre-flood SAR VV.
        sar = sar[
            sar["Phase"] == "Pre-Flood"
        ].copy()

        # Only the first 10 rainfall days.
        rainfall = rainfall[
            rainfall["Day"].between(1, PRE_FLOOD_DAYS)
        ].copy()

        # =================================================
        # 4. TEMPORAL SPLIT
        # =================================================

        sar = sar[sar["Year"].isin(selected_years)]
        rainfall = rainfall[
            rainfall["Year"].isin(selected_years)
        ]
        reference = reference[
            reference["Year"].isin(selected_years)
        ]

        # =================================================
        # 5. PREPARE DEM GRIDS
        # =================================================

        self.dem_grids = {}

        for district in sorted(dem["District"].unique()):

            district_dem = dem[
                dem["District"] == district
            ].copy()

            district_dem = district_dem.sort_values(
                PIXEL_KEYS
            )

            if len(district_dem) != 100:
                raise ValueError(
                    f"Invalid DEM pixel count: {district}"
                )

            if district_dem.duplicated(PIXEL_KEYS).any():
                raise ValueError(
                    f"Duplicate DEM pixels: {district}"
                )

            self.dem_grids[district] = (
                district_dem["Elevation_m"]
                .to_numpy(dtype=np.float32)
                .reshape(1, GRID_SIZE, GRID_SIZE)
            )

            # Cache DEM coordinates for alignment checks.
            self.dem_coordinates = getattr(
                self, "dem_coordinates", {}
            )

            self.dem_coordinates[district] = (
                district_dem[PIXEL_KEYS].to_numpy()
            )

        # =================================================
        # 6. VERIFY EVENT IDENTIFIERS
        # =================================================

        sar_events = set(
            map(tuple, sar[EVENT_KEYS].drop_duplicates().values)
        )

        rainfall_events = set(
            map(tuple, rainfall[EVENT_KEYS].drop_duplicates().values)
        )

        reference_events = set(
            map(
                tuple,
                reference[EVENT_KEYS].drop_duplicates().values
            )
        )

        if not (
            sar_events == rainfall_events == reference_events
        ):
            raise ValueError(
                "Event IDs do not match across modalities."
            )

        self.events = sorted(
            reference_events,
            key=lambda x: (x[1], x[0], x[2])
        )

        self.sar = sar
        self.rainfall = rainfall
        self.reference = reference

        print(f"\n{split.upper()} DATASET INITIALIZED")
        print("Years:", selected_years)
        print("Number of events:", len(self.events))

    # -----------------------------------------------------
    # Dataset length
    # -----------------------------------------------------

    def __len__(self):
        return len(self.events)

    # -----------------------------------------------------
    # Retrieve one event
    # -----------------------------------------------------

    def __getitem__(self, index):

        district, year, event_id = self.events[index]

        # =================================================
        # 7. SELECT EVENT DATA
        # =================================================

        event_filter = (
            (self.sar["District"] == district)
            & (self.sar["Year"] == year)
            & (self.sar["Event_ID"] == event_id)
        )

        sar_event = self.sar[event_filter].copy()

        rainfall_filter = (
            (self.rainfall["District"] == district)
            & (self.rainfall["Year"] == year)
            & (self.rainfall["Event_ID"] == event_id)
        )

        rainfall_event = self.rainfall[
            rainfall_filter
        ].copy()

        reference_filter = (
            (self.reference["District"] == district)
            & (self.reference["Year"] == year)
            & (self.reference["Event_ID"] == event_id)
        )

        reference_event = self.reference[
            reference_filter
        ].copy()

        # =================================================
        # 8. SAR VV GRID
        # Shape: [1, 10, 10]
        # =================================================

        sar_event = sar_event.sort_values(PIXEL_KEYS)

        if len(sar_event) != 100:
            raise ValueError("Incorrect SAR pixel count.")

        sar_coordinates = sar_event[
            PIXEL_KEYS
        ].to_numpy()

        sar_grid = (
            sar_event["SAR_VV_dB"]
            .to_numpy(dtype=np.float32)
            .reshape(1, GRID_SIZE, GRID_SIZE)
        )

        # =================================================
        # 9. DEM GRID
        # Shape: [1, 10, 10]
        # =================================================

        dem_grid = self.dem_grids[district].copy()

        # =================================================
        # 10. RAINFALL SEQUENCE
        # Shape: [100, 10, 1]
        # =================================================

        rainfall_pivot = rainfall_event.pivot(
            index=["Pixel_Y", "Pixel_X"],
            columns="Day",
            values="Rainfall_mm"
        )

        rainfall_pivot = rainfall_pivot.sort_index()

        rainfall_pivot = rainfall_pivot.reindex(
            columns=range(1, PRE_FLOOD_DAYS + 1)
        )

        if rainfall_pivot.shape != (100, 10):
            raise ValueError(
                f"Incorrect rainfall shape: {rainfall_pivot.shape}"
            )

        if rainfall_pivot.isnull().any().any():
            raise ValueError("Missing rainfall values.")

        rainfall_array = rainfall_pivot.to_numpy(
            dtype=np.float32
        ).reshape(100, PRE_FLOOD_DAYS, 1)

        # =================================================
        # 11. INDEPENDENT REFERENCE MASK
        # Shape: [1, 10, 10]
        # =================================================

        reference_event = reference_event.sort_values(
            PIXEL_KEYS
        )

        if len(reference_event) != 100:
            raise ValueError(
                "Incorrect reference pixel count."
            )

        reference_coordinates = reference_event[
            PIXEL_KEYS
        ].to_numpy()

        mask_grid = (
            reference_event["Flooded_Label"]
            .to_numpy(dtype=np.float32)
            .reshape(1, GRID_SIZE, GRID_SIZE)
        )

        depth_grid = (
            reference_event["Peak_Water_Depth_m"]
            .to_numpy(dtype=np.float32)
            .reshape(1, GRID_SIZE, GRID_SIZE)
        )

        # =================================================
        # 12. COORDINATE ALIGNMENT
        # =================================================

        if not np.array_equal(
            sar_coordinates, reference_coordinates
        ):
            raise ValueError(
                "SAR and reference coordinates do not match."
            )

        if not np.array_equal(
            sar_coordinates,
            self.dem_coordinates[district]
        ):
            raise ValueError(
                "SAR and DEM coordinates do not match."
            )

        # =================================================
        # 13. NORMALIZATION
        # Training-only statistics are reused for all splits.
        # =================================================

        sar_grid = (
            sar_grid - self.norm_stats["sar"]["mean"]
        ) / self.norm_stats["sar"]["std"]

        dem_grid = (
            dem_grid - self.norm_stats["dem"]["mean"]
        ) / self.norm_stats["dem"]["std"]

        rainfall_array = (
            rainfall_array
            - self.norm_stats["rainfall"]["mean"]
        ) / self.norm_stats["rainfall"]["std"]

        # =================================================
        # 14. RETURN TENSORS
        # =================================================

        return {
            "sar": torch.tensor(
                sar_grid, dtype=torch.float32
            ),

            "dem": torch.tensor(
                dem_grid, dtype=torch.float32
            ),

            "rainfall": torch.tensor(
                rainfall_array, dtype=torch.float32
            ),

            "mask": torch.tensor(
                mask_grid, dtype=torch.float32
            ),

            "water_depth": torch.tensor(
                depth_grid, dtype=torch.float32
            ),

            "district": district,
            "year": year,
            "event_id": event_id
        }


# =========================================================
# 15. DATALOADERS
# =========================================================

def get_dataloaders(batch_size=4):

    train_dataset = AssamFloodDataset("train")
    val_dataset = AssamFloodDataset("val")
    test_dataset = AssamFloodDataset("test")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, test_loader


# =========================================================
# 16. EXECUTION TEST
# =========================================================

if __name__ == "__main__":

    train_loader, val_loader, test_loader = (
        get_dataloaders(batch_size=4)
    )

    batch = next(iter(train_loader))

    print("\nNORMALIZED BATCH SHAPES")
    print("SAR:", batch["sar"].shape)
    print("DEM:", batch["dem"].shape)
    print("Rainfall:", batch["rainfall"].shape)
    print("Reference:", batch["mask"].shape)
    print("Water depth:", batch["water_depth"].shape)

    print("\nNORMALIZED INPUT STATISTICS")

    for name in ["sar", "dem", "rainfall"]:
        values = batch[name]
        print(
            name,
            "mean:", round(values.mean().item(), 4),
            "std:", round(values.std().item(), 4),
            "finite:", torch.isfinite(values).all().item()
        )

    print("\nTEMPORAL SPLIT")
    print("Training:", len(train_loader.dataset))
    print("Validation:", len(val_loader.dataset))
    print("Testing:", len(test_loader.dataset))

    print("\nNORMALIZED DATASET TEST COMPLETED")
