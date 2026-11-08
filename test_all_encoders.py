
import torch

from pre_flood_dataset import AssamFloodDataset, DataLoader
from sar_encoder import SAREncoder
from dem_encoder import DEMEncoder
from rainfall_encoder import RainfallEncoder


# ============================================================
# 1. LOAD PRE-FLOOD DATA
# ============================================================

print("\n======================================")
print("       ENCODER INTEGRATION TEST")
print("======================================")

dataset = AssamFloodDataset()

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False
)

batch = next(iter(dataloader))

sar_input = batch["sar"]
dem_input = batch["dem"]
rainfall_input = batch["rainfall"]


# ============================================================
# 2. INITIALIZE ALL THREE ENCODERS
# ============================================================

sar_model = SAREncoder(output_channels=32)
dem_model = DEMEncoder(output_channels=32)

rainfall_model = RainfallEncoder(
    input_size=1,
    hidden_size=32,
    num_layers=2
)


# ============================================================
# 3. FORWARD PASS
# ============================================================

sar_features = sar_model(sar_input)
dem_features = dem_model(dem_input)
rainfall_features = rainfall_model(rainfall_input)


# ============================================================
# 4. DISPLAY FEATURE SHAPES
# ============================================================

print("\nInput shapes:")

print("SAR:", sar_input.shape)
print("DEM:", dem_input.shape)
print("Rainfall:", rainfall_input.shape)

print("\nExtracted feature shapes:")

print("SAR features:", sar_features.shape)
print("DEM features:", dem_features.shape)
print("Rainfall features:", rainfall_features.shape)


# ============================================================
# 5. FEATURE SHAPE COMPATIBILITY
# ============================================================

assert sar_features.shape == dem_features.shape
assert dem_features.shape == rainfall_features.shape

print("\nAll three feature shapes are compatible!")


# ============================================================
# 6. FEATURE FUSION SHAPE TEST
# ============================================================

fused_features = torch.cat(
    [sar_features, dem_features, rainfall_features],
    dim=1
)

print("\nFused feature shape:")
print(fused_features.shape)


# ============================================================
# 7. COMBINED BACKPROPAGATION TEST
# ============================================================

loss = fused_features.mean()

loss.backward()

print("\nBackpropagation test:")

print(
    "SAR gradients:",
    all(p.grad is not None for p in sar_model.parameters())
)

print(
    "DEM gradients:",
    all(p.grad is not None for p in dem_model.parameters())
)

print(
    "Rainfall gradients:",
    all(p.grad is not None for p in rainfall_model.parameters())
)


# ============================================================
# 8. FINAL TEST RESULT
# ============================================================

print("\n======================================")
print("ALL THREE ENCODERS INTEGRATION TEST")
print("COMPLETED SUCCESSFULLY")
print("======================================")
