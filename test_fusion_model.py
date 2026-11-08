
import torch

from sar_encoder import SAREncoder
from dem_encoder import DEMEncoder
from rainfall_encoder import RainfallEncoder

from fusion_decoder import MultimodalFloodModel
from pre_flood_dataset import get_dataloaders


# ---------------------------------------------------------
# 1. GPU CONFIGURATION
# ---------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

if DEVICE.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=2
)

batch = next(iter(train_loader))

sar = batch["sar"].to(DEVICE)
dem = batch["dem"].to(DEVICE)
rainfall = batch["rainfall"].to(DEVICE)
mask = batch["mask"].to(DEVICE)


# ---------------------------------------------------------
# 3. INITIALIZE ENCODERS
# ---------------------------------------------------------

sar_encoder = SAREncoder()
dem_encoder = DEMEncoder()
rainfall_encoder = RainfallEncoder()


# ---------------------------------------------------------
# 4. BUILD COMPLETE MODEL
# ---------------------------------------------------------

model = MultimodalFloodModel(
    sar_encoder,
    dem_encoder,
    rainfall_encoder
).to(DEVICE)

print("\nMODEL INITIALIZED")
print("Device:", next(model.parameters()).device)


# ---------------------------------------------------------
# 5. FORWARD PASS
# ---------------------------------------------------------

model.train()

outputs = model(
    sar,
    dem,
    rainfall
)

print("\nFEATURE SHAPES")

print(
    "SAR features:",
    outputs["sar_features"].shape
)

print(
    "DEM features:",
    outputs["dem_features"].shape
)

print(
    "Rainfall features:",
    outputs["rainfall_features"].shape
)

print(
    "Fused features:",
    outputs["fused_features"].shape
)

print(
    "Logits:",
    outputs["logits"].shape
)

print(
    "Probability map:",
    outputs["probabilities"].shape
)


# ---------------------------------------------------------
# 6. BINARY FLOOD MAP
# ---------------------------------------------------------

binary_mask = model.predict_binary(
    outputs["probabilities"],
    threshold=0.5
)

print(
    "Binary flood map:",
    binary_mask.shape
)


# ---------------------------------------------------------
# 7. LOSS AND BACKPROPAGATION TEST
# ---------------------------------------------------------

criterion = torch.nn.BCEWithLogitsLoss()

loss = criterion(
    outputs["logits"],
    mask
)

loss.backward()

print("\nBACKPROPAGATION TEST")
print("Loss:", loss.item())

for name, parameter in model.named_parameters():

    if parameter.grad is not None:
        if parameter.grad.abs().sum().item() > 0:
            print(name, "gradient: OK")

print("\nCOMPLETE MULTIMODAL MODEL TEST FINISHED")
