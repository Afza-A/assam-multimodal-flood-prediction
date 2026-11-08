
import torch
from pre_flood_dataset import get_dataloaders

train_loader, val_loader, test_loader = get_dataloaders(
    batch_size=4
)

batch = next(iter(train_loader))

for name in ["sar", "dem", "rainfall"]:
    values = batch[name]

    print(f"\n{name.upper()}")
    print("Shape:", values.shape)
    print("Mean:", round(values.mean().item(), 4))
    print("Std:", round(values.std().item(), 4))
    print("All finite:", torch.isfinite(values).all().item())

print("\nNormalization test completed.")
