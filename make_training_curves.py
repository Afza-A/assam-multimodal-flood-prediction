import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

df = pd.read_csv("training_history.csv")

output = Path("presentation_figures")
output.mkdir(exist_ok=True)

# Loss curves
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_loss"], marker="o", label="Training Loss")
plt.plot(df["epoch"], df["val_loss"], marker="o", label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output / "training_validation_loss.png", dpi=300)
plt.close()

# Dice curves
plt.figure(figsize=(8, 5))
plt.plot(df["epoch"], df["train_dice"], marker="o", label="Training Dice")
plt.plot(df["epoch"], df["val_dice"], marker="o", label="Validation Dice")
plt.xlabel("Epoch")
plt.ylabel("Dice Score")
plt.title("Training and Validation Dice")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(output / "training_validation_dice.png", dpi=300)
plt.close()

print("Training curves saved successfully!")