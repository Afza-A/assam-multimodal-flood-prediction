
import torch
import torch.nn as nn


# ============================================================
# DEM CNN ENCODER
# ============================================================

class DEMEncoder(nn.Module):

    def __init__(self, output_channels=32):
        super(DEMEncoder, self).__init__()

        self.encoder = nn.Sequential(

            # First convolution layer
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Second convolution layer
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Final feature extraction layer
            nn.Conv2d(
                in_channels=32,
                out_channels=output_channels,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU()
        )

    def forward(self, x):

        features = self.encoder(x)

        return features


# ============================================================
# TEST DEM ENCODER
# ============================================================

if __name__ == "__main__":

    print("======================================")
    print("        DEM CNN ENCODER TEST")
    print("======================================")

    # Create encoder
    model = DEMEncoder(output_channels=32)

    # Synthetic input matching our dataset shape
    dem_input = torch.randn(2, 1, 10, 10)

    # Enable gradient tracking
    dem_input.requires_grad = True

    # Forward propagation
    output = model(dem_input)

    print("\nInput shape:")
    print(dem_input.shape)

    print("\nOutput feature shape:")
    print(output.shape)

    # Backpropagation test
    loss = output.mean()
    loss.backward()

    print("\nBackpropagation:")
    print("Input gradient available:",
          dem_input.grad is not None)

    print("Model gradients available:",
          all(
              parameter.grad is not None
              for parameter in model.parameters()
              if parameter.requires_grad
          ))

    print("\n======================================")
    print("DEM CNN ENCODER TEST COMPLETED")
    print("======================================")
