
import torch
import torch.nn as nn


# ============================================================
# SENTINEL-1 SAR CNN ENCODER
# ============================================================

class SAREncoder(nn.Module):

    def __init__(self, output_channels=32):
        super(SAREncoder, self).__init__()

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
# TEST SAR ENCODER
# ============================================================

if __name__ == "__main__":

    print("======================================")
    print("       SAR CNN ENCODER TEST")
    print("======================================")

    # Create encoder
    model = SAREncoder(output_channels=32)

    # Synthetic input matching our dataset shape
    sar_input = torch.randn(2, 1, 10, 10)

    # Enable gradient tracking
    sar_input.requires_grad = True

    # Forward propagation
    output = model(sar_input)

    print("\nInput shape:")
    print(sar_input.shape)

    print("\nOutput feature shape:")
    print(output.shape)

    # Backpropagation test
    loss = output.mean()
    loss.backward()

    print("\nBackpropagation:")
    print("Input gradient available:",
          sar_input.grad is not None)

    print("Model gradients available:",
          all(
              parameter.grad is not None
              for parameter in model.parameters()
              if parameter.requires_grad
          ))

    print("\n======================================")
    print("SAR CNN ENCODER TEST COMPLETED")
    print("======================================")
