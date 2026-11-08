
import torch
import torch.nn as nn


# ============================================================
# RAINFALL GRU ENCODER
# ============================================================

class RainfallEncoder(nn.Module):

    def __init__(
        self,
        input_size=1,
        hidden_size=32,
        num_layers=2
    ):
        super(RainfallEncoder, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):

        # x shape: [B, 100, 10, 1]

        batch_size = x.size(0)
        num_pixels = x.size(1)
        sequence_length = x.size(2)
        input_features = x.size(3)

        # Process every pixel as an independent sequence
        x = x.reshape(
            batch_size * num_pixels,
            sequence_length,
            input_features
        )

        # GRU feature extraction
        output, hidden = self.gru(x)

        # Take the final time-step feature
        features = output[:, -1, :]

        # Restore spatial grid structure
        features = features.reshape(
            batch_size,
            num_pixels,
            -1
        )

        # Convert 100 pixels into a 10x10 grid
        features = features.permute(0, 2, 1)

        features = features.reshape(
            batch_size,
            -1,
            10,
            10
        )

        return features


# ============================================================
# TEST RAINFALL GRU ENCODER
# ============================================================

if __name__ == "__main__":

    print("======================================")
    print("       RAINFALL GRU ENCODER TEST")
    print("======================================")

    # Create encoder
    model = RainfallEncoder(
        input_size=1,
        hidden_size=32,
        num_layers=2
    )

    # Synthetic rainfall input
    rainfall_input = torch.randn(2, 100, 10, 1)

    rainfall_input.requires_grad = True

    # Forward propagation
    output = model(rainfall_input)

    print("\nInput shape:")
    print(rainfall_input.shape)

    print("\nOutput feature shape:")
    print(output.shape)

    # Backpropagation test
    loss = output.mean()
    loss.backward()

    print("\nBackpropagation:")

    print(
        "Input gradient available:",
        rainfall_input.grad is not None
    )

    print(
        "Model gradients available:",
        all(
            parameter.grad is not None
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )

    print("\n======================================")
    print("RAINFALL GRU ENCODER TEST COMPLETED")
    print("======================================")
