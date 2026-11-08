
import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 1. MULTIMODAL FEATURE FUSION
# =========================================================

class FeatureFusion(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(
        self,
        sar_features,
        dem_features,
        rainfall_features
    ):

        if not (
            sar_features.shape == dem_features.shape
            == rainfall_features.shape
        ):
            raise ValueError(
                "Encoder feature dimensions do not match."
            )

        if sar_features.shape[1] != 32:
            raise ValueError(
                "Each encoder must output 32 channels."
            )

        fused_features = torch.cat(
            [
                sar_features,
                dem_features,
                rainfall_features
            ],
            dim=1
        )

        return fused_features


# =========================================================
# 2. CONVOLUTION BLOCK
# =========================================================

class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels):

        super().__init__()

        self.block = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


# =========================================================
# 3. U-NET STYLE FLOOD DECODER
# =========================================================

class FloodDecoder(nn.Module):

    def __init__(self):

        super().__init__()

        # -------------------------------------------------
        # ENCODER PATH
        # -------------------------------------------------

        # Input: [B, 96, 10, 10]
        # Output: [B, 64, 10, 10]

        self.encoder_block = ConvBlock(
            in_channels=96,
            out_channels=64
        )

        # Downsampling:
        # [B, 64, 10, 10] -> [B, 128, 5, 5]

        self.downsample = nn.Sequential(

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # -------------------------------------------------
        # BOTTLENECK
        # -------------------------------------------------

        self.bottleneck = ConvBlock(
            in_channels=128,
            out_channels=128
        )

        # -------------------------------------------------
        # DECODER PATH
        # -------------------------------------------------

        # Upsample:
        # [B, 128, 5, 5] -> [B, 64, 10, 10]

        self.upsample = nn.ConvTranspose2d(
            in_channels=128,
            out_channels=64,
            kernel_size=2,
            stride=2
        )

        # Skip connection:
        # 64 decoder channels + 64 encoder channels = 128

        self.decoder_block = ConvBlock(
            in_channels=128,
            out_channels=32
        )

        # -------------------------------------------------
        # OUTPUT HEAD
        # -------------------------------------------------

        self.output_layer = nn.Conv2d(
            in_channels=32,
            out_channels=1,
            kernel_size=1
        )

    def forward(self, fused_features):

        # Encoder feature map / skip connection.
        skip_features = self.encoder_block(
            fused_features
        )

        # Downsample.
        x = self.downsample(skip_features)

        # Bottleneck.
        x = self.bottleneck(x)

        # Upsample.
        x = self.upsample(x)

        # Ensure exact spatial alignment.
        if x.shape[-2:] != skip_features.shape[-2:]:
            x = F.interpolate(
                x,
                size=skip_features.shape[-2:],
                mode="bilinear",
                align_corners=False
            )

        # Concatenate encoder and decoder features.
        x = torch.cat(
            [x, skip_features],
            dim=1
        )

        # Decoder refinement.
        x = self.decoder_block(x)

        # Raw prediction logits.
        logits = self.output_layer(x)

        return logits


# =========================================================
# 4. COMPLETE MULTIMODAL FLOOD MODEL
# =========================================================

class MultimodalFloodModel(nn.Module):

    def __init__(
        self,
        sar_encoder,
        dem_encoder,
        rainfall_encoder
    ):

        super().__init__()

        self.sar_encoder = sar_encoder
        self.dem_encoder = dem_encoder
        self.rainfall_encoder = rainfall_encoder

        self.fusion = FeatureFusion()
        self.decoder = FloodDecoder()

    def forward(self, sar, dem, rainfall):

        # Three modality encoders.
        sar_features = self.sar_encoder(sar)

        dem_features = self.dem_encoder(dem)

        rainfall_features = self.rainfall_encoder(
            rainfall
        )

        # Multimodal fusion.
        fused_features = self.fusion(
            sar_features,
            dem_features,
            rainfall_features
        )

        # U-Net-style decoder.
        logits = self.decoder(fused_features)

        # Probability map.
        probabilities = torch.sigmoid(logits)

        return {
            "sar_features": sar_features,
            "dem_features": dem_features,
            "rainfall_features": rainfall_features,
            "fused_features": fused_features,
            "logits": logits,
            "probabilities": probabilities
        }

    @staticmethod
    def predict_binary(
        probabilities,
        threshold=0.5
    ):

        return (
            probabilities >= threshold
        ).float()
