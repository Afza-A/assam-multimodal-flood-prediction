
# PERSON 2 – MODALITY-SPECIFIC FEATURE EXTRACTION

## 1. Overview

Person 2 is responsible for implementing modality-specific feature extraction for the multimodal flood prediction framework.

Three separate neural network encoders are implemented:

1. Sentinel-1 SAR CNN Encoder
2. Digital Elevation Model (DEM) CNN Encoder
3. Rainfall GRU Encoder

Each encoder extracts features from its respective input modality. The extracted features are designed to be compatible with the feature fusion component.

## 2. Dataset Inputs

The implementation uses a synthetic Assam flood dataset covering the Kopili River study region.

The spatial grid size is 10 × 10 pixels.

### 2.1 Sentinel-1 SAR

- Input variable: Pre-flood VV polarization.
- Input channels: 1.
- Spatial dimensions: 10 × 10.
- Data source: Synthetic SAR dataset.

### 2.2 Digital Elevation Model

- Input variable: Elevation.
- Input channels: 1.
- Spatial dimensions: 10 × 10.
- Data source: Synthetic DEM dataset.
- DEM is treated as static geographical information.

### 2.3 Rainfall

- Input variable: Daily rainfall observations.
- Sequence length: 10 days.
- Observation period: June 1–10 (pre-flood period).
- Spatial grid: 10 × 10 pixels.
- Data source: Synthetic rainfall dataset.

## 3. Sentinel-1 SAR CNN Encoder

File: `sar_encoder.py`

The SAR encoder extracts spatial features from pre-flood Sentinel-1 VV imagery.

### Architecture

1. Convolution layer: 1 input channel, 16 output channels.
2. Batch Normalization.
3. ReLU activation.
4. Convolution layer: 16 input channels, 32 output channels.
5. Batch Normalization.
6. ReLU activation.
7. Final convolution layer: 32 input channels, 32 output channels.
8. ReLU activation.

All convolution layers use a 3 × 3 kernel with padding 1.

### Input and Output

Input: [B, 1, 10, 10]

Output: [B, 32, 10, 10]

## 4. DEM CNN Encoder

File: `dem_encoder.py`

The DEM encoder extracts spatial terrain features from the elevation grid.

### Architecture

1. Convolution layer: 1 input channel, 16 output channels.
2. Batch Normalization.
3. ReLU activation.
4. Convolution layer: 16 input channels, 32 output channels.
5. Batch Normalization.
6. ReLU activation.
7. Final convolution layer: 32 input channels, 32 output channels.
8. ReLU activation.

All convolution layers use a 3 × 3 kernel with padding 1.

### Input and Output

Input: [B, 1, 10, 10]

Output: [B, 32, 10, 10]

## 5. Rainfall GRU Encoder

File: `rainfall_encoder.py`

The rainfall encoder extracts temporal features from daily rainfall sequences.

A two-layer Gated Recurrent Unit (GRU) is implemented.

### Architecture

- Input size: 1.
- Hidden size: 32.
- Number of GRU layers: 2.
- Sequence length: 10 days.
- Batch-first configuration: True.

Each pixel is treated as an independent rainfall sequence. The final time-step feature is extracted and rearranged into the original 10 × 10 spatial grid.

### Input and Output

Input: [B, 100, 10, 1]

Output: [B, 32, 10, 10]

## 6. Encoder Integration

File: `test_all_encoders.py`

All three encoders were integrated and tested using the pre-flood DataLoader.

The output feature maps were concatenated along the channel dimension.

### Feature Concatenation

SAR features: [B, 32, 10, 10]

DEM features: [B, 32, 10, 10]

Rainfall features: [B, 32, 10, 10]

Combined features: [B, 96, 10, 10]

The resulting 96-channel feature tensor can be passed to the feature fusion component, subject to the integration interface agreed with Person 3.

## 7. Testing and Validation

The following tests were performed:

1. Individual forward-pass tests for all three encoders.
2. Input and output tensor shape verification.
3. Input gradient verification.
4. Model parameter gradient verification.
5. Combined encoder integration test.
6. Feature concatenation compatibility test.
7. Combined backpropagation test.

### Test Results

| Component | Test Result |
|---|---|
| SAR CNN encoder | Passed |
| DEM CNN encoder | Passed |
| Rainfall GRU encoder | Passed |
| Feature shape compatibility | Passed |
| Feature concatenation | Passed |
| Combined backpropagation | Passed |

## 8. Handover to Person 3

Person 2 provides the following feature outputs:

- SAR encoder: 32 feature channels.
- DEM encoder: 32 feature channels.
- Rainfall encoder: 32 feature channels.

The concatenated feature tensor has 96 channels and a spatial resolution of 10 × 10.

Person 3 can use this tensor as the input to the subsequent feature fusion and decoder stages, subject to agreement on the integration interface.

## 9. Important Methodological Considerations

- Only pre-flood SAR VV is used as a forecasting input.
- Only pre-flood daily rainfall observations from June 1–10 are used.
- DEM is treated as static geographical input.
- Peak-flood SAR is not used as an encoder input.
- Post-flood observations are not used as forecasting inputs.
- Flood reference labels are kept separate from the encoder inputs.

## 10. Limitations

The current dataset is synthetically generated.

The tests establish that the encoders execute, produce the expected tensor dimensions, support gradient propagation, and can be integrated through feature concatenation.

These tests do not establish real-world flood prediction accuracy or generalisation performance.

Evaluation using authentic, spatially and temporally aligned datasets is required for those claims.
