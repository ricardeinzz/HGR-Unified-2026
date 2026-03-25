# HGR-Unified-2026

## Gesture Recognition System (Hand Gesture Recognition - HGR)
Unified project for HGR using Convolutional Neural Networks (CNN) and CNN-LSTM architectures.

### Description
This project implements a gesture recognition system based on Electromyography (EMG) signals. It includes:
- **Shared Architecture**: A unified `Shared.m` class for preprocessing and signal management.
- **Dual Dataset Support**: Compatibility with the original JSON dataset and the new high-performance MAT dataset.
- **Enhanced Data Processing**: Corrected index handling and empty sample skipping for robust training.

### Project Structure
- `Shared.m`: Central configuration and utility functions.
- `SpectrogramDatastore.m`: Custom datastore for CNN training.
- `SpectrogramDatastoreLSTM.m`: Custom datastore for sequence-based CNN-LSTM training.
- `CNN/`: Implementation and evaluation scripts for the CNN model.
- `CNN-LSTM/`: Implementation, generation, and evaluation scripts for the CNN-LSTM model.

### Requirements
- MATLAB R2021b or higher (Tested up to R2025b).
- Deep Learning Toolbox.
- Signal Processing Toolbox.

### Usage
1. Generate the datastores using `CNN-LSTM/spectrogramDatasetGeneration.m`.
2. Train the models using the respective `modelCreation.m` scripts.
3. Evaluate performance with `modelEvaluation.m`.

---
*Laboratorio de Inteligencia y Visión Artificial - EPN*
