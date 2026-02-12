# Architecture Documentation

## System Overview

This project implements an LSTM-based classifier for predicting player actions in CS:GO professional matches. The system processes sequential player data and predicts the next action from 5 possible classes.

## Data Flow

```
Raw ESTA Data (.json.xz files)
        ↓
    Data Preprocessing (awpy parser)
        ↓
    Feature Engineering (13 features)
        ↓
    Sequence Creation (10 timesteps)
        ↓
    LSTM Model (2 layers, 128 units)
        ↓
    Action Prediction (5 classes)
```

## Model Architecture

### LSTM Classifier Specifications

- **Architecture**: Bidirectional LSTM
- **Layers**: 2 LSTM layers + 2 fully connected layers
- **Hidden Size**: 128 units per layer
- **Dropout**: 0.3 (regularization)
- **Batch Normalization**: Applied after LSTM layers
- **Total Parameters**: 214,661

### Input Shape
- **Sequence Length**: 10 timesteps
- **Features per Timestep**: 13
- **Input Tensor Shape**: `(batch_size, 10, 13)`

### Output Shape
- **Classes**: 5 (dead, duck, idle, jump, move)
- **Output Tensor Shape**: `(batch_size, 5)`

## Feature Engineering

### Input Features (13 total)

**Position (3)**: 
- `x`, `y`, `z` - Player coordinates in 3D space

**Movement (5)**:
- `velocity` - Total velocity magnitude
- `dx`, `dy`, `dz` - Positional changes per tick
- `speed` - Horizontal speed
- `acceleration` - Rate of velocity change

**Player State (2)**:
- `hp` - Health points (0-100)
- `armor` - Armor points (0-100)

**View Direction (2)**:
- `viewX` - Horizontal viewing angle
- `viewY` - Vertical viewing angle

### Target Variable

**action_label**: Categorical variable with 5 classes
- `dead` - Player eliminated
- `duck` - Crouching
- `idle` - Standing still
- `jump` - Jumping
- `move` - Walking/running

## Data Processing Pipeline

### 1. Raw Data Extraction (`src/data/data_prep.py`)
- Parses `.json.xz` files using awpy library
- Extracts player trajectories per round
- Outputs batch parquet files

### 2. Batch Merging (`src/data/merge_batches.py`)
- Combines all batch files
- Creates single consolidated dataset
- Optimizes memory usage

### 3. Feature Derivation (`src/data/data_der.py`)
- Calculates derived features (dx, dy, dz, speed, acceleration)
- Assigns action labels based on player state
- Generates final labeled dataset

### 4. Validation (`src/data/validation.py`)
- Checks data quality
- Validates feature distributions
- Detects missing values and outliers

## Model Components

### Dataset Class (`src/models/dataset.py`)
- **Purpose**: PyTorch Dataset for sequence generation
- **Key Methods**:
  - `_create_sequences()`: Generates sliding windows of 10 timesteps
  - `get_class_weights()`: Calculates weights for imbalanced classes
  - `save_preprocessors()`: Saves StandardScaler and LabelEncoder
- **Preprocessing**:
  - StandardScaler normalization
  - LabelEncoder for categorical labels
  - Grouped by (map, round, player) for sequence continuity

### LSTM Model (`src/models/model.py`)
- **Class**: `ActionClassifierLSTM`
- **Key Methods**:
  - `forward()`: Forward pass through LSTM and FC layers
  - `predict()`: Returns class probabilities
  - `predict_class()`: Returns predicted class label
- **Architecture Details**:
  ```
  Input (batch, 10, 13)
      ↓
  LSTM Layer 1 (bidirectional, 128 units)
      ↓
  LSTM Layer 2 (bidirectional, 128 units)
      ↓
  BatchNorm1d
      ↓
  Dropout (0.3)
      ↓
  Fully Connected (256 → 128)
      ↓
  ReLU + Dropout
      ↓
  Fully Connected (128 → 5)
  ```

### Training Pipeline (`src/models/train.py`)
- **Optimizer**: Adam
- **Loss Function**: CrossEntropyLoss with class weights
- **Learning Rate Scheduler**: ReduceLROnPlateau
- **Early Stopping**: Saves best model based on validation loss
- **Data Split**: 70% train, 15% val, 15% test

### Evaluation (`src/models/evaluate.py`)
- Calculates accuracy, precision, recall, F1-score
- Generates confusion matrix visualization
- Plots probability distributions per class
- Saves detailed metrics in JSON format

## Performance Optimization

### Memory Management
- **Sample Fraction**: Can limit dataset to percentage (default: 20%)
- **Max Sequences**: Caps total sequences to avoid OOM errors
- **Batch Processing**: Uses DataLoader with configurable batch size

### Training Speed
- **Batch Size**: 256 (optimized for CPU/GPU)
- **Num Workers**: 0 (Windows compatibility)
- **Pin Memory**: Enabled for GPU training

## Results

Based on evaluation with 32,989 sequences:

| Metric | Score |
|--------|-------|
| Accuracy | 87.90% |
| Precision | 90.04% |
| Recall | 87.90% |
| F1-Score | 88.36% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| dead  | 0.97 | 0.98 | 0.97 | 9,222 |
| duck  | 0.23 | 0.95 | 0.36 | 19 |
| idle  | 0.56 | 0.95 | 0.70 | 2,145 |
| jump  | 0.79 | 0.88 | 0.83 | 5,535 |
| move  | 0.94 | 0.81 | 0.87 | 16,068 |

## Configuration Files

### Environment Variables (`config/.env.example`)
Contains default hyperparameters and paths

### Requirements (`requirements.txt`)
- numpy, pandas - Data manipulation
- scikit-learn - Preprocessing
- torch - Deep learning framework
- matplotlib - Visualization
- awpy - CS:GO demo parsing
- tqdm - Progress bars

## Future Improvements

1. **Model Enhancements**
   - Experiment with attention mechanisms
   - Try Transformer-based architectures
   - Ensemble multiple models

2. **Feature Engineering**
   - Add team coordination features
   - Include weapon information
   - Incorporate map-specific features

3. **Data Augmentation**
   - Synthetic sequence generation
   - Time warping
   - Noise injection

4. **Deployment**
   - Create REST API for predictions
   - Real-time inference pipeline
   - Docker containerization
