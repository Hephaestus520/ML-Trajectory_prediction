<div align="center">

# CS:GO Player Action Predictor

Deep learning system that predicts player actions in Counter-Strike: Global Offensive using LSTM neural networks. Trained on 680 professional match demos from the ESTA dataset, the model classifies player movements into five distinct action categories with 87.9% accuracy.

<br/>

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white) ![PyTorch](https://img.shields.io/badge/PyTorch-2.9-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white) ![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white) ![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white) ![scikit--learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)

</div>

---

## Overview

This machine learning project implements a bidirectional LSTM classifier that analyzes player trajectories in CS:GO professional matches and predicts their next action. The model processes sequences of 10 timesteps, each containing 13 features related to position, movement, health, and view direction to classify actions into five categories: move, jump, duck, idle, or dead.

The system achieves 87.9% overall accuracy with particularly strong performance on the "dead" (97.1% F1) and "move" (87.4% F1) classes. The project includes a complete pipeline for data extraction, preprocessing, training, evaluation, and interactive prediction.

## Quick Start

### Prerequisites

- Python 3.11+
- 8GB+ RAM (for full dataset)
- 2GB+ disk space

### Installation

```bash
# Clone repository
git clone https://github.com/Hephaestus520/ML-Trajectory_prediction.git
cd ML-Trajectory_prediction

# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1  # Windows
# source venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Usage

**Interactive Menu**:
```bash
python main.py
```

Options:
1. Process raw data
2. Merge batch files
3. Generate features and labels
4. Train LSTM model
5. Evaluate model
6. Make predictions

**Quick Training** (uses 20% of data, ~45 minutes):
```bash
python scripts/train_quick.py
```

**Data Validation**:
```bash
python src/data/validation.py
```

## Project Structure

```
ML-Trajectory_prediction/
├── config/
│   └── .env.example           # Configuration template
├── data/
│   ├── raw/lan/               # Original ESTA dataset (680 .json.xz files)
│   └── processed/             # Preprocessed parquet files
├── docs/
│   ├── ARCHITECTURE.md        # Technical architecture details
│   ├── PRESENTATION.md        # Project presentation slides
│   └── RESULTS.md             # Evaluation results and metrics
├── outputs/
│   └── run_*/                 # Training runs with models and logs
├── scripts/
│   ├── train_quick.py         # Fast training script
│   ├── debug_data.py          # Data inspection utility
│   └── test_simple.py         # Simple test script
├── src/
│   ├── data/                  # Data processing pipeline
│   │   ├── data_prep.py       # Extract player data from demos
│   │   ├── merge_batches.py   # Combine batch files
│   │   ├── data_der.py        # Feature engineering
│   │   └── validation.py      # Data quality checks
│   └── models/                # ML model components
│       ├── model.py           # LSTM architecture
│       ├── dataset.py         # PyTorch Dataset class
│       ├── train.py           # Training pipeline
│       ├── evaluate.py        # Model evaluation
│       └── predict.py         # Inference utilities
├── main.py                    # Interactive CLI menu
├── requirements.txt           # Python dependencies
└── LICENSE                    # MIT License
```

## Model Architecture

### LSTM Specifications
- **Type**: Bidirectional LSTM with fully connected layers
- **Layers**: 2 LSTM layers (128 units each) + 2 FC layers
- **Regularization**: Dropout (0.3), BatchNorm1d
- **Total Parameters**: 214,661
- **Input**: Sequences of 10 timesteps × 13 features
- **Output**: 5 action classes

### Features (13 total)

| Category | Features | Description |
|----------|----------|-------------|
| Position | x, y, z | 3D coordinates in game space |
| Movement | velocity, dx, dy, dz, speed, acceleration | Motion-related metrics |
| Player State | hp, armor | Health and armor points |
| View Direction | viewX, viewY | Camera orientation angles |

### Action Classes (5 total)

- **move**: Player walking or running
- **jump**: Player jumping
- **duck**: Player crouching
- **idle**: Player standing still
- **dead**: Player eliminated

## Results

Evaluated on 32,989 test sequences from professional matches:

| Metric | Score |
|--------|-------|
| Overall Accuracy | 87.90% |
| Weighted Precision | 90.04% |
| Weighted Recall | 87.90% |
| Weighted F1-Score | 88.36% |

### Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| dead  | 97.06% | 97.77% | 97.41% | 9,222 |
| duck  | 22.50% | 94.74% | 36.36% | 19 |
| idle  | 55.78% | 94.73% | 70.21% | 2,145 |
| jump  | 79.46% | 87.70% | 83.37% | 5,535 |
| move  | 94.30% | 81.39% | 87.37% | 16,068 |

See [docs/RESULTS.md](docs/RESULTS.md) for detailed evaluation metrics and visualizations.

## Configuration

Edit `config/.env.example` and save as `config/.env`:

```env
# Model hyperparameters
SEQUENCE_LENGTH=10
HIDDEN_SIZE=128
NUM_LAYERS=2
DROPOUT=0.3

# Training configuration
BATCH_SIZE=256
NUM_EPOCHS=20
LEARNING_RATE=0.001

# Memory optimization
SAMPLE_FRACTION=0.2        # Use 20% of dataset
MAX_SEQUENCES=1000000      # Cap at 1M sequences
```

## Dataset

**ESTA (Esports Trajectories and Actions)**
- Source: Professional CS:GO match demos
- Format: 680 compressed JSON files (`.json.xz`)
- Size: ~12.8 GB processed
- Total observations: 34+ million player states

## Training

### Full Training
```bash
python src/models/train.py
```

### Quick Training (Optimized)
```bash
python scripts/train_quick.py
```
- Uses 20% of dataset (~6.8M observations)
- Limits to 1M sequences
- Completes in ~45 minutes
- Achieves 85-90% validation accuracy

Training outputs saved to `outputs/run_<timestamp>/`:
- `best_model.pt` - Model checkpoint
- `config.json` - Hyperparameters
- `training_history.png` - Loss/accuracy plots
- `scaler.pkl`, `label_encoder.pkl` - Preprocessing objects

## Evaluation

```bash
python main.py  # Select option 5
# or
python src/models/evaluate.py
```

Generates:
- Confusion matrix visualization
- Per-class probability distributions
- Detailed metrics report (JSON)

## Technical Details

- **Framework**: PyTorch 2.9
- **Optimizer**: Adam
- **Loss**: CrossEntropyLoss with class weights
- **Scheduler**: ReduceLROnPlateau
- **Data Split**: 70% train / 15% val / 15% test
- **Normalization**: StandardScaler
- **Sequence Strategy**: Sliding window with 10-step lookback

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for complete technical documentation.

## Troubleshooting

**Out of Memory Error**:
- Reduce `SAMPLE_FRACTION` in config (try 0.1)
- Lower `MAX_SEQUENCES` to 500,000
- Decrease `BATCH_SIZE` to 128

**Slow Training**:
- Increase `BATCH_SIZE` to 512
- Reduce `NUM_EPOCHS`
- Close other applications

**Low Accuracy**:
- Increase `SAMPLE_FRACTION` to use more data
- Train for more epochs
- Check class distribution in output logs

## Contributing

Contributions are welcome! Areas for improvement:
- Attention mechanisms for sequence modeling
- Real-time prediction API
- Additional game-specific features
- Model compression for deployment

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **awpy Library**: CS:GO demo parsing functionality
- **PyTorch Team**: Deep learning framework
- **ESTA Dataset**: Esports trajectory data providers

## Authors

**Sebastian Salazar** & **Miguel Ángel Escudero**
- EAFIT University - Machine Learning Course Project
- November 2025

---

<div align="center">

**[View Documentation](docs/ARCHITECTURE.md) • [See Results](docs/RESULTS.md) • [Report Issue](https://github.com/Hephaestus520/ML-Trajectory_prediction/issues)**

</div>
