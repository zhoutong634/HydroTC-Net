# HydroTC-Net

HydroTC-Net is a deep-learning surrogate model for urban flood prediction. It uses rainfall history, a digital elevation model (DEM), and an explicit target time to predict two-dimensional urban flood-depth fields.

## Project Description

High-resolution hydrodynamic simulation of urban flooding is computationally expensive, which limits its use in real-time warning, multi-scenario analysis, and rapid planning assessment. HydroTC-Net is a deep-learning surrogate framework for spatiotemporal urban pluvial-flood prediction. It approximates high-resolution hydrodynamic flood-depth fields at substantially lower inference cost.

The model takes causally truncated rainfall history, a digital elevation model, and an explicit target-time encoding as input. It extends a U-Net encoder-decoder with cross-scale channel feature fusion and channel-wise cross attention, allowing the network to select spatial features from different scales according to the current decoder state. HydroTC-Net uses a non-autoregressive formulation: target times within a rainfall event can be queried independently without recursively conditioning on previously predicted depth fields, which avoids the accumulation of autoregressive prediction errors.

The project provides composite-loss training, data preprocessing, validation, and multi-event evaluation workflows. The model outputs two-dimensional water-depth grids and can generate prediction maps, target maps, error maps, and evaluation metrics including MAE, RMSE, R², Bias, NSE, F1, and IoU. The code is intended for urban flood surrogate modelling, flood-risk assessment, and deep-learning-based hydrodynamic emulation.

This project is implemented with PyTorch and supports both model training and multi-event testing.

## Main Features

- Non-autoregressive target-time querying: each target time is predicted independently.
- 38 input channels: 36 rainfall-history channels, one DEM channel, and one target-time channel.
- U-Net encoder-decoder backbone enhanced with Channel Transformer and channel-wise cross attention.
- Composite loss that constrains continuous depth, inundation extent, spatial gradients, and underprediction.
- Multi-event evaluation with spatial prediction maps, NumPy arrays, and metric CSV files.

## Project Structure

```text
HydroTC-Net/
├── nets/
│   ├── HydroTC_Net.py       # HydroTC-Net model
│   ├── CTrans.py            # Channel Transformer module
│   ├── UNet.py              # U-Net baseline
│   └── WideUNet.py          # Wide U-Net baseline
├── Config.py                # Training and data-path configuration
├── flood_dataset.py         # Data loading, normalization, and input construction
├── train_flood.py           # Training entry point
├── Train_one_epoch.py       # Training and validation for one epoch
├── test_flood.py            # Multi-event evaluation entry point
└── utils.py                 # Composite loss and evaluation metrics
```

## Installation

Python 3.9 or later is recommended. Install a PyTorch build compatible with your CUDA version.

```bash
conda create -n hydrotc python=3.10 -y
conda activate hydrotc
pip install torch torchvision torchaudio
pip install numpy pandas xarray netCDF4 scipy scikit-learn tqdm tensorboardX ml-collections matplotlib
```

## Data Layout

The default data root is `data/` inside the project directory:

```text
data/
├── static/dem.input
├── train/<event_name>/{rain.txt,output.nc}
├── val/<event_name>/{rain.txt,output.nc}
└── test/<event_name>/{rain.txt,output.nc}
```

`dem.input` is an ASCII-grid DEM; the first six header lines are skipped. `rain.txt` contains the rainfall sequence. `output.nc` must contain a water-depth variable named `h`, with time as its first dimension. Invalid terrain cells use `-9999`, and building-mask cells use `10`.

## Model Configuration

Open `Config.py` and select HydroTC-Net:

```python
model_name = 'HydroTC_Net'
```

Important configuration values include:

```python
n_channels = 38
img_size = (560, 624)
batch_size = 4
epochs = 500
learning_rate = 3e-5
flood_threshold = 0.05
```

## Training

Run the following command from the project root:

```bash
python train_flood.py
```

Training outputs are saved by default under `FloodPrediction/HydroTC_Net/<session_name>/`, including `models/`, `tensorboard_logs/`, `visualize_val/`, and the training log. To launch TensorBoard:

```bash
tensorboard --logdir FloodPrediction/HydroTC_Net
```

## Evaluation

```bash
python test_flood.py \
  --checkpoint "FloodPrediction/HydroTC_Net/<session_name>/models/best_model-HydroTC_Net.pth.tar" \
  --test_dir "data/test" \
  --dem_path "data/static/dem.input" \
  --save_dir "results_hydrotc"
```

Windows PowerShell:

```powershell
python .\test_flood.py --checkpoint ".\FloodPrediction\HydroTC_Net\<session_name>\models\best_model-HydroTC_Net.pth.tar" --test_dir ".\data\test" --dem_path ".\data\static\dem.input" --save_dir ".\results_hydrotc"
```

The evaluation output includes `overall_metrics.csv`, `event_metrics.csv`, and event-level prediction maps, target maps, error maps, and NumPy arrays.

## Reproducibility Notes

- The default model input size is `(560, 624)`; the original DEM is padded to a multiple of 16.
- The default rainfall normalization factor is `320 mm/h`, and the target-time normalization range is `180 min`.
- The default inundation threshold is `0.05 m`; F1 and IoU are computed using this threshold.
- Training and evaluation must use the same channel count, DEM preprocessing, and normalization parameters.
- Datasets, model weights, and generated results should generally be excluded from GitHub with a `.gitignore` file.

## Code Availability

After the repository is made public, the following statement can be used in the paper:

```latex
\section*{Code Availability}

The source code of HydroTC-Net is available at
\url{https://github.com/zhoutong634/HydroTC-Net}.
```

Replace `<username>` with the GitHub username and create a versioned release for reproducible citation.

## License

This repository does not currently include a separate license. Before public release, add a `LICENSE` file and select a license consistent with the code sources and publication requirements.
