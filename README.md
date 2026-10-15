# HydroTC-Net

HydroTC-Net 是一个面向城市内涝预测的深度学习代理模型。模型根据降雨历史、数字高程模型（DEM）和目标时间，预测二维城市积水深度场。

## 项目描述

城市内涝的高分辨率水动力模拟通常需要较高的计算成本，难以满足实时预警、多情景分析和快速方案评估的需求。HydroTC-Net 是一个用于城市暴雨积涝时空预测的深度学习代理模型，旨在用较低的推理成本近似高分辨率水动力模型生成的积水深度场。

模型以因果降雨历史、数字高程模型和显式目标时间编码作为输入，在 U-Net 编码器—解码器结构基础上引入跨尺度通道特征融合和通道交叉注意力机制，使网络能够根据当前解码状态选择性利用不同尺度的空间特征。HydroTC-Net 采用非自回归预测方式，可对降雨事件中的不同目标时刻进行独立查询，避免连续预测中误差逐步累积的问题。

项目同时提供复合损失函数、数据预处理、训练、验证和多事件测试流程。模型输出为二维水深栅格，并支持生成预测结果、目标结果、误差图以及 MAE、RMSE、R²、Bias、NSE、F1 和 IoU 等评价指标。该代码可用于城市内涝代理建模、洪涝风险评估和深度学习水动力模拟研究。

本项目基于 PyTorch，支持训练和多事件测试，并计算 MAE、RMSE、R²、Bias、NSE、F1 和 IoU 等指标。

## 主要特点

- 非自回归的目标时间查询：每个目标时刻独立预测，不依赖前一时刻的预测水深。
- 输入由 36 个降雨历史通道、1 个 DEM 通道和 1 个目标时间通道组成，共 38 个通道。
- 以 U-Net 编码器解码器为基础，引入 Channel Transformer 和通道交叉注意力。
- 复合损失同时约束连续水深、淹没范围、空间梯度和低估误差。
- 支持对多个降雨事件进行统一测试并输出空间预测图、NumPy 文件和指标 CSV。

## 项目结构

```text
HydroTC-Net/
├── nets/
│   ├── HydroTC_Net.py       # HydroTC-Net 模型
│   ├── CTrans.py            # Channel Transformer 模块
│   ├── UNet.py              # U-Net 基线
│   └── WideUNet.py          # 宽 U-Net 基线
├── Config.py                # 训练和数据路径配置
├── flood_dataset.py         # 数据读取、归一化和输入构造
├── train_flood.py           # 训练入口
├── Train_one_epoch.py       # 单轮训练与验证
├── test_flood.py            # 多事件测试入口
└── utils.py                 # 复合损失和评价指标
```

## 环境安装

建议使用 Python 3.9 或更高版本，并根据 CUDA 版本安装对应的 PyTorch。

```bash
conda create -n hydrotc python=3.10 -y
conda activate hydrotc
pip install torch torchvision torchaudio
pip install numpy pandas xarray netCDF4 scipy scikit-learn tqdm tensorboardX ml-collections matplotlib
```

## 数据目录

默认数据根目录为项目目录下的 `data/`：

```text
data/
├── static/dem.input
├── train/<event_name>/{rain.txt,output.nc}
├── val/<event_name>/{rain.txt,output.nc}
└── test/<event_name>/{rain.txt,output.nc}
```

`dem.input` 为 ASCII 栅格 DEM，读取时跳过前 6 行头信息；`rain.txt` 为降雨序列；`output.nc` 必须包含名为 `h` 的水深变量，第一维为时间维度。无效地形值使用 `-9999`，建筑物掩膜值使用 `10`。

## 配置模型

打开 `Config.py`，将模型设置为 HydroTC-Net：

```python
model_name = 'HydroTC_Net'
```

主要配置包括：

```python
n_channels = 38
img_size = (560, 624)
batch_size = 4
epochs = 500
learning_rate = 3e-5
flood_threshold = 0.05
```

## 训练

在项目根目录执行：

```bash
python train_flood.py
```

训练结果默认保存到 `FloodPrediction/HydroTC_Net/<session_name>/`，其中包含 `models/`、`tensorboard_logs/`、`visualize_val/` 和训练日志。启动 TensorBoard：

```bash
tensorboard --logdir FloodPrediction/HydroTC_Net
```

## 测试

```bash
python test_flood.py \
  --checkpoint "FloodPrediction/HydroTC_Net/<session_name>/models/best_model-HydroTC_Net.pth.tar" \
  --test_dir "data/test" \
  --dem_path "data/static/dem.input" \
  --save_dir "results_hydrotc"
```

Windows PowerShell：

```powershell
python .\test_flood.py --checkpoint ".\FloodPrediction\HydroTC_Net\<session_name>\models\best_model-HydroTC_Net.pth.tar" --test_dir ".\data\test" --dem_path ".\data\static\dem.input" --save_dir ".\results_hydrotc"
```

测试结果包括 `overall_metrics.csv`、`event_metrics.csv`，以及每个事件的预测图、目标图、误差图和 NumPy 数组。

## 复现实验注意事项

- 默认输入尺寸为 `(560, 624)`，原始 DEM 会填充到 16 的倍数。
- 默认降雨归一化因子为 `320 mm/h`，目标时间归一化范围为 `180 min`。
- 默认淹没阈值为 `0.05 m`，F1 和 IoU 按该阈值计算。
- 训练和测试必须使用一致的输入通道数、DEM 预处理方式和归一化参数。
- 数据集、模型权重和生成结果不建议提交到 GitHub，应使用 `.gitignore` 排除。

## 代码可用性

正式公开仓库后，可在论文中使用：

```latex
\section*{Code Availability}

The source code of HydroTC-Net is available at
\url{https://github.com/<username>/HydroTC-Net}.
```

请将 `<username>` 替换为 GitHub 用户名，并在发布后固定一个 release 版本。

## 许可协议

当前仓库未附带独立许可证。公开发布前，建议添加 `LICENSE` 文件，并根据代码来源和论文要求选择合适的开源协议。
