# Neural Real-Time RGB-D SLAM in Dynamic Environments

This repository provides the official implementation of our dynamic RGB-D SLAM system based on **Co-SLAM**, designed for real-time neural scene reconstruction in dynamic environments.

Our method extends neural implicit SLAM by incorporating dynamic object handling and static point sampling, enabling robust camera tracking and scene reconstruction under non-static conditions.

---

## Features

- Real-time RGB-D neural SLAM
- Dynamic / static point separation
- Robust camera tracking in dynamic scenes
- Neural implicit scene representation
- Compatible with ScanNet-style datasets
- Built on top of Co-SLAM

---

## System Requirements

### Hardware
- NVIDIA GPU with ≥ 8GB VRAM (tested on RTX 3090 / A6000)
- 16GB+ system RAM recommended

### Software
- Ubuntu 20.04 / 22.04
- CUDA ≥ 11.3
- Python ≥ 3.8
- Conda

---

## Installation

We recommend using Conda or Docker for full reproducibility.

### Clone the repository
```bash
git clone https://github.com/yourname/dynaslam.git
cd dynaslam
```

### Create environment
```bash
conda create -n coslam python=3.8
conda activate coslam
pip install -r requirements.txt
pip install open3d
```

(Optional) Using Docker:
```bash
docker build -t dynaslam .
docker run -it --gpus all --shm-size=5G dynaslam bash
```

---

## Dataset

We follow the data format of ScanNet and Dyna3DBench.

ScanNet is for static scenes and Dyna3DBench is for the dynamic ones.

### Directory structure
```
Co-SLAM/data/scannet_d/scene0000_00_d/
 ├── color/
 ├── depth/
 ├── pose/
 └── intrinsic/
```

### Download ScanNet

Please download ScanNet from the official website:

https://github.com/ScanNet/ScanNet
https://github.com/seiyaito/Dyna3DBench

Then preprocess it into the required format.

(Preprocessing scripts are provided in `tools/`.)

---

## Running the System

```bash
cd Co-SLAM
python coslam.py --config ./configs/ScanNet/scene0000_d.yaml
```

---

## Evaluation

The system outputs:

- Camera trajectory
- Reconstructed neural map
- Static-only point clouds
- Dynamic object masks

Run evaluation:
```bash
python tools/eval_recon.py \
  --pred ./output \
  --gt ./data/scannet_d/scene0000_00_d
```

Metrics:
- IoU
- Precision / Recall
- Absolute Trajectory Error (ATE)

---

## Configuration

Main configs:
```
configs/ScanNet/*.yaml
```

Important parameters:

| Parameter | Description |
|----------|-------------|
| mapping_iters | Mapping iterations per frame |
| sampling_strategy | static / dynamic |
| mask_threshold | Dynamic object threshold |
| voxel_size | Reconstruction resolution |

---

## Reproducibility

Random seeds:
```yaml
seed: 42
```

All reported results:
- RTX 3090
- CUDA 11.7
- PyTorch 1.13

---


## Contact

Qinyuan Zhou  
Email: zhouqinyuanrunner@gmail.com
Website: https://zhouqinyuan.com

