
# Adaptive NeRF Framework for City-Scale Emergency Awareness

This repository provides the real-system integration of the Adaptive NeRF framework introduced in the paper:

```bibtex
@inproceedings{sklavos2026adaptivenerf,
  title     = {An Adaptive NeRF Framework for City-Scale Emergency Awareness},
  author    = {Panagiotis Sklavos and Georgios Anestis and Antonios Deligiannakis},
  booktitle = {Proceedings of the 14th International Conference on Emerging Internet, Data \& Web Technologies (EIDWT 2026)},
  year      = {2026},
  note      = {Accepted for publication}
}
```
for the European Union’s Horizon project: [**CREXDATA**](https://crexdata.eu/).

This framework implements a city-scale Neural Radiance Field (NeRF) that can be updated incrementally as new aerial images become available. Instead of retraining a single NeRF from scratch, the framework:

- Uses compact and efficient NeRF architectures
- Decomposes large scenes into spatial NeRF experts
- Pre-trains these experts to form a robust initialization
- Updates only the relevant experts when new data arrive

This allows the model to:
- Adapt rapidly to scene changes
- Preserve previously learned structure
- Scale to large outdoor environments

---

## Overview

The system is centered around a persistent orchestrator, called the **Mediator**, that is designed to run as a long-lived process that orchestrates NeRF workloads based on incoming job configurations. 

The mediator continuously listens for structured JSON job descriptions to a known **Kafka** topic and launches **NeRF jobs** to handle the received configurations. 
Jobs are executed in isolation and are launched in a non-blocking manner, allowing multiple jobs to run concurrently or sequentially in the background

Each job corresponds to one of the following operations:
- **Offline Training** using meta-learning: Learn a coarse initialization of the area, that allows rapid adaptaton.
- **Runtime Evaluation**: Adapt rapidly to new data and evaluate reconstruction metrics like PSNR, SSIM, and LPIPS.
- **Interactive Visualization**: Launch the NeRF viewer for navigation or live adaptation monitoring.



![Overview of the adaptive NeRF system architecture](nerf_arch.jpg)

## Structure

```text
.
├── adaptive_nerf/                  # Core Adaptive NeRF framework
│   ├── common/                     # Shared utilities (logging, helpers)
│   ├── data/                       # Dataset storage and parsing
│   ├── models/                     # NeRF models and architectures
│   ├── nerfs/                      # NeRF-specific helpers
│   ├── pipelines/                  # Meta-training and runtime adaptation pipelines
│   ├── scripts/                    # Internal execution scripts for data preparation or logging statistics
│   ├── viewer/                     # Interactive NeRF viewer
│   ├── nerf_runner.py              # Entry point for executing NeRF operations
│   ├── utils.py                    # Entry point utilities
│   └── __init__.py
│
├── configs/                        # Job and experiment configuration files
├── kafka_utils/                    # Kafka helpers for the server communication
├── logs/                           # Job outputs, logs, and checkpoints
├── scripts/                        # System-level utility scripts
│
├── mediator.py                     # Long-running orchestrator (system entry point)
├── README.md                       # System documentation
└── requirements.txt                # Python dependencies

```

---

## Environment Setup
To get started, clone the repository, create a Conda environment, and install the required dependencies.
Python 3.11 and CUDA 11.8 are verified for compatibility.

### 1. Clone the repository
```bash
git clone https://github.com/psklavos1/NeRF-Sys.git
cd NeRF-Sys
```

### 2) Create the environmet.
We provide an example setup using conda.
Install the correct version of PyTorch and dependencies:
```bash
conda create -n nerfenv python=3.11 -y
conda activate nerfenv
conda install -y pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

### 3) Install tiny-cuda-nn.
Efficient Instant-NGP primitives require `tiny-cuda-nn`.

Make sure CUDA is compatible:
```bash
conda install -y -c nvidia cuda-nvcc=11.8 cuda-cudart-dev=11.8
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CONDA_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
nvcc --version   # should say 11.8
```

Install using the official [NVLabs](https://github.com/NVlabs/tiny-cuda-nn?utm_source) instructions:
```bash
pip install --no-build-isolation -v "git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch"
```
In case of failure assert all the provided [requirements](https://github.com/NVlabs/tiny-cuda-nn?utm_source) are satisfied.

---

## Data Preparation
The framework operates on image-based datasets with known or recoverable camera poses.
Camera intrinsics and poses must be estimated prior to use. [COLMAP](https://colmap.github.io/) is the
recommended tool for this step. If GPS coordinates are not available, COLMAP’s model alignment utilities
(e.g. Manhattan-world alignment) can be used as a fallback as the data need to be geo-referenced.

> COLMAP models can be exported in either **ECEF** or **ENU** coordinates. The recommended
workflow is to export in **ECEF** and perform the ECEF→ENU conversion internally using
the provided scripts. Direct ENU exports are also supported.

### Dataset Preparation Scripts

The following scripts are provided for dataset preparation and clustering.

- `adaptive_nerfs/scripts/prepare_dataset.py` 
converts a COLMAP reconstruction into the framework’s internal dataset format. The
script normalizes scene scale for stable training and optionally converts poses to a
common ENU reference frame, storing all outputs using the framework’s coordinate
conventions.

**Input Data:**
```text
data_path/
  ├── model/    # COLMAP sparse model (cameras.bin, images.bin, points3D.bin)
  └── images/   # All registered images used by the COLMAP model
```
### Example Usage
```bash
./scripts/prepare_dataset.py --data_path data/drz --output_path data/drz/out/prepared --val_split 0.3 --scale_strategy camera_max --ecef_to_enu --enu_ref median
```

- `scripts/create_clusters.py` generates spatial training partitions for NeRF experts using distance-based routing.
Rays are assigned to one or more spatial regions, with optional boundary overlap.
2D clustering is recommended to reduce computation without affecting results.

### Example Usage

```bash
 ./scripts/create_clusters.py --data_path data/drz/out/prepared --grid_dim 2 2 --cluster_2d --boundary_margin 1.05 --ray_samples 256 --center_pixels --scene_scale 1.1 --output g22_grid_bm105_ss11 --resume
```

---

## Demo
To experiment with the mediator 3 demo configurations are provided in the `/configs` for training, evaluation and viewer operation respectively. Prepared data provided by: [DRZ](https://rettungsrobotik.de/) are located at `data/drz/out/example` while a demo checkpoint with 4 expert NeRFs is provided for testing: [`checkpoint`](https://github.com/psklavos1/adaptive-city-nerf/releases/tag/v1.0/4_experts.zip)

> After downloading, extract into: `logs/example/`

### Demo experiment

1. **Run Mediator**  
    ```bash
    python mediator.py --topic nerfConfigs
    ```


2. **Send Demo config**  
    ```bash
    ./scripts/send_config.py --topic nerfConfigs --configPath <configs/train.json | configs/eval.json | configs/view.json>
    ```

3.  **Monitor outputs** 

> Outputs provided at `logs/mediator.txt` for the mediator and `logs/<job_id>` for the executed job. Job outputs contain both operation artifacts like model checkpoints or rendered images and job logs.
    
---

## Configuration

Each job is defined by a JSON configuration file.  
The configuration fully specifies the dataset, model, optimization, and runtime
behavior for a single job.

All jobs are submitted through a single entry point and are distinguished by the
`op` field.


### Train (`op: "train"`)

Offline training that initializes the NeRF model and produces a checkpoint.

| Field | Description |
|------|------------|
| `op` | Operation type (`"train"`) |
| `dataset` | Dataset identifier |
| `data_path` | Root dataset directory |
| `data_dirname` | Prepared dataset name |
| `mask_dirname` | Spatial clustering masks |
| `downscale` | Image resolution scaling |
| `near`, `far` | Ray bounds |
| `ray_samples` | Samples per ray |
| `batch_size` | Number of tasks per expert |
| `support_rays` | Rays per task (support set) |
| `query_rays` | Rays per task (query set) |
| `num_submodules` | Number of NeRF experts |
| `log2_hashmap_size` | Hash grid size |
| `max_resolution` | Max encoding resolution |
| `optimizer` | Optimizer type |
| `lr` | Base learning rate |
| `outer_steps` | Meta-optimization steps |
| `inner_iter` | Inner-loop iterations |
| `inner_lr` | Inner-loop learning rate |
| `checkpoint_path` | (Optional) chekpoint to continue training from |
| `prefix` | Checkpoint prefix |

---

### Eval (`op: "eval"`)

Runtime adaptation of a pre-trained model using newly available images.

| Field | Description |
|------|------------|
| `op` | Operation type (`"eval"`) |
| `dataset` | Dataset identifier |
| `data_path` | Root dataset directory |
| `data_dirname` | Prepared dataset name |
| `mask_dirname` | Spatial clustering masks |
| `downscale` | Image resolution scaling |
| `near`, `far` | Ray bounds |
| `ray_samples` | Samples per ray |
| `support_rays` | Rays used for adaptation |
| `optimizer` | Optimizer type |
| `lr` | Adaptation learning rate |
| `tto` | Number of adaptation steps |
| `checkpoint_path` | Input checkpoint directory |
| `prefix` | Checkpoint prefix |

---

### View (`op: "view"`)

Launches the interactive NeRF viewer.

| Field | Description |
|------|------------|
| `op` | Operation type (`"view"`) |
| `dataset` | Dataset identifier |
| `data_path` | Root dataset directory |
| `data_dirname` | Prepared dataset name |
| `mask_dirname` | Spatial clustering masks |
| `downscale` | Image resolution scaling |
| `ray_samples` | Samples per ray for rendering |
| `checkpoint_path` | Input checkpoint directory |
| `prefix` | Checkpoint prefix |

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.


---

## Acknowledgments
This research is supported by the European Union’s Horizon 2020 research and innovation programme under grant agreement No. 101092749, project [**CREXDATA**](https://crexdata.eu/). We sincerely thank the members of the [Deutsches Rettungsrobotik Zentrum (DRZ)](https://rettungsrobotik.de/), for supporting the data acquisition and providing the aerial dataset used in this work.
