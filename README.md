# MI-DETR

Official repository for **Motion Integration DETR (MI-DETR)**, a framework for moving infrared small target detection with explicit motion modeling and appearance–motion interaction.

**Target journal:** IEEE Transactions on Image Processing (TIP).

This README follows the revised manuscript terminology: **Recurrent Interpretable Motion Cue Aggregation (RIMCA)** and **Pathway Mutual Interaction (PMI)**. The repository provides the detector implementation, model configurations, training and validation entry points, and dataset/checkpoint download instructions.

## Abstract

Detecting moving infrared small targets is challenging because tiny, low-contrast targets occupy few pixels and are easily obscured by dynamic backgrounds. Existing multi-frame methods aggregate temporal information across frames to capture motion. However, dynamic background changes can generate similar motion cues, making it difficult to distinguish between target motion and background interference. Furthermore, even when motion cues are extracted, combining them with current-frame appearance features remains difficult. To address these issues, we propose Motion Integration DETR (MI-DETR), a three-stage framework that explicitly models motion and fuses it with appearance features. First, to suppress background clutter while preserving target-related motion cues, Recurrent Interpretable Motion Cue Aggregation (RIMCA) maintains a recurrent temporal state that accumulates motion across consecutive frames, producing a causal and spatially aligned motion representation. Second, to integrate spatial and temporal information, Pathway Mutual Interaction (PMI) preserves separate appearance and motion pathways while enabling bidirectional feature exchange between them. Finally, an RT-DETR-based detector uses these refined features for end-to-end target localization. Experiments on DAUB-R, ITSDT-15K, and IRDST-H show that explicit motion modeling and pathway interaction effectively improve moving infrared small target detection.

Our code is available at [github.com/nliu-25/MI-DETR](https://github.com/nliu-25/MI-DETR). A LaTeX version of this abstract is provided in [docs/abstract.tex](./docs/abstract.tex).

## Overview

MI-DETR has three stages:

1. **RIMCA — motion representation.** A recurrent temporal state aggregates motion across consecutive frames to produce a causal motion representation aligned with the current appearance image.
2. **PMI — pathway interaction.** Separate appearance and motion pathways exchange features bidirectionally.
3. **RT-DETR-based detection.** The refined features support end-to-end target localization.

The motion representation depends on temporal history, with one newly acquired frame at each step. In this code snapshot, training and validation load **precomputed motion maps**; they do not run RIMCA online. A standalone RIMCA preprocessing entry point is not included in this snapshot.

## Paper and Version Information

- The description and abstract above correspond to the revised manuscript targeting **TIP**.
- Existing download names such as `Dataset_retina` and directory examples ending in `_retina` are retained for compatibility with the available files. They are not new TIP-specific dataset or checkpoint releases.

## Model Summary

The default model configuration is [brain_fuse.yaml](./improve_multimodal/our_resnet18_brain/brain_fuse.yaml), which uses **300 object queries**. A separate [brain_fuse_400.yaml](./improve_multimodal/our_resnet18_brain/brain_fuse_400.yaml) is also provided. Choose the configuration associated with the experiment being reproduced; the default alone does not establish a match to a manuscript result.

Key points:

- The model uses a 6-channel input.
- `images/` denotes the **appearance modality**.
- `image/` denotes the **motion modality**.
- During data loading, each file under `images/...` is paired with the file of the same name under `image/...`.
- The loader concatenates the paired motion and appearance inputs to form the 6-channel input used by the model.
- The backbone is a dual-branch architecture with appearance and motion streams.
- PMI is implemented at P3 using two independent `TransformerFusionBlock` instances with reversed pathway inputs. P4 and P5 are subsequently extracted from the two P3 outputs.
- The fused multi-scale features are finally fed into the RT-DETR decoder.

## Repository Structure

```text
MI-DETR/
├── checkpoints/                  # checkpoint placeholder and usage notes
├── datasets/                     # dataset placeholder and layout notes
├── docs/
│   ├── abstract.tex              # revised manuscript abstract
│   └── DOWNLOADS.md              # dataset and checkpoint download instructions
├── improve_multimodal/
│   └── our_resnet18_brain/
│       ├── brain_fuse.yaml       # default model config
│       └── brain_fuse_*.yaml     # additional configs
├── ultralytics/                  # local Ultralytics-based implementation
├── data.yaml                     # default dataset template
├── train.py                      # unified training entry
├── val.py                        # unified validation entry
├── requirements.txt
├── .gitignore
└── LICENSE
```

## Environment Setup

Python 3.10 or 3.11 is recommended. If you use CUDA, install the PyTorch and torchvision build that matches your local CUDA environment first, then install the remaining dependencies.

```bash
conda create -n midetr python=3.10 -y
conda activate midetr

pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset Preparation

### 1. Raw Datasets

Please refer to the MoPKL repository for the download links of the original DAUB-R, ITSDT-15K, and IRDST-H datasets:

- <https://github.com/UESTC-nnLab/MoPKL>

### 2. Precomputed Motion Maps and Checkpoints

The existing download bundle provides paired appearance images, precomputed motion maps, and checkpoints. Its original file name is retained:

- File name: `Dataset_retina`
- Baidu Netdisk: <https://pan.baidu.com/s/1p5409A7rldXrFzzcwC_ALQ?pwd=5paw>
- Extraction code: `5paw`

More details are available in [docs/DOWNLOADS.md](./docs/DOWNLOADS.md).

### 3. Expected Dataset Layout

The default layout is:

```text
datasets/
└── DAUB-R_retina/
    ├── images/
    │   ├── train/
    │   └── test/
    ├── image/
    │   ├── train/
    │   └── test/
    └── labels/
        ├── train/
        └── test/
```

Important notes:

- `images/` is the **appearance modality** directory.
- `image/` is the **motion modality** directory.
- File names under `images/` and `image/` must be strictly aligned one by one.
- The loader maps `images/.../xxx.png` to `image/.../xxx.png` automatically.
- Labels follow the standard YOLO detection format.
- The bundled `data.yaml` retains the legacy mapping `val: images/test`, with no separate `test` entry. For new experiments, create a dataset YAML with independent training, validation, and test splits. Use validation data for checkpoint selection and evaluate the frozen checkpoint on test data.

### 4. How to Use `data.yaml`

The default dataset template is `data.yaml`. There are two common ways to use it:

1. Put the dataset inside the repository under `datasets/` and use `data.yaml` directly.
2. Keep the dataset anywhere on your machine and override the dataset root with `--dataset-root`.

Example:

```bash
python train.py --data data.yaml --dataset-root /path/to/DAUB-R_retina --device 0
```

For ITSDT-15K or IRDST-H, switch `--dataset-root` to the corresponding paired dataset. Copy and edit `data.yaml` to match the actual split definitions; changing the dataset root does not change the split mapping.

## Training

`train.py` is the unified public entry point and does not keep any local absolute paths or fixed GPU IDs.

### Single-GPU Training

```bash
python train.py \
  --data data.yaml \
  --dataset-root /path/to/DAUB-R_retina \
  --device 0 \
  --batch 32 \
  --epochs 600 \
  --imgsz 512 \
  --name daub-r
```

### Multi-GPU Training

```bash
python train.py \
  --data data.yaml \
  --dataset-root /path/to/DAUB-R_retina \
  --device 0,1 \
  --batch 32 \
  --epochs 600 \
  --imgsz 512 \
  --name daub-r_ddp
```

Notes:

- `--device 0` runs single-GPU training.
- `--device 0,1` runs multi-GPU training through Ultralytics DDP.
- `--batch` is the global batch size and will be split automatically across GPUs in DDP mode.
- The default script keeps the main training settings used in the public release, including `AdamW`, `imgsz=512`, `epochs=600`, and `close_mosaic=80`.

## Validation

Put released checkpoints into `checkpoints/`, or pass an external path with `--weights`.

```bash
python val.py \
  --weights checkpoints/DAUB-R.pt \
  --data data.yaml \
  --dataset-root /path/to/DAUB-R_retina \
  --device 0 \
  --imgsz 512 \
  --batch 1 \
  --name daub-r_val
```

For other datasets, switch:

- `--weights`
- `--dataset-root`
- `--data` if needed

## Reproducibility Notes

For a traceable experiment:

- record the dataset version, motion-map generation procedure, frame order, and sequence reset boundaries
- keep `images/` as the appearance modality and `image/` as the motion modality
- keep strict one-to-one pairing between the two modality folders
- record the exact model configuration, training arguments, code revision, and checkpoint hash
- use the image size and evaluation protocol associated with the result being reproduced
- use an independent validation split to select `best.pt` before final test evaluation

The revised terminology does not establish that existing checkpoints or precomputed motion maps reproduce every revised-manuscript experiment. Detector timing from the provided validation path excludes online motion-map generation and should not be reported as full video-to-detection throughput.

By default, outputs are saved to:

- training: `runs/train/<name>`
- validation: `runs/val/<name>`

## Open-Source Release Notes

This public repository has been cleaned up to:

- remove local absolute paths
- unify training and validation entry points
- remove caches, zip files, temporary runtime results, and large checkpoint files
- provide the detector training and validation code with paired appearance/motion inputs

This documentation update aligns the project description with the revised manuscript. It does not change the detector, regenerate motion maps, or introduce new experimental results.

## License

This repository contains a modified Ultralytics-based implementation. To stay consistent with the upstream licensing basis, this repository is released under **AGPL-3.0**. See [LICENSE](./LICENSE) for details.
