# MI-DETR

Official code release for **MI-DETR: A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration**.

This repository is organized for public release alongside the paper. It keeps the core training and validation pipeline used in the paper while removing local machine paths, private runtime artifacts, cached files, and large binary files that should not be uploaded to GitHub. The repository does not include raw datasets, retina-processed datasets, or large pretrained checkpoints. It only provides code, configuration files, and download instructions.

## Paper

- Title: *MI-DETR: A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration*
- arXiv: <https://arxiv.org/abs/2603.05071>

```bibtex
@misc{liu2026midetrstrongbaselinemoving,
      title={MI-DETR: A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration},
      author={Nian Liu and Jin Gao and Shubo Lin and Yutong Kou and Sikui Zhang and Fudong Ge and Zhiqiang Pu and Liang Li and Gang Wang and Yizheng Wang and Weiming Hu},
      year={2026},
      eprint={2603.05071},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.05071},
}
```

## Overview

MI-DETR is a dual-branch multimodal detector for moving infrared small target detection, built on RT-DETR. This public release focuses on the following:

- keeping the paper-consistent model structure, training settings, and validation workflow
- supporting both single-GPU and multi-GPU training
- providing a clean repository layout for reproduction
- documenting environment setup, dataset preparation, training, and validation

## Model Summary

The default model configuration is located at `improve_multimodal/our_resnet18_brain/brain_fuse.yaml`.

Key points:

- The model uses a 6-channel input.
- `images/` denotes the **appearance modality**.
- `image/` denotes the **motion modality**.
- During data loading, each file under `images/...` is paired with the file of the same name under `image/...`.
- The loader concatenates the paired motion and appearance inputs to form the 6-channel input used by the model.
- The backbone is a dual-branch architecture with appearance and motion streams.
- A bidirectional cross-modal interaction block `TransformerFusionBlock` is inserted at the P3 stage.
- The fused multi-scale features are finally fed into the RT-DETR decoder.

## Repository Structure

```text
MI-DETR/
├── checkpoints/                  # checkpoint placeholder and usage notes
├── datasets/                     # dataset placeholder and layout notes
├── docs/
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

### 2. Retina-Processed Datasets and Checkpoints

For paper reproduction, we recommend using the retina-processed datasets and the released checkpoints.

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
- The default `data.yaml` uses `images/test` as the validation split.

### 4. How to Use `data.yaml`

The default dataset template is `data.yaml`. There are two common ways to use it:

1. Put the dataset inside the repository under `datasets/` and use `data.yaml` directly.
2. Keep the dataset anywhere on your machine and override the dataset root with `--dataset-root`.

Example:

```bash
python train.py --data data.yaml --dataset-root /path/to/DAUB-R_retina --device 0
```

To reproduce ITSDT-15K or IRDST-H, switch `--dataset-root` to the corresponding retina-processed dataset. If your local split names differ from the default template, copy and edit `data.yaml` accordingly.

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

To stay close to the paper setting:

- use the retina-processed datasets
- keep `images/` as the appearance modality and `image/` as the motion modality
- keep strict one-to-one pairing between the two modality folders
- use the default model config `improve_multimodal/our_resnet18_brain/brain_fuse.yaml`
- keep `imgsz=512`, `epochs=600`, and `optimizer=AdamW`
- validate with the released checkpoint or the `best.pt` obtained from training

By default, outputs are saved to:

- training: `runs/train/<name>`
- validation: `runs/val/<name>`

## Open-Source Release Notes

This public repository has been cleaned up to:

- remove local absolute paths
- unify training and validation entry points
- remove caches, zip files, temporary runtime results, and large checkpoint files
- keep the core code path required for paper reproduction

## License

This repository contains a modified Ultralytics-based implementation. To stay consistent with the upstream licensing basis, this repository is released under **AGPL-3.0**. See [LICENSE](./LICENSE) for details.

## Citation

If you find this repository useful, please cite:

```bibtex
@misc{liu2026midetrstrongbaselinemoving,
      title={MI-DETR: A Strong Baseline for Moving Infrared Small Target Detection with Bio-Inspired Motion Integration},
      author={Nian Liu and Jin Gao and Shubo Lin and Yutong Kou and Sikui Zhang and Fudong Ge and Zhiqiang Pu and Liang Li and Gang Wang and Yizheng Wang and Weiming Hu},
      year={2026},
      eprint={2603.05071},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.05071},
}
```
