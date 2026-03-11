import argparse
import tempfile
import warnings
from pathlib import Path

import yaml

from ultralytics import RTDETR

warnings.filterwarnings("ignore")

DEFAULT_MODEL = "improve_multimodal/our_resnet18_brain/brain_fuse.yaml"
DEFAULT_DATA = "data.yaml"


def resolve_data_config(data_path: str, dataset_root: str | None) -> str:
    """Resolve dataset YAML to an absolute-path temporary config for reproducible public release usage."""
    data_file = Path(data_path).expanduser().resolve()
    config = yaml.safe_load(data_file.read_text(encoding="utf-8"))

    if dataset_root:
        config["path"] = str(Path(dataset_root).expanduser().resolve())
    elif config.get("path"):
        config["path"] = str((data_file.parent / Path(config["path"]).expanduser()).resolve())

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False, encoding="utf-8") as tmp:
        yaml.safe_dump(config, tmp, allow_unicode=True, sort_keys=False)
        return tmp.name


def parse_args():
    parser = argparse.ArgumentParser(description="Train MI-DETR with single-GPU or multi-GPU Ultralytics DDP.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Model config (.yaml) or checkpoint (.pt).")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Dataset YAML file.")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root path defined in the YAML file.")
    parser.add_argument("--device", default="", help="Training device, e.g. '0' or '0,1'.")
    parser.add_argument("--epochs", type=int, default=600, help="Number of training epochs.")
    parser.add_argument("--imgsz", type=int, default=512, help="Input image size.")
    parser.add_argument("--batch", type=int, default=32, help="Global batch size.")
    parser.add_argument("--workers", type=int, default=16, help="Number of dataloader workers.")
    parser.add_argument("--project", default="runs/train", help="Directory to save training runs.")
    parser.add_argument("--name", default="mi-detr", help="Run name.")
    parser.add_argument("--amp", action="store_true", help="Enable AMP training.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_config = resolve_data_config(args.data, args.dataset_root)
    model = RTDETR(args.model)
    model.train(
        data=data_config,
        ch=6,
        epochs=args.epochs,
        imgsz=args.imgsz,
        workers=args.workers,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        optimizer="AdamW",
        amp=args.amp,
        patience=200,
        close_mosaic=80,
        max_det=1000,
        lr0=0.00008,
        lrf=0.0008,
        momentum=0.937,
        weight_decay=0.0008,
        warmup_epochs=15.0,
        warmup_momentum=0.8,
        warmup_bias_lr=0.1,
        box=12.0,
        cls=2.5,
        dfl=2.5,
        pose=12.0,
        kobj=1.0,
        label_smoothing=0.05,
        nbs=64,
        hsv_h=0.005,
        hsv_s=0.3,
        hsv_v=0.2,
        degrees=3.0,
        translate=0.08,
        scale=0.25,
        shear=1.0,
        perspective=0.00005,
        flipud=0.0,
        fliplr=0.5,
        bgr=0.0,
        mosaic=0.2,
        mixup=0.05,
        copy_paste=0.05,
        copy_paste_mode="mixup",
        auto_augment="randaugment",
        erasing=0.05,
        crop_fraction=1.0,
    )


if __name__ == "__main__":
    main()

