import argparse
import tempfile
import warnings
from pathlib import Path

import yaml

from ultralytics import RTDETR

warnings.filterwarnings("ignore")

DEFAULT_DATA = "data.yaml"
DEFAULT_WEIGHTS = "checkpoints/DAUB-R.pt"


def resolve_data_config(data_path: str, dataset_root: str | None) -> str:
    """Resolve dataset YAML to an absolute-path temporary config for validation."""
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
    parser = argparse.ArgumentParser(description="Validate MI-DETR checkpoints.")
    parser.add_argument("--weights", default=DEFAULT_WEIGHTS, help="Checkpoint path.")
    parser.add_argument("--data", default=DEFAULT_DATA, help="Dataset YAML file.")
    parser.add_argument("--dataset-root", default=None, help="Override dataset root path defined in the YAML file.")
    parser.add_argument("--device", default="", help="Validation device, e.g. '0'.")
    parser.add_argument("--imgsz", type=int, default=512, help="Input image size.")
    parser.add_argument("--batch", type=int, default=1, help="Validation batch size.")
    parser.add_argument("--split", default="val", help="Dataset split to evaluate.")
    parser.add_argument("--project", default="runs/val", help="Directory to save validation runs.")
    parser.add_argument("--name", default="mi-detr", help="Run name.")
    return parser.parse_args()


def main():
    args = parse_args()
    data_config = resolve_data_config(args.data, args.dataset_root)
    model = RTDETR(args.weights)
    model.val(
        data=data_config,
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        device=args.device,
        project=args.project,
        name=args.name,
        save_json=True,
        verbose=False,
    )


if __name__ == "__main__":
    main()
 
