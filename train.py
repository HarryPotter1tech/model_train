import argparse
import yaml
from ultralytics import YOLO


def train(config_path: str):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    model = YOLO(cfg.pop("model"))

    data_keys = {"path", "train", "val", "names"}
    data_cfg = {k: cfg.pop(k) for k in data_keys if k in cfg}
    cfg["data"] = data_cfg

    export_cfg = cfg.pop("export", None)

    model.train(**cfg)

    if export_cfg:
        for exp in export_cfg:
            exp.setdefault("imgsz", cfg.get("imgsz", 640))
            model.export(**exp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", nargs="+",
                        default=["config_stage1.yaml",
                                 "config_stage2.yaml",
                                 "config_stage3.yaml"])
    args = parser.parse_args()
    for cfg in args.config:
        print(f"\n{'='*60}\nStart training: {cfg}\n{'='*60}")
        train(cfg)
