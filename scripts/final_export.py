#!/usr/bin/env python3
"""Export corrected .pt to a unified final_export bundle (ONNX + OpenVINO)."""

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ultralytics import YOLO

CLASS_NAMES = {
    0: "hero-red",
    1: "engineer-red",
    2: "infantry3-red",
    3: "infantry4-red",
    4: "sentry-red",
    5: "drone-red",
    6: "hero-blue",
    7: "engineer-blue",
    8: "infantry3-blue",
    9: "infantry4-blue",
    10: "sentry-blue",
    11: "drone-blue",
}

SIZES = (1280, 640)


def print_names(title: str, names: dict) -> None:
    print(f"{title}:")
    for class_id, name in sorted(names.items(), key=lambda x: int(x[0])):
        print(f"  {class_id}: {name}")


def move_export(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    shutil.move(str(src), str(dst))


def final_export(input_pt: Path, output_dir: Path, half: bool = True) -> None:
    if not input_pt.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {input_pt}")

    model = YOLO(str(input_pt))
    names = dict(model.names)
    print(f"Input: {input_pt}")
    print_names("Model names", names)

    output_dir = output_dir.resolve()
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pt_dst = output_dir / input_pt.name
    shutil.copy2(input_pt, pt_dst)

    artifacts = []
    for imgsz in SIZES:
        # ONNX
        onnx_path = Path(
            model.export(format="onnx", imgsz=imgsz, half=half, simplify=True, device="cpu")
        )
        onnx_dst = output_dir / "onnx" / str(imgsz) / f"{input_pt.stem}_{imgsz}.onnx"
        move_export(onnx_path, onnx_dst)
        artifacts.append(str(onnx_dst.relative_to(output_dir)))

        # OpenVINO
        ov_path = Path(
            model.export(format="openvino", imgsz=imgsz, half=half, device="cpu")
        )
        ov_dst = output_dir / "openvino" / str(imgsz) / f"{input_pt.stem}_{imgsz}_openvino_model"
        move_export(ov_path, ov_dst)
        artifacts.append(str(ov_dst.relative_to(output_dir)))

    manifest = {
        "source": str(input_pt),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "half": half,
        "imgsz": list(SIZES),
        "names": {str(k): v for k, v in CLASS_NAMES.items()},
        "artifacts": artifacts,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "classes.txt").write_text(
        "\n".join(CLASS_NAMES[i] for i in range(len(CLASS_NAMES))) + "\n",
        encoding="utf-8",
    )

    print(f"\nFinal export saved to: {output_dir}")
    for item in sorted(output_dir.rglob("*")):
        if item.is_file():
            print(f"  {item.relative_to(output_dir)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export ONNX + OpenVINO into final_export/")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("model/best_fixed_names.pt"),
        help="Input .pt (default: model/best_fixed_names.pt)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output dir (default: final_export/<stem>)",
    )
    parser.add_argument(
        "--half",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="FP16 export (default: true)",
    )
    args = parser.parse_args()
    output_dir = args.output_dir or Path("final_export") / args.input.stem
    final_export(args.input, output_dir, half=args.half)


if __name__ == "__main__":
    main()
