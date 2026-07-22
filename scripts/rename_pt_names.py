#!/usr/bin/env python3
"""Rename Ultralytics checkpoint class-name metadata without changing weights."""

import argparse
from pathlib import Path

import torch


NAMES = {
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


def print_names(title: str, names: dict) -> None:
    print(f"{title}:")
    for class_id, name in names.items():
        print(f"  {class_id}: {name}")


def default_output_path(input_path: Path) -> Path:
    return input_path.with_name(f"{input_path.stem}_fixed_names{input_path.suffix}")


def rename_names(input_path: Path, output_path: Path) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {input_path}")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Output path must differ from the input path to preserve the original checkpoint.")

    checkpoint = torch.load(input_path, map_location="cpu", weights_only=False)
    model = checkpoint.get("model")
    if model is None or not hasattr(model, "names"):
        raise ValueError("This checkpoint does not contain model.names metadata.")

    old_names = dict(model.names)
    print(f"Input: {input_path}")
    print_names("Old names", old_names)

    model.names = NAMES.copy()
    ema = checkpoint.get("ema")
    if ema is not None and hasattr(ema, "names"):
        ema.names = NAMES.copy()

    for key in ("args", "train_args"):
        metadata = checkpoint.get(key)
        if isinstance(metadata, dict) and "names" in metadata:
            metadata["names"] = NAMES.copy()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output_path)

    saved_checkpoint = torch.load(output_path, map_location="cpu", weights_only=False)
    saved_model = saved_checkpoint["model"]
    new_names = dict(saved_model.names)
    if new_names != NAMES:
        raise RuntimeError("Saved checkpoint class names do not match the requested mapping.")

    print_names("New names", new_names)
    print(f"Saved to: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rename Ultralytics .pt class-name metadata without changing model weights."
    )
    parser.add_argument("input", type=Path, help="Input Ultralytics .pt checkpoint")
    parser.add_argument(
        "--output",
        type=Path,
        help="Output checkpoint path. Defaults to <input>_fixed_names.pt",
    )
    args = parser.parse_args()

    rename_names(args.input, args.output or default_output_path(args.input))


if __name__ == "__main__":
    main()
