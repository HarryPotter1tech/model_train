import random
import shutil
from pathlib import Path

SRC = Path("/home/pinkpanda/Downloads/Done/Done")
DST = Path("/home/pinkpanda/Downloads/RadarDataset")
TRAIN_RATIO = 0.8

samples = []
source_dirs = ["其他", "哈工大", "大连理工"]

for src in source_dirs:
    img_dir = SRC / src / "images"
    lbl_dir = SRC / src / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        print(f"skip {src}: images or labels not found")
        continue
    for img in sorted(img_dir.glob("*.jpg")):
        lbl = lbl_dir / img.with_suffix(".txt").name
        if lbl.exists():
            samples.append((img, lbl))
    print(f"{src}: {len(list(img_dir.glob('*.jpg')))} images, {len(samples)} samples (with label)")

print(f"\nTotal samples with labels: {len(samples)}")

random.seed(42)
random.shuffle(samples)

split = int(len(samples) * TRAIN_RATIO)
for i, (img, lbl) in enumerate(samples):
    prefix = "train" if i < split else "val"
    img_dst = DST / "images" / prefix / img.name
    lbl_dst = DST / "labels" / prefix / lbl.name
    img_dst.parent.mkdir(parents=True, exist_ok=True)
    lbl_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img, img_dst)
    shutil.copy2(lbl, lbl_dst)

print(f"Done: {split} train, {len(samples) - split} val")
