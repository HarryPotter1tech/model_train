# RADAR-MODEL

基于 Ultralytics YOLO26n 的雷达目标检测训练项目。

## 类别（12类）

| id | 类别 | id | 类别 |
|----|------|----|------|
| 0 | hero-red | 1 | hero-blue |
| 2 | engine-red | 3 | engine-blue |
| 4 | infantry_3-red | 5 | infantry_3-blue |
| 6 | infantry_4-red | 7 | infantry_4-blue |
| 8 | sentry-red | 9 | sentry-blue |
| 10 | drone-red | 11 | drone-blue |

## 目录结构

```
RADAR-MODEL/
├── config_stage1.yaml      # 阶段1: 300epoch @1280
├── config_stage2.yaml      # 阶段2: 500epoch @640 (resume stage1)
├── config_stage3.yaml      # 阶段3: 700epoch @1280 (resume stage2)
├── train.py                # 训练入口
├── data/
│   └── radar_dataset/
│       ├── images/train/   # 训练图片 (YOLO格式)
│       ├── images/val/     # 验证图片
│       ├── labels/train/   # 训练标签 (.txt)
│       └── labels/val/     # 验证标签
├── runs/train/             # 训练输出（自动生成）
│   ├── stage1/
│   ├── stage2/
│   └── stage3/
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

## 快速开始

### 本地训练

```bash
# 运行全部三阶段（默认）
python train.py

# 运行指定阶段
python train.py --config config_stage1.yaml
python train.py --config config_stage1.yaml config_stage2.yaml config_stage3.yaml

# 可视化
tensorboard --logdir runs/train
```

### Docker 训练

```bash
# 构建并训练（自动挂载 data/ 和 runs/）
docker compose up

# 单阶段
docker compose run --rm radar-train python train.py --config config_stage1.yaml

# TensorBoard
docker compose run --rm -p 6006:6006 radar-train tensorboard --logdir runs/train --bind_all
```

## 训练策略

| 阶段 | 配置 | epochs | imgsz | 说明 |
|------|------|--------|-------|------|
| 1 | config_stage1.yaml | 300 | 1280 | 高分辨率初始训练 |
| 2 | config_stage2.yaml | 500 | 640 | 低分辨率微调 |
| 3 | config_stage3.yaml | 700 | 1280 | 高分辨率精调 |

每阶段自动加载上一阶段最优权重（`runs/train/stageN/weights/best.pt`），训练完成后自动导出 ONNX + OpenVINO。

## 数据准备

图片放入 `data/radar_dataset/images/{train,val}/`，对应 YOLO 格式标签放入 `data/radar_dataset/labels/{train,val}/`。

标签格式：每行 `class_id x_center y_center width height`（归一化 0-1）。

## 模型导出

训练完成后自动导出到 `runs/train/stageN/weights/`：

```
best.onnx
best_openvino_model/
```

也可手动导出：

```python
from ultralytics import YOLO
YOLO("runs/train/stage3/weights/best.pt").export(format="onnx", half=True)
YOLO("runs/train/stage3/weights/best.pt").export(format="openvino", half=True)
```
