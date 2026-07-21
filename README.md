# RADAR-MODEL

基于 Ultralytics YOLO26n 的雷达目标检测训练项目。

## 类别（12类）

| id | 类别 | id | 类别 |
|----|------|----|------|
| 0 | hero-red | 1 | hero-blue |
| 2 | engineer-red | 3 | engineer-blue |
| 4 | infantry3-red | 5 | infantry3-blue |
| 6 | infantry4-red | 7 | infantry4-blue |
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
│       ├── images/train/   # → 放入训练图片
│       ├── images/val/     # → 放入验证图片
│       ├── labels/train/   # → 放入训练标签
│       └── labels/val/     # → 放入验证标签
├── Dockerfile
├── docker-compose.yml
└── .dockerignore
```

## 数据准备

### 格式要求

YOLO 格式，每张图片对应一个同名 `.txt` 标签文件，每行：

```
class_id x_center y_center width height
```

坐标归一化到 0-1，class_id 对应上表 0-11。

```
data/radar_dataset/
├── images/train/00001.jpg
├── images/val/00002.jpg
├── labels/train/00001.txt
└── labels/val/00002.txt
```

### 通过百度网盘获取数据

1. 从百度网盘下载 `RadarDataset.zip`
2. 解压到项目目录：

```bash
unzip RadarDataset.zip -d data/
```

最终结构：

```
data/radar_dataset/
├── images/train/   (4000 张)
├── images/val/     (1000 张)
├── labels/train/   (4000 个 .txt)
└── labels/val/     (1000 个 .txt)
```

### Docker

数据放在项目 `data/` 下即可，`docker-compose.yml` 已配置自动挂载。

## 训练

```bash
# 运行全部三阶段（默认）
python train.py

# 运行指定阶段
python train.py --config config_stage1.yaml

# 可视化
tensorboard --logdir runs/train
```

## 训练策略

| 阶段 | 配置 | epochs | imgsz | lr | 说明 |
|------|------|--------|-------|-----|------|
| 1 | config_stage1.yaml | 300 | 1280 | 0.01 | 高分辨率初始训练 |
| 2 | config_stage2.yaml | 500 | 640 | 0.001 | 低分辨率微调（resume stage1） |
| 3 | config_stage3.yaml | 700 | 1280 | 0.0005 | 高分辨率精调（resume stage2） |

训练完成后自动导出 ONNX + OpenVINO（1280 和 640 双尺寸）。

## Docker

```bash
# 构建镜像（数据不打包进镜像）
docker build -t radar-model .

# 数据放到项目 data/ 下，直接使用 docker compose
docker compose up
```
