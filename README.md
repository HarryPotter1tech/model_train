# RADAR-MODEL

基于 Ultralytics YOLO26n 的雷达目标检测训练项目。

---

## 一、环境要求

- Python 3.10+
- CUDA 12.4+（GPU 训练）
- 显存：推荐 48GB × 8 卡（L40）或等量算力
- 依赖：`torch`, `ultralytics`, `opencv-python`, `pyyaml`

---

## 二、项目目录结构

```
RADAR-MODEL/
├── config_stage1.yaml       # 阶段1 训练参数
├── config_stage2.yaml       # 阶段2 训练参数
├── config_stage3.yaml       # 阶段3 训练参数
├── train.py                 # 训练脚本
├── Dockerfile               # Docker 构建
├── docker-compose.yml       # Docker 编排
├── .dockerignore
├── README.md
└── data/
    └── radar_dataset/       # 数据目录（需自行准备）
        ├── images/
        │   ├── train/       # 训练图片
        │   └── val/         # 验证图片
        └── labels/
            ├── train/       # 训练标签
            └── val/         # 验证标签
```

---

## 三、数据准备

### 3.1 格式说明

YOLO 格式，每张 `.jpg` 图片对应一个同名 `.txt` 标签文件。每行代表一个目标：

```
class_id x_center y_center width height
```

- 坐标值归一化到 0-1（相对于图片宽高）
- `class_id` 取值为 0-11

```
data/radar_dataset/
├── images/train/000001.jpg
├── images/train/000002.jpg
├── labels/train/000001.txt
├── labels/train/000002.txt
├── images/val/...
└── labels/val/...
```

### 3.2 类别映射表

| id | 类别 | id | 类别 |
|----|------|----|------|
| 0 | hero-red | 1 | hero-blue |
| 2 | engineer-red | 3 | engineer-blue |
| 4 | infantry3-red | 5 | infantry3-blue |
| 6 | infantry4-red | 7 | infantry4-blue |
| 8 | sentry-red | 9 | sentry-blue |
| 10 | drone-red | 11 | drone-blue |

### 3.3 百度网盘获取数据

数据已标注整理完成，通过百度网盘分发。

**步骤：**

```bash
# 1. 下载 RadarDataset.zip（从网盘）
# 2. 放到项目目录下
# 3. 解压到 data/
unzip RadarDataset.zip -d data/
```

解压后目录结构：

```
data/radar_dataset/
├── images/train/    (4000 张)
├── images/val/      (1000 张)
├── labels/train/    (4000 个 .txt)
└── labels/val/      (1000 个 .txt)
```

### 3.4 自行准备数据

如需标注新数据，按以下步骤：

1. 收集 5120×3840 雷达图片
2. 使用 labelme 或任何 YOLO 标注工具标注
3. 导出为 YOLO 格式 `.txt` 文件
4. 按 `train` / `val` 分类放入对应目录
5. class_id 与上表一致

---

## 四、本地训练

### 4.1 安装依赖（推荐 conda 或 venv）

```bash
pip install ultralytics opencv-python pyyaml
```

### 4.2 运行训练

```bash
# 默认执行全部三阶段（训练 + 自动导出 ONNX / OpenVINO）
python train.py
```

```bash
# 仅执行某个阶段
python train.py --config config_stage1.yaml
python train.py --config config_stage2.yaml
python train.py --config config_stage3.yaml
```

```bash
# 自定义多阶段顺序
python train.py --config config_stage1.yaml config_stage2.yaml
```

### 4.3 训练策略

| 阶段 | 参数文件 | epochs | imgsz | 学习率 | 说明 |
|------|---------|--------|-------|--------|------|
| 1 | config_stage1.yaml | 300 | 1280 | 0.01 | 高分辨率初始训练 |
| 2 | config_stage2.yaml | 500 | 640 | 0.001 | 低分辨率微调（自动加载 stage1 最优权重） |
| 3 | config_stage3.yaml | 700 | 1280 | 0.0005 | 高分辨率精调（自动加载 stage2 最优权重） |

- 硬件：8× L40（48GB），batch=128（每卡 16）
- 每阶段训练完成后自动导出 ONNX + OpenVINO（1280 和 640 双尺寸）

### 4.4 训练输出

```
runs/train/
├── stage1/
│   └── weights/
│       ├── best.pt         # 最优权重
│       ├── last.pt         # 最后 epoch 权重
│       ├── best.onnx       # 导出 ONNX (1280)
│       ├── best.onnx       # 导出 ONNX (640)
│       ├── best_openvino_model/
│       └── best_openvino_model/
├── stage2/
│   └── weights/...
└── stage3/
    └── weights/...
```

### 4.5 可视化

```bash
tensorboard --logdir runs/train
```

浏览器打开 `http://localhost:6006`，查看 loss、mAP、学习率等曲线。

---

## 五、Docker 训练

### 5.1 构建镜像

```bash
docker build -t radar-model .
```

- 镜像不含数据（约 8GB）
- 数据通过 volume 挂载

### 5.2 准备数据

把从网盘下载的 `RadarDataset.zip` 解压到项目目录：

```bash
unzip RadarDataset.zip -d data/
```

### 5.3 运行训练

```bash
docker compose up
```

等价于：

```bash
docker run --gpus all \
  -v ./data:/app/data \
  -v ./runs:/app/runs \
  radar-model
```

### 5.4 单阶段训练

```bash
docker compose run --rm radar-train python train.py --config config_stage1.yaml
```

### 5.5 TensorBoard

```bash
docker compose run --rm -p 6006:6006 radar-train tensorboard --logdir runs/train --bind_all
```

### 5.6 推送镜像到 GitHub Container Registry

```bash
# 构建
docker build -t ghcr.io/你的用户名/model_train:latest .

# 登录 GitHub
echo "你的TOKEN" | docker login ghcr.io -u 你的用户名 --password-stdin

# 推送
docker push ghcr.io/你的用户名/model_train:latest
```

---

## 六、模型导出

训练完成后自动导出。也可手动执行：

```python
from ultralytics import YOLO

model = YOLO("runs/train/stage3/weights/best.pt")

model.export(format="onnx",   half=True, imgsz=1280)
model.export(format="onnx",   half=True, imgsz=640)
model.export(format="openvino", half=True, imgsz=1280)
model.export(format="openvino", half=True, imgsz=640)
```

---

## 七、自定义训练参数

修改 `config_stage*.yaml` 中的参数：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 300/500/700 | 训练轮数 |
| `batch` | 128 | 总 batch size |
| `imgsz` | 1280/640 | 输入图片分辨率 |
| `lr0` | 0.01/0.001/0.0005 | 初始学习率 |
| `device` | 0,1,2,3,4,5,6,7 | GPU 列表 |
| `workers` | 32 | 数据加载线程 |
| `amp` | true | 混合精度训练 |
| `cos_lr` | false | 余弦学习率衰减（false=线性） |
