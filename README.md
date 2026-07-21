# RADAR-MODEL

基于 Ultralytics YOLO26n 的雷达目标检测训练项目，12 类别（6 兵种 × red/blue）。

---

## 一、项目结构

```
RADAR-MODEL/                        # 项目根目录
├── config_stage1.yaml              # 阶段1 训练参数
├── config_stage2.yaml              # 阶段2 训练参数
├── config_stage3.yaml              # 阶段3 训练参数
├── train.py                        # 训练脚本
├── Dockerfile
├── docker-compose.yml
├── .dockerignore
├── README.md
└── data/                           # ← 数据放这里
    └── radar_dataset/
        ├── images/train/           # 训练图片
        ├── images/val/             # 验证图片
        ├── labels/train/           # 训练标签
        └── labels/val/             # 验证标签
```

---

## 二、类别映射

| id | 类别 | id | 类别 |
|----|------|----|------|
| 0 | hero-red | 1 | hero-blue |
| 2 | engineer-red | 3 | engineer-blue |
| 4 | infantry3-red | 5 | infantry3-blue |
| 6 | infantry4-red | 7 | infantry4-blue |
| 8 | sentry-red | 9 | sentry-blue |
| 10 | drone-red | 11 | drone-blue |

---

## 三、数据准备

### 3.1 从百度网盘获取

下载 `RadarDataset.zip`，解压到项目目录：

```bash
# 在 RADAR-MODEL/ 目录下执行
unzip RadarDataset.zip -d data/
```

最终目录结构：

```
data/radar_dataset/
├── images/train/    (4000 张 .jpg)
├── images/val/      (1000 张 .jpg)
├── labels/train/    (4000 个 .txt)
└── labels/val/      (1000 个 .txt)
```

### 3.2 自行标注

如需标注新数据：
1. 收集 5120×3840 雷达图片
2. 使用 labelme 等工具标注
3. 导出 YOLO 格式 `.txt`（每行 `class_id x_center y_center width height`，归一化 0-1）
4. 按 train/val 分类，放入 `data/radar_dataset/` 对应目录
5. class_id 按上表

---

## 四、环境准备

要求：
- NVIDIA 驱动 ≥ 550（支持 CUDA 12.4）
- Python 3.10+

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install ultralytics
```

## 五、本地训练

```bash
# 全部三阶段（默认）
python train.py

# 单阶段
python train.py --config config_stage1.yaml

# 可视化
tensorboard --logdir runs/train
```

## 六、Docker 训练（备选）

### 4.1 构建镜像

```bash
# 在项目根目录 RADAR-MODEL/ 下执行
docker build -t radar-model .
```

> 镜像约 8GB，不包含数据。

### 4.2 运行训练

**完整三阶段（默认）：**

```bash
docker compose up
```

**单阶段：**

```bash
docker compose run --rm radar-train python train.py --config config_stage1.yaml
```

**自定义多阶段：**

```bash
docker compose run --rm radar-train python train.py \
  --config config_stage1.yaml config_stage2.yaml config_stage3.yaml
```

### 4.3 docker-compose.yml 说明

```yaml
services:
  radar-train:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./data:/app/data        # 挂载数据
      - ./runs:/app/runs        # 输出持久化
    command: python train.py
```

> 容器内的 `/app/data/radar_dataset/` 对应你主机上的 `./data/radar_dataset/`。

### 4.4 TensorBoard 可视化

```bash
docker compose run --rm -p 6006:6006 radar-train \
  tensorboard --logdir runs/train --bind_all
```

浏览器打开 `http://localhost:6006`。

---

## 七、训练策略

| 阶段 | 参数文件 | epochs | imgsz | lr | 说明 |
|------|---------|--------|-------|-----|------|
| 1 | config_stage1.yaml | 300 | 1280 | 0.01 | 高分辨率初始训练 |
| 2 | config_stage2.yaml | 500 | 640 | 0.001 | 低分辨率微调（加载 stage1 最优权重） |
| 3 | config_stage3.yaml | 700 | 1280 | 0.0005 | 高分辨率精调（加载 stage2 最优权重） |

硬件：8× L40（48GB），batch=128（每卡 16），AMP 混合精度。

每阶段训练完成后自动导出 ONNX + OpenVINO（imgsz=1280 和 imgsz=640 各一份）。

### 训练输出

```
runs/train/
├── stage1/
│   └── weights/
│       ├── best.pt
│       ├── last.pt
│       ├── best.onnx
│       └── best_openvino_model/
├── stage2/
│   └── weights/...
└── stage3/
    └── weights/...
```

---

## 八、自定义配置

修改 `config_stage*.yaml`：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `epochs` | 300/500/700 | 训练轮数 |
| `batch` | 128 | 总 batch size |
| `imgsz` | 1280 / 640 | 输入分辨率 |
| `lr0` | 0.01 / 0.001 / 0.0005 | 初始学习率 |
| `device` | 0,1,2,3,4,5,6,7 | GPU 列表 |
| `workers` | 32 | 数据加载线程 |
| `amp` | true | 混合精度 |

---

## 九、常见问题

**Q: `data/` 目录不存在？**
A: 手动创建：`mkdir -p data/radar_dataset/images/{train,val} data/radar_dataset/labels/{train,val}`

**Q: 解压后路径不对怎么办？**
A: 确保 `data/radar_dataset/` 下面直接是 `images/` 和 `labels/`，没有多余嵌套。

**Q: 显存不够？**
A: 调低 `batch` 或 `imgsz`。例如 `batch: 64`、`imgsz: 640`。
