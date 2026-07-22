# RADAR-MODEL

基于 Ultralytics YOLO26n 的雷达目标检测训练项目，12 类别（6 兵种 × red/blue）。

---

## 一、项目结构

```
RADAR-MODEL/
├── config_stage1.yaml       # 阶段1 训练参数
├── config_stage2.yaml       # 阶段2 训练参数
├── config_stage3.yaml       # 阶段3 训练参数
├── train.py                 # 训练入口
├── scripts/
│   ├── rename_pt_names.py   # 修正 .pt 中类别名元数据
│   └── final_export.py      # 最终导出 ONNX + OpenVINO
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── data/                    # 训练数据（百度网盘下载，不进 git）
├── model/                   # 本地权重（不进 git）
└── final_export/            # 最终交付导出（不进 git）
```

---

## 二、类别映射

| id | 类别 | id | 类别 |
|----|------|----|------|
| 0 | hero-red | 6 | hero-blue |
| 1 | engineer-red | 7 | engineer-blue |
| 2 | infantry3-red | 8 | infantry3-blue |
| 3 | infantry4-red | 9 | infantry4-blue |
| 4 | sentry-red | 10 | sentry-blue |
| 5 | drone-red | 11 | drone-blue |

---

## 三、数据准备

### 3.1 百度网盘

```bash
unzip RadarDataset.zip -d data/
```

最终：

```
data/radar_dataset/
├── images/train/
├── images/val/
├── labels/train/
└── labels/val/
```

### 3.2 标签格式

每行：`class_id x_center y_center width height`（归一化 0-1），`class_id` 按上表。

---

## 四、环境准备

- NVIDIA 驱动（支持 CUDA 12.4+）
- Python 3.10+

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

导出 OpenVINO 还需：

```bash
pip install openvino
```

---

## 五、本地训练

```bash
# 三阶段
python train.py

# 单阶段
python train.py --config config_stage1.yaml
```

训练产物在 `runs/train/stageN/weights/`。

| 阶段 | 配置 | epochs | imgsz | lr |
|------|------|--------|-------|-----|
| 1 | config_stage1.yaml | 300 | 1280 | 0.01 |
| 2 | config_stage2.yaml | 500 | 640 | 0.001 |
| 3 | config_stage3.yaml | 700 | 1280 | 0.0005 |

---

## 六、修正 .pt 类别名

若权重顺序正确但名称仍是红蓝交替，用脚本只改元数据名：

```bash
python scripts/rename_pt_names.py model/best.pt
# 输出: model/best_fixed_names.pt
```

会先打印 Old names，再保存后打印 New names。

---

## 七、最终导出（ONNX + OpenVINO）

统一导出到 `final_export/`，与训练中间产物分离：

```bash
python scripts/final_export.py model/best_fixed_names.pt
```

输出结构：

```
final_export/best_fixed_names/
├── best_fixed_names.pt
├── classes.txt
├── manifest.json
├── onnx/
│   ├── 1280/best_fixed_names_1280.onnx
│   └── 640/best_fixed_names_640.onnx
└── openvino/
    ├── 1280/best_fixed_names_1280_openvino_model/
    └── 640/best_fixed_names_640_openvino_model/
```

可选：

```bash
python scripts/final_export.py model/best_fixed_names.pt --output-dir final_export/release
python scripts/final_export.py model/best_fixed_names.pt --no-half   # FP32
```

---

## 八、Docker 训练（备选）

```bash
docker build -t radar-model .
docker compose up
```

数据放在 `./data`，输出在 `./runs`。

---

## 九、常见问题

**Q: 数据太大不能推 git？**  
A: `data/`、`model/`、`final_export/`、`runs/` 已在 `.gitignore` 中。

**Q: 显存不够？**  
A: 调低 `batch` 或 `imgsz`。

**Q: 导出失败缺少 openvino？**  
A: `pip install openvino`
