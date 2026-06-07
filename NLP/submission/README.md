# NLP — IMDB 情感分析

**所选方向：NLP（自然语言处理）**

**任务：IMDB 电影评论情感二分类（Positive / Negative）**

**模型路径：A（BERT 微调）**

---

## 环境配置

| 项目 | 版本 |
|------|------|
| Python | 3.10 |
| PyTorch | 2.12.0.dev20260408+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 Laptop (8 GB VRAM) |
| Transformers | 4.47.0 |
| datasets | 3.2.0 |

安装依赖：

```bash
pip install -r requirements.txt
```

---

## 项目结构

```
submission/
├── README.md                         # 本文件
├── requirements.txt                  # Python 依赖
├── IMDB_Sentiment_Analysis_Report.docx  # 实验报告（Word）
├── src/
│   ├── train.py                      # 训练脚本
│   ├── evaluate.py                   # 评估脚本（含错误分析）
│   └── inference.py                  # 推理脚本（交互式 + 命令行）
└── results/
    ├── loss_curve.png                # 训练/验证损失曲线
    └── confusion_matrix.png          # 测试集混淆矩阵
```

---

## 数据准备

IMDB 数据集（50,000 条影评）需提前下载，以 parquet 格式放置于 `NLP/Data/` 目录：

- `train-00000-of-00001.parquet`
- `test-00000-of-00001.parquet`
- `unsupervised-00000-of-00001.parquet`

BERT-base-uncased 模型文件放置于 `NLP/bert-base-uncased/` 目录。

> 注：数据和模型未包含在提交中（已在 .gitignore 中排除，因为文件过大，可从网盘：下载），运行前需自行准备。

---

## 运行命令

```bash
# 1. 训练（约 23 分钟，GPU）
cd NLP/src
python train.py

# 2. 评估（含错误分析）
python evaluate.py

# 3. 交互推理
python inference.py

# 或命令行模式：
python inference.py --text "This movie was fantastic!"
```

---

## 实验结果

| 指标 | 数值 |
|------|------|
| 测试集 Accuracy | 92.12% |
| 测试集 F1-Score | 92.11% |
| 验证集最佳 Acc | 91.40% |
| 训练耗时 | ~23 分钟 |
| 误分类率 | 7.88% (1,971/25,000) |

---

## 关键超参数

| 超参数 | 取值 |
|--------|------|
| MAX_LENGTH | 256 |
| BATCH_SIZE | 16 |
| LEARNING_RATE | 2×10⁻⁵ |
| NUM_EPOCHS | 3 |
| WEIGHT_DECAY | 0.01 |
| 优化器 | AdamW |

---
