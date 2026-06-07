# NLP — IMDB 情感分析

**所选方向：NLP（自然语言处理）**

**任务：IMDB 电影评论情感二分类（Positive / Negative）**

**模型：BERT-base-uncased 微调（路径 A），约 1.1 亿参数**

---

## 环境配置

| 项目 | 版本 / 说明 |
|------|------------|
| Python | 3.10 |
| PyTorch | 2.12.0.dev20260408+cu128 |
| CUDA | 12.8 |
| GPU | NVIDIA GeForce RTX 5060 Laptop (8 GB VRAM) |
| Transformers | 4.47.0 |
| datasets | 3.2.0 |
| scikit-learn | 1.7.2 |
| matplotlib / seaborn | 3.10.9 / 0.13.2 |

安装依赖：

```bash
pip install -r requirements.txt
```

> 训练也可在 CPU 上运行（将 `device` 自动切换为 CPU），耗时约 2-3 小时。

---

## 项目结构

```
submission/
├── README.md                              # 本文件
├── requirements.txt                       # Python 依赖列表
├── IMDB_Sentiment_Analysis_Report.docx    # 实验报告（含文献综述 + 实验分析）
├── src/
│   ├── train.py                           # 训练脚本
│   ├── evaluate.py                        # 评估脚本（含混淆矩阵 + 错误分析）
│   └── inference.py                       # 推理脚本（支持交互式 / 命令行两种模式）
└── results/
    ├── loss_curve.png                     # 训练 / 验证损失曲线
    └── confusion_matrix.png               # 测试集混淆矩阵
```

---

## 数据准备

### IMDB 数据集（3 个 parquet 文件）

放置于 `NLP/Data/` 目录：

- `train-00000-of-00001.parquet`（25,000 条训练评论）
- `test-00000-of-00001.parquet`（25,000 条测试评论）
- `unsupervised-00000-of-00001.parquet`（50,000 条无监督评论，本实验未使用）

### BERT-base-uncased 模型文件（4 个文件）

放置于 `NLP/bert-base-uncased/` 目录：

- `config.json`
- `model.safetensors`（约 440 MB）
- `tokenizer_config.json`
- `vocab.txt`

> 注：数据和模型文件较大，未包含在 Git 提交中（已在 .gitignore 中排除）。可从网盘：________ 下载，解压后按上述目录结构放置即可。

---

## 运行命令

所有命令在 `NLP/src/` 目录下执行：

```bash
cd NLP/src

# 第一步：训练（约 23 分钟，GPU，RTX 5060）
python train.py
#   - 自动从 ../Data/ 加载本地 IMDB 数据
#   - 自动从 ../bert-base-uncased/ 加载本地 BERT 模型
#   - 训练 3 个 Epoch，自动保存最佳模型到 ./best_model.pt
#   - 生成 loss_curve.png 到 ../results/

# 第二步：评估
python evaluate.py
#   - 加载训练好的 best_model.pt
#   - 在 25,000 条测试集上计算 Accuracy / F1-Score
#   - 输出前 10 个误分类案例及错误类型统计
#   - 生成 confusion_matrix.png 到 ../results/

# 第三步：交互推理
python inference.py
#   - 输入任意英文影评，实时返回情感判断
#   - 输入 quit 退出

# 或者命令行模式：
python inference.py --text "This movie was absolutely brilliant!"
```

---

## 实验结果

### 训练过程

| Epoch | 训练 Loss | 验证 Loss | 验证 Acc |
|-------|----------|----------|---------|
| 1 | 0.2667 | 0.2209 | 91.00% |
| 2 | 0.1319 | 0.2391 | 91.36% |
| 3 | 0.0511 | 0.2947 | 91.40% |

### 最终指标

| 指标 | 数值 |
|------|------|
| 测试集 Accuracy | **92.12%** |
| 测试集 F1-Score | **92.11%** |
| 训练耗时 | ~23 分钟（GPU） |
| 误分类率 | 7.88%（1,971 / 25,000） |

### 关键超参数

| 超参数 | 取值 | 说明 |
|--------|------|------|
| MAX_LENGTH | 256 | 覆盖大部分 IMDB 评论（均值 ~230 词） |
| BATCH_SIZE | 16 | 8 GB 显存安全值 |
| LEARNING_RATE | 2×10⁻⁵ | BERT 微调黄金学习率 |
| NUM_EPOCHS | 3 | 超过 3 轮验证 Loss 上升（过拟合） |
| WEIGHT_DECAY | 0.01 | AdamW 正则化 |
| 优化器 | AdamW | 自适应学习率 + 权重衰减 |
