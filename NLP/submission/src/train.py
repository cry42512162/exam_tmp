# ========== 导入库 ==========
import torch                          # PyTorch 深度学习框架
from torch import nn                  # 神经网络模块（各种层）
from torch.utils.data import DataLoader # 批量加载数据的工具
from transformers import (            # HuggingFace 核心库
    AutoTokenizer,                    # 自动分词器（把文字变成数字）
    AutoModelForSequenceClassification, # 用于文本分类的预训练模型
    get_linear_schedule_with_warmup,  # 学习率调度器
)
from datasets import load_dataset     # 一键加载 IMDB 数据集
from tqdm.auto import tqdm            # 漂亮的进度条
import matplotlib.pyplot as plt       # 绘图库（画损失曲线）
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块
import numpy as np                    # 数值计算
import os                             # 操作系统接口（创建目录等）


# ========== 超参数配置 ==========
# 超参数 = 你手动设定的、控制训练行为的参数（不是模型自己学的）

MODEL_NAME = "../bert-base-uncased"  # 本地 BERT 模型路径（离线）
MAX_LENGTH = 256                     # 每条评论最大保留多少词
BATCH_SIZE = 16                      # 每次喂给模型多少条评论
LEARNING_RATE = 2e-5                 # 学习率（控制每次更新步伐大小）
NUM_EPOCHS = 3                       # 训练轮数（完整遍历数据几次）
WARMUP_STEPS = 0                     # 预热步数（一开始不急着大步更新）
WEIGHT_DECAY = 0.01                  # 权重衰减（防止过拟合的正则化）
OUTPUT_DIR = "../results"            # 结果保存目录
MODEL_SAVE_PATH = "./best_model.pt"  # 最佳模型保存路径

# 创建输出目录（如果不存在）
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 检测 GPU（如果能用就用，不能用就用 CPU）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


# ========== 加载 IMDB 数据集 ==========
# 从本地 parquet 文件加载（无需网络下载）
# 三个文件需提前放在 ../Data/ 目录下
print("正在加载 IMDB 数据集（本地 parquet 文件）...")
data_dir = "../Data"
dataset = load_dataset("parquet", data_files={
    "train": f"{data_dir}/train-00000-of-00001.parquet",
    "test": f"{data_dir}/test-00000-of-00001.parquet",
    "unsupervised": f"{data_dir}/unsupervised-00000-of-00001.parquet",
})

# 查看数据结构
print(f"训练集大小: {len(dataset['train'])}")   # 应输出 25000
print(f"测试集大小: {len(dataset['test'])}")    # 应输出 25000
print(f"一条样本示例: {dataset['train'][0]}")   # {text: "...", label: 0}


# ========== 加载分词器（Tokenizer）==========
# 分词器把文字转成数字序列。例："I loved this movie!"
# → [101, 1045, 5709, 2023, 3185, 999, 102]
# 自动做：转小写、拆子词、补零/截断
print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


# ========== 数据预处理函数 ==========
def preprocess_function(examples):
    """
    把原始文本转成模型需要的张量（多维数字数组）

    输入: examples = {"text": ["评论1", "评论2", ...], "label": [0, 1, ...]}
    输出: {"input_ids": [...], "attention_mask": [...], "label": [...]}
    """
    return tokenizer(
        examples["text"],
        truncation=True,      # 超过 MAX_LENGTH 的部分直接截断
        padding="max_length", # 不足 MAX_LENGTH 的补零
        max_length=MAX_LENGTH,
    )

# 对整个数据集执行预处理（batch 处理，速度快）
print("正在预处理数据（分词）...")
encoded_dataset = dataset.map(preprocess_function, batched=True)

# 移除不需要的列，只保留模型需要的 input_ids, attention_mask, label
encoded_dataset = encoded_dataset.remove_columns(["text"])
encoded_dataset.set_format("torch")  # 转为 PyTorch 张量格式


# ========== 创建数据加载器 ==========
# 训练集 90% 用于训练，10% 作为验证集（监控过拟合）
train_valid = encoded_dataset["train"].train_test_split(test_size=0.1)
train_dataset = train_valid["train"]    # 22500 条
val_dataset = train_valid["test"]       # 2500 条
test_dataset = encoded_dataset["test"]  # 25000 条

# DataLoader：将数据按 BATCH_SIZE 分组，每次取一小批
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

print(f"训练集批次数: {len(train_loader)}")   # ≈ 22500/16 ≈ 1406 批
print(f"验证集批次数: {len(val_loader)}")     # ≈ 2500/16  ≈ 157 批


# ========== 加载预训练模型 ==========
# BERT 本体 + 分类头（768维 → 2类：正面/负面）
print(f"正在加载模型 {MODEL_NAME}...")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,  # 二分类：正面(1) vs 负面(0)
)
model.to(device)  # 把模型移到 GPU（如果有的话）
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
# 应输出约 110,000,000（1.1 亿参数）


# ========== 优化器 ==========
# AdamW：自适应调整学习率 + 权重衰减防止过拟合
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,       # 学习率
    weight_decay=WEIGHT_DECAY,  # 权重衰减
)

# ========== 学习率调度器（可选但推荐）==========
# 学习率从 2e-5 线性衰减到 0，让训练后期更稳定
total_steps = len(train_loader) * NUM_EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=WARMUP_STEPS,
    num_training_steps=total_steps,
)


# ========== 训练函数 ==========
def train_epoch(model, dataloader, optimizer, scheduler, device):
    """训练一个 Epoch"""
    model.train()  # 切换到训练模式（启用 Dropout 等训练专用机制）
    total_loss = 0

    progress_bar = tqdm(dataloader, desc="训练中")
    for batch in progress_bar:
        # 第一步：把数据移到 GPU
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        # 第二步：清空上次计算的梯度
        # （PyTorch 默认累积梯度，不清空会叠加）
        optimizer.zero_grad()

        # 第三步：前向传播 —— 模型"预测"
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss  # 模型自动计算了交叉熵损失

        # 第四步：反向传播 —— 计算梯度（每个参数对 loss 的贡献）
        loss.backward()

        # 第五步：参数更新 —— 根据梯度调整参数
        optimizer.step()

        # 第六步：更新学习率（线性衰减）
        scheduler.step()

        total_loss += loss.item()
        progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    return total_loss / len(dataloader)  # 返回平均损失


# ========== 验证函数 ==========
def evaluate(model, dataloader, device):
    """在验证集/测试集上评估模型"""
    model.eval()  # 切换到评估模式（关闭 Dropout 等）
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():  # 告诉 PyTorch：不要记录梯度！
        # ↑ 这行极其重要，能大幅减少显存占用和加速推理
        for batch in tqdm(dataloader, desc="评估中"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            loss = outputs.loss
            logits = outputs.logits  # 原始输出：[正面分, 负面分]

            total_loss += loss.item()
            predictions = torch.argmax(logits, dim=-1)  # 取分数高的
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return avg_loss, accuracy


# ========== 主训练循环 ==========
print("开始训练！")
train_losses, val_losses = [], []  # 记录每个 epoch 的 loss

best_val_acc = 0  # 记录最佳验证准确率，用于保存最佳模型

for epoch in range(NUM_EPOCHS):
    print(f"\n===== Epoch {epoch + 1}/{NUM_EPOCHS} =====")

    # 训练一个 epoch
    train_loss = train_epoch(model, train_loader, optimizer, scheduler, device)
    train_losses.append(train_loss)

    # 验证
    val_loss, val_acc = evaluate(model, val_loader, device)
    val_losses.append(val_loss)

    print(f"训练 Loss: {train_loss:.4f} | 验证 Loss: {val_loss:.4f} | 验证 Acc: {val_acc:.4f}")

    # 保存最佳模型（只有验证准确率超过历史最好时才保存）
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        print(f"→ 新最佳模型已保存！(Acc: {val_acc:.4f})")

print(f"\n训练完成！最佳验证准确率: {best_val_acc:.4f}")


# ========== 绘制 Loss 曲线 ==========
plt.figure(figsize=(10, 5))
plt.plot(range(1, NUM_EPOCHS + 1), train_losses, "b-o", label="训练 Loss")
plt.plot(range(1, NUM_EPOCHS + 1), val_losses, "r-o", label="验证 Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("训练 / 验证损失曲线")
plt.legend()
plt.grid(True)
plt.savefig(f"{OUTPUT_DIR}/loss_curve.png", dpi=150, bbox_inches="tight")
print(f"Loss 曲线已保存到 {OUTPUT_DIR}/loss_curve.png")
