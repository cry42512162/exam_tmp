import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号 '-' 显示为方块
import seaborn as sns
import numpy as np
from tqdm.auto import tqdm
import os

# ========== 配置 ==========
MODEL_NAME = "../bert-base-uncased"  # 本地 BERT 模型路径（离线）
MAX_LENGTH = 256
BATCH_SIZE = 16
MODEL_PATH = "./best_model.pt"    # 加载训练好的模型
OUTPUT_DIR = "../results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 加载测试数据 ==========
# 从本地 parquet 文件加载（与 train.py 一致）
print("加载数据和模型...")
data_dir = "../Data"
dataset = load_dataset("parquet", data_files={
    "train": f"{data_dir}/train-00000-of-00001.parquet",
    "test": f"{data_dir}/test-00000-of-00001.parquet",
    "unsupervised": f"{data_dir}/unsupervised-00000-of-00001.parquet",
})
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    return tokenizer(examples["text"], truncation=True,
                     padding="max_length", max_length=MAX_LENGTH)

test_dataset = dataset["test"].map(preprocess, batched=True)
test_dataset = test_dataset.remove_columns(["text"])
test_dataset.set_format("torch")
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# ========== 加载模型 ==========
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ========== 评估 ==========
print("开始评估...")
all_preds, all_labels = [], []

with torch.no_grad():
    for batch in tqdm(test_loader, desc="评估中"):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"]

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=-1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.numpy())

# 计算指标
acc = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds)

print(f"\n{'='*40}")
print(f"测试集准确率 (Accuracy): {acc:.4f} ({acc*100:.2f}%)")
print(f"测试集 F1 分数 (F1-Score): {f1:.4f}")
print(f"{'='*40}")

# ========== 混淆矩阵 ==========
cm = confusion_matrix(all_labels, all_preds)
# cm 是一个 2×2 矩阵：
#   [[真负例数, 假正例数],
#    [假负例数, 真正例数]]

plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"])
plt.xlabel("预测标签")
plt.ylabel("真实标签")
plt.title(f"混淆矩阵 (Accuracy={acc:.4f})")
plt.savefig(f"{OUTPUT_DIR}/confusion_matrix.png", dpi=150, bbox_inches="tight")
print(f"混淆矩阵已保存到 {OUTPUT_DIR}/confusion_matrix.png")

# ========== 错误分析：找出误分类样本 ==========
# 注意：dataset["test"] 保留了原始 text 列，可以通过索引获取原文
errors = []
for i, (true_label, pred_label) in enumerate(zip(all_labels, all_preds)):
    if true_label != pred_label:
        errors.append((i, true_label, pred_label))

print(f"\n{'='*40}")
print(f"误分类样本数: {len(errors)} / {len(all_labels)} ({len(errors)/len(all_labels)*100:.2f}%)")
print(f"{'='*40}")

# 显示前 10 个误分类案例（便于报告中分析）
print("\n前 10 个误分类案例:\n")
for idx, (i, true_label, pred_label) in enumerate(errors[:10]):
    # dataset["test"] 仍保留原始 text（未被 remove_columns 影响）
    text = dataset["test"][i]["text"]
    # 清理文本中的 HTML 标签便于阅读
    clean_text = text.replace("<br />", " ").replace("<br/>", " ")[:200]
    print(f"--- 案例 {idx+1} ---")
    print(f"  真实标签: {'正面(Positive)' if true_label else '负面(Negative)'}")
    print(f"  预测标签: {'正面(Positive)' if pred_label else '负面(Negative)'}")
    print(f"  评论文本: {clean_text}...")
    print()

# 统计错误类型
false_positives = sum(1 for _, true, pred in errors if true == 0 and pred == 1)  # 负面误判为正面
false_negatives = sum(1 for _, true, pred in errors if true == 1 and pred == 0)  # 正面误判为负面
print(f"{'='*40}")
print(f"错误类型统计:")
print(f"  假正面 (负面→正面): {false_positives} 条")
print(f"  假负面 (正面→负面): {false_negatives} 条")
print(f"{'='*40}")
