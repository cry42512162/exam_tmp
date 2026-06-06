import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import argparse

# ========== 配置 ==========
MODEL_NAME = "../bert-base-uncased"  # 本地 BERT 模型路径（离线）
MAX_LENGTH = 256
MODEL_PATH = "./best_model.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ========== 加载模型和分词器 ==========
print("加载模型...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME, num_labels=2
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

def predict(text):
    """对单条文本进行情感预测"""
    # 分词
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",    # 返回 PyTorch 张量
    )
    # 移到 GPU
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推理
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    # 转概率（softmax）
    probs = torch.softmax(logits, dim=-1).squeeze()
    negative_prob, positive_prob = probs[0].item(), probs[1].item()

    # 判断
    pred_label = 1 if positive_prob > negative_prob else 0
    label_name = "Positive :-)" if pred_label == 1 else "Negative :-("
    confidence = max(negative_prob, positive_prob)

    return label_name, confidence, positive_prob

# ========== 交互式模式 ==========
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IMDB 情感分析推理")
    parser.add_argument("--text", type=str, help="要分析的评论文本")
    args = parser.parse_args()

    if args.text:
        # 命令行模式：从参数读入文本
        label, conf, pos_prob = predict(args.text)
        print(f"评论文本: {args.text[:100]}...")
        print(f"情感判断: {label}")
        print(f"置信度: {conf:.4f}")
    else:
        # 交互模式：循环等待用户输入
        print("=" * 50)
        print("IMDB 情感分析 — 交互推理模式")
        print("输入评论后按回车，输入 'quit' 退出")
        print("=" * 50)

        while True:
            text = input("\n请输入影评: ").strip()
            if text.lower() == "quit":
                print("再见！")
                break
            if not text:
                continue
            label, conf, pos_prob = predict(text)
            print(f"-> {label}  (正面概率: {pos_prob:.4f}, 置信度: {conf:.4f})")
