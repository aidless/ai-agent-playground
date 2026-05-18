"""Hermes 加载测试 —— 结果写入文件"""
import sys, os, time

os.chdir(r"C:\Users\Administrator\Desktop\ai-agent-playground")
report_path = r"C:\Users\Administrator\Desktop\ai-agent-playground\.hermes_test_result.txt"

lines = []
def log(msg):
    print(msg)
    lines.append(msg)

try:
    import torch
    log(f"PyTorch: {torch.__version__}")
    log(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log(f"GPU: {torch.cuda.get_device_name(0)}")
    log(f"RAM total: {round(torch.cuda.get_device_properties(0).total_mem / 1024**3, 1)}GB" if torch.cuda.is_available() else "Running on CPU")

    import transformers
    log(f"Transformers: {transformers.__version__}")

    log("Loading Hermes tokenizer...")
    t0 = time.time()
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")
    log(f"Tokenizer loaded in {time.time()-t0:.1f}s | vocab size: {len(tokenizer)}")

    log("DONE - 分词器加载成功！模型本体还没加载（~16GB 需要几分钟）")

except Exception as e:
    log(f"ERROR: {e}")

with open(report_path, "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
