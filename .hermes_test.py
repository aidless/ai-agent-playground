"""Hermes 加载测试 —— 逐步骤写入文件"""
import sys, os, time, traceback

os.chdir(r"C:\Users\Administrator\Desktop\ai-agent-playground")
REPORT = ".hermes_test_result.txt"

def log(msg):
    print(msg)
    with open(REPORT, "w", encoding="utf-8") as f:
        f.write(msg + "\n")

try:
    import torch
    log(f"Step 1: PyTorch {torch.__version__} CUDA:{torch.cuda.is_available()}")
except Exception as e:
    log(f"FAIL at torch: {e}")
    sys.exit(1)

try:
    import transformers
    log(f"Step 2: transformers {transformers.__version__}")
except Exception as e:
    log(f"FAIL at transformers: {traceback.format_exc()}")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    t0 = time.time()
    log(f"Step 3: Downloading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B")
    log(f"Step 4: Tokenizer OK ({time.time()-t0:.1f}s)")
except Exception as e:
    log(f"FAIL at tokenizer: {traceback.format_exc()}")
    sys.exit(1)

try:
    from transformers import AutoModelForCausalLM
    t0 = time.time()
    log(f"Step 5: Downloading model (this can take 5-30 min)...")
    model = AutoModelForCausalLM.from_pretrained(
        "NousResearch/Hermes-3-Llama-3.1-8B",
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    log(f"Step 6: Model loaded ({time.time()-t0:.1f}s) params={model.num_parameters()/1e9:.1f}B")
except Exception as e:
    log(f"FAIL at model: {traceback.format_exc()}")
    sys.exit(1)

log("SUCCESS: Hermes 3 is ready to use!")
