"""Daily task feeder — sends real-world tasks to /chat/completions and logs results.

Usage: uv run python scripts/daily_tasks.py [--count 20]
Output: task_log.jsonl (appends each day)

Each task simulates what a real user would ask a design/engineering AI.
Results feed into CostTracker + CLEAR + SLO metrics.
"""

import asyncio
import json
import sys
import time
import httpx

BASE = "http://127.0.0.1:8000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 20

# Real-world tasks from design/engineering work
TASKS = [
    "商业建筑内歌舞娱乐场所的防火分隔要求有哪些？请引用规范条文。",
    "高层建筑避难层的设置要求是什么？面积、间距、设施有哪些规定？",
    "防火门的耐火等级如何划分？甲级、乙级、丙级防火门分别适用于什么部位？",
    "建筑消防给水系统中，高位消防水箱的有效容积如何计算？",
    "地下汽车库的防火分区最大允许面积是多少？设置自动灭火系统后可以扩大多少？",
    "建筑内疏散走道的净宽度有什么要求？医院、学校、商业建筑的疏散宽度如何计算？",
    "建筑防烟楼梯间和前室的设计要求有哪些？什么情况下需要设置防烟楼梯间？",
    "建筑防火间距的一般规定是什么？多层与高层、高层与高层之间分别多少米？",
    "自动喷水灭火系统的设置场所和设计参数有哪些？",
    "建筑内装修材料的燃烧性能等级如何划分？不同场所对顶棚、墙面、地面的要求是什么？",
    "消防电梯的设置条件和技术要求有哪些？",
    "建筑内防火阀的设置位置和要求是什么？",
    "建筑燃气管道穿越防火墙、楼板时应采取什么防火措施？",
    "建筑消防救援窗的设置要求是什么？间距、尺寸、标识有什么规定？",
    "建筑消防控制室的设置位置和面积有什么要求？",
    "建筑内消防水泵房的设置要求有哪些？防火分隔、安全出口的规定是什么？",
    "建筑消防车道设置标准是什么？宽度、净空、转弯半径、承载力的要求？",
    "建筑中庭的防火设计要求是什么？防火卷帘、排烟、疏散的具体规定？",
    "建筑消防供配电系统的设计要求是什么？消防负荷分级和供电方式如何确定？",
    "建筑内消防应急照明和疏散指示标志的设置标准是什么？",
    "Cite the fire safety requirements for high-rise residential buildings in China",
    "What are the key differences between GB 50016-2014 and the 2023 General Fire Code?",
    "Explain the fire compartmentation requirements for mixed-use commercial buildings",
    "How should smoke control systems be designed for atriums per Chinese building code?",
    "Summarize the fire hydrant and sprinkler system requirements for underground parking",
]

print(f"Sending {N} tasks...")
ok = 0
fail = 0

for i, task in enumerate(TASKS[:N]):
    try:
        start = time.perf_counter()
        r = httpx.post(
            f"{BASE}/chat/completions",
            json={
                "messages": [{"role": "user", "content": task}],
                "stream": False,
            },
            timeout=180,
        )
        elapsed = (time.perf_counter() - start) * 1000
        success = r.status_code == 200

        if success:
            ok += 1
        else:
            fail += 1

        # Log
        with open("task_log.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "task": task[:100],
                "status": r.status_code,
                "latency_ms": round(elapsed),
                "success": success,
            }, ensure_ascii=False) + "\n")

        bar = "=" * int((i+1)/N*30)
        empty = " " * (30 - len(bar))
        print(f"\r[{bar}{empty}] {i+1}/{N} ok={ok} fail={fail} last={elapsed:.0f}ms", end="")

    except Exception as e:
        fail += 1
        print(f"\r[{i+1}/{N}] ERROR: {e}")

print(f"\nDone: {ok}/{N} OK ({ok/N*100:.0f}%)")
print("Log: task_log.jsonl")
print(f"View metrics: {BASE}/clear")
