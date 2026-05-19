#!/usr/bin/env python3
"""
🚀 知识库极速扩容器
目标：一次性精准下载 100 篇高质量 AI Agent 论文
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

# 初始化路径
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "papers"
META_FILE = DATA_DIR / "metadata.json"

# 已下载的去重 ID
existing_ids = set()
if META_FILE.exists():
    with open(META_FILE, 'r', encoding='utf-8') as f:
        meta_list = json.load(f)
        existing_ids = {item['id'] for item in meta_list}

print(f"✅ 当前知识库已有: {len(existing_ids)} 篇论文")
print("🚀 开始批量扩容（目标新增 100 篇）...\n")

try:
    import arxiv
except ImportError:
    print("❌ 请先安装库: pip install arxiv")
    sys.exit(1)


def safe_download(paper, target_dir):
    """安全下载 PDF"""
    try:
        filename = f"{paper.entry_id.split('/')[-1]}.pdf"
        filepath = target_dir / filename

        # 只有不存在时才下载
        if not filepath.exists():
            paper.download_pdf(dirpath=str(target_dir), filename=filename)
            return True
        else:
            return False  # 跳过已存在的
    except Exception as e:
        print(f"    ⚠️ 下载跳过 ({e})")
        return False


target_total = len(existing_ids) + 100
collected_count = 0

# 定义高命中率关键词列表
keywords = [
    "LLM autonomous agent",  # 通用智能体
    "ReAct prompting",  # ReAct 架构
    "Tool learning function call",  # 工具调用
    "Multi-agent collaboration",  # 多智能体协作
    "Memory augmented reasoning"  # 记忆推理
]

client = arxiv.Client()

for kw in keywords:
    if collected_count >= 100: break

    print(f"🔍 搜索: '{kw}' ...")

    # 每次搜 25 篇，凑齐 100 篇即可
    search = arxiv.Search(query=f"cat:\"cs.AI\" AND ti:\"{kw}\"", max_results=25, sort_by=arxiv.SortCriterion.Relevance)

    for paper in client.results(search):
        paper_id = paper.entry_id.split('/')[-1]

        if paper_id not in existing_ids:
            # 下载 PDF
            success = safe_download(paper, DATA_DIR)

            if success:
                collected_count += 1
                total_now = len(existing_ids) + collected_count

                # 记录元数据
                entry = {
                    "id": paper_id,
                    "title": paper.title,
                    "abstract": paper.summary[:200] + "...",  # 摘要存一部分
                    "authors": ", ".join([a.name for a in paper.authors]),
                    "published": paper.published.strftime("%Y-%m-%d"),
                    "source": "ArXiv"
                }

                # 追加写入
                with open(META_FILE, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(entry, ensure_ascii=False) + "\n")

                print(f"   ✅ [{total_now}/100+]: {paper.title[:50]}...")

        # 礼貌延迟，防止被封
        time.sleep(0.8)

print(f"\n🎉 任务完成！本次新增 {collected_count} 篇，总库容量已达 {len(existing_ids) + collected_count} 篇。")