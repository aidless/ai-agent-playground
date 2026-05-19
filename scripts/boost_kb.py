#!/usr/bin/env python3
"""
🛡️ 知识库扩容器（生产级鲁棒版）
功能：安全、稳定地收集 AI Agent 论文
特性：
  - MaxResults=20 (防封禁策略)
  - 指数退避重试 (自动应对 HTTP 429)
  - 5秒硬性间隔 (尊重服务器负载)
"""

import sys
import os
import json
import time
import random
from pathlib import Path

# === 配置区 ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "papers"
META_FILE = DATA_DIR / "metadata.json"

# 核心配置：严格遵守 ArXiv 限制
BATCH_SIZE = 20  # 单批次请求数量 (最大推荐值)
REQUEST_INTERVAL = 5  # 两次请求间的休眠时间 (秒)

# 关键词列表
KEYWORDS = [
    "LLM autonomous agent",  # 通用智能体
    "ReAct prompting",  # ReAct 架构
    "Tool learning function call",  # 工具调用
    "Multi-agent collaboration",  # 多智能体协作
    "Memory augmented reasoning"  # 记忆推理
]


def load_existing_metadata():
    """加载已有的元数据，避免重复处理"""
    if META_FILE.exists():
        try:
            with open(META_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 读取元数据失败: {e}，将创建新文件")
    return []


def save_metadata(meta_list):
    """保存元数据到 JSON"""
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    print(f"💾 已更新元数据: {META_FILE}")


def download_and_process(paper, seen_ids, meta_list, collected_count):
    """处理单篇论文：下载 PDF 并记录信息"""
    pid = paper.entry_id.split('/')[-1]

    if pid in seen_ids:
        return collected_count  # 跳过已存在的

    print(f"   📥 正在获取: {paper.title[:60]}...")

    # 1. 尝试下载 PDF
    pdf_filename = f"{pid}.pdf"
    pdf_path = DATA_DIR / pdf_filename

    downloaded = False
    if not pdf_path.exists():
        try:
            paper.download_pdf(dirpath=str(DATA_DIR), filename=pdf_filename)
            print(f"      ✅ 成功下载 PDF: {pdf_filename}")
            downloaded = True
        except Exception as e:
            print(f"      ❌ PDF 下载失败 (仅记录摘要): {e}")

    # 2. 记录元数据
    entry = {
        "id": pid,
        "title": paper.title,
        "abstract": paper.summary.replace('\n', ' '),
        "authors": ", ".join([a.name for a in paper.authors]),
        "published": paper.published.strftime("%Y-%m-%d"),
        "source": "ArXiv"
    }
    meta_list.append(entry)
    seen_ids.add(pid)
    collected_count += 1

    return collected_count


def fetch_with_retry(query, max_results):
    """
    带指数退避的重试抓取函数
    专门解决 HTTP 429 (Too Many Requests) 问题
    """
    import arxiv

    max_retries = 5
    delay_start = 30  # 初始等待 30 秒

    for attempt in range(max_retries):
        try:
            print(f"      🔍 (Attempt {attempt + 1}/{max_retries}) 正在从 ArXiv 获取...")
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            # 执行网络请求 (这里可能会抛出 HTTPError)
            return list(client.results(search))

        except arxiv.HTTPError as e:
            status_code = e.args[0] if e.args else "?"
            print(f"      ⚠️ 遭遇封锁 (HTTP {status_code})")

            if attempt < max_retries - 1:
                # 指数退避逻辑：30s -> 60s -> 120s ...
                sleep_time = delay_start * (2 ** attempt)
                print(f"      💤 系统将休眠 {sleep_time // 60} 分钟以解除封锁...")
                time.sleep(sleep_time)
            else:
                print(f"      ❌ 经过多次重试仍失败，跳过该批次。")
                return []

        except Exception as e:
            print(f"      ⚠️ 其他错误: {e}")
            time.sleep(5)

    return []


def main():
    print("🛡️ 开始执行安全扩库任务...")
    print("   策略: 每批 20 篇 | 间隔 5 秒 | 遇封锁自动休眠\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    meta_list = load_existing_metadata()
    seen_ids = {p['id'] for p in meta_list}

    target_total = 100

    for i, kw in enumerate(KEYWORDS):
        if len(seen_ids) >= target_total:
            break

        # 计算本轮需求
        needed = target_total - len(seen_ids)
        current_batch_size = min(BATCH_SIZE, needed)

        print(f"\n[{i + 1}] 🧠 搜索主题: '{kw}' (目标: 还需要约 {needed} 篇)")

        # 使用带重试的抓取函数
        papers = fetch_with_retry(
            query=f'cat:"cs.AI" AND ti:"{kw}"',
            max_results=current_batch_size
        )

        # 遍历结果
        for paper in papers:
            if len(seen_ids) >= target_total:
                break
            downloaded_and_process(paper, seen_ids, meta_list, len(seen_ids))

        # 核心防御：请求间隔 5 秒
        if papers:
            print(f"   💤 请求完成，系统休眠 5 秒以保护服务器...")
            time.sleep(REQUEST_INTERVAL)

    # 最终保存
    save_metadata(meta_list)
    print(f"\n🏁 任务结束！")
    print(f"   当前总库容量: {len(seen_ids)} 篇")


if __name__ == "__main__":
    main()