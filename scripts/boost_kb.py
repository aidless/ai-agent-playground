#!/usr/bin/env python3
"""
🛡️ 知识库扩容器 (最终修复版)
策略：每批 20 篇 | 间隔 5 秒 | 遇封锁自动休眠
修复：使用 requests 直连下载，彻底解决库版本冲突导致的报错
"""

import sys
import json
import time
import requests
from pathlib import Path

# === 配置区 ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "papers"
META_FILE = DATA_DIR / "metadata.json"

# 核心参数（严格符合你的要求）
BATCH_SIZE = 20  # 严格遵守单次 20 篇
REQUEST_INTERVAL = 5  # 严格间隔 5 秒

KEYWORDS = [
    "LLM autonomous agent",  # 通用智能体
    "ReAct prompting",  # ReAct 架构
    "Tool learning function call",  # 工具调用
    "Multi-agent collaboration",  # 多智能体协作
    "Memory augmented reasoning"  # 记忆推理
]


def load_existing_metadata():
    """加载已有的元数据，防止重复"""
    if META_FILE.exists():
        try:
            with open(META_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 读取元数据失败: {e}")
    return []


def save_metadata(meta_list):
    """保存元数据到 JSON"""
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    print(f"💾 已更新本地索引: {len(meta_list)} 条记录")


def download_via_requests(url, filepath):
    """
    核心修复：使用 requests 直接通过 URL 下载，绕过 arxiv 库的内部 bug
    """
    try:
        print(f"      💻 正在直连下载 PDF...")
        resp = requests.get(url, timeout=30, stream=True)
        resp.raise_for_status()  # 检查网络状态

        with open(filepath, 'wb') as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        return True
    except Exception as e:
        print(f"      ❌ 下载失败: {str(e)[:50]}")
        return False


def fetch_with_retry(query, max_results):
    """带重试机制的网络抓取"""
    try:
        import arxiv
    except ImportError:
        print("❌ 缺少 arxiv 库，请先运行 pip install arxiv")
        sys.exit(1)

    max_retries = 3
    delay_start = 30  # 遇到 429 自动等待 30 秒起跳

    for attempt in range(max_retries):
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            return list(client.results(search))

        except arxiv.HTTPError as e:
            status = e.args[0] if e.args else "?"
            if attempt < max_retries - 1:
                sleep_time = delay_start * (2 ** attempt)
                print(f"   ⚠️ 遭遇封禁 ({status})，系统休眠 {sleep_time // 60} 分钟后继续...")
                time.sleep(sleep_time)
            else:
                print(f"   ❌ 多次尝试均被拒，跳过此主题。")
                return []

    return []


def process_paper(paper, seen_ids, meta_list, count):
    """处理单篇论文：检查 ID -> 下载 PDF -> 记录信息"""
    pid = paper.entry_id.split('/')[-1]
    if pid in seen_ids:
        return count

    print(f"   📥 [{count + 1}] {paper.title[:60]}...")

    # --- 修复点：这里不再调用 paper.download_pdf，而是直接抓 URL ---
    if hasattr(paper, 'pdf_url'):
        pdf_path = DATA_DIR / f"{pid}.pdf"
        if not pdf_path.exists():
            success = download_via_requests(paper.pdf_url, pdf_path)
            if success:
                print(f"      ✅ 完成存储")
            else:
                print(f"      ⚠️ PDF 未保存，仅记录标题")
    else:
        print(f"      ⚠️ 未找到 PDF 地址")

    # 记录元数据
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
    return count + 1


def main():
    print("🛡️ 开始执行安全扩库任务...")
    print("   策略：每批 20 篇 | 间隔 5 秒 | 使用直连下载\n")

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    meta_list = load_existing_metadata()
    seen_ids = {p['id'] for p in meta_list}

    target_total = 100

    for i, kw in enumerate(KEYWORDS):
        if len(seen_ids) >= target_total:
            break

        needed = target_total - len(seen_ids)
        current_batch_size = min(BATCH_SIZE, needed)

        print(f"\n[{i + 1}] 🔍 搜索主题: '{kw}' (还需约 {needed} 篇)...")

        # 抓取数据
        papers = fetch_with_retry(query=f'cat:"cs.AI" AND ti:"{kw}"', max_results=current_batch_size)

        # 处理每一篇
        for paper in papers:
            if len(seen_ids) >= target_total:
                break
            process_paper(paper, seen_ids, meta_list, len(seen_ids))

        if papers:
            print(f"   💤 本批完成，系统休眠 5 秒...")
            time.sleep(REQUEST_INTERVAL)

    save_metadata(meta_list)
    print(f"\n🎉 任务结束！当前知识库总容量: {len(seen_ids)} 篇")


if __name__ == "__main__":
    main()