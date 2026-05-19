#!/usr/bin/env python3
"""
🛡️ 知识库扩容器（修正版）
特性：每批 20 篇 | 间隔 5 秒 | 遇封锁自动休眠
"""

import sys
import json
import time
from pathlib import Path

# === 配置区 ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "papers"
META_FILE = DATA_DIR / "metadata.json"

# ArXiv 限制设置
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
    """加载已有的元数据"""
    if META_FILE.exists():
        try:
            with open(META_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ 读取失败: {e}")
    return []


def save_metadata(meta_list):
    """保存元数据"""
    with open(META_FILE, 'w', encoding='utf-8') as f:
        json.dump(meta_list, f, ensure_ascii=False, indent=2)
    print(f"💾 已保存元数据: {len(meta_list)} 条记录")


def fetch_with_retry(query, max_results):
    """带重试的抓取函数"""
    try:
        import arxiv
    except ImportError:
        print("❌ 缺少 arxiv 库")
        sys.exit(1)

    max_retries = 5
    delay_start = 30  # 初始等待 30 秒

    for attempt in range(max_retries):
        try:
            client = arxiv.Client()
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            # 执行获取
            results = list(client.results(search))
            return results

        except arxiv.HTTPError as e:
            # 遇到 429 封锁
            status = e.args[0] if e.args else "?"
            if attempt < max_retries - 1:
                sleep_time = delay_start * (2 ** attempt)
                print(f"   ⚠️ 遭遇封锁 ({status})，休眠 {sleep_time // 60} 分钟自动重试...")
                time.sleep(sleep_time)
            else:
                print(f"   ❌ 重试多次后仍失败。")
                return []

        except Exception as e:
            print(f"   ⚠️ 其他错误: {e}")
            time.sleep(5)

    return []


def download_and_process(paper, seen_ids, meta_list, collected_count):
    """处理单篇论文"""
    pid = paper.entry_id.split('/')[-1]
    if pid in seen_ids:
        return collected_count

    print(f"   📥 [{len(seen_ids) + 1}] {paper.title[:60]}...")

    # 下载 PDF
    pdf_filename = f"{pid}.pdf"
    if not (DATA_DIR / pdf_filename).exists():
        try:
            paper.download_pdf(dirpath=str(DATA_DIR), filename=pdf_filename)
            print(f"      ✅ PDF 下载成功")
        except Exception as e:
            print(f"      ⚠️ PDF 下载失败: {e}")

    # 记录信息
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
    return collected_count + 1


def main():
    print("🛡️ 开始执行安全扩库任务...")
    print(f"   策略: 每批 20 篇 | 间隔 5 秒 | 遇封锁自动休眠\n")

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

        # 使用带重试逻辑的抓取
        papers = fetch_with_retry(
            query=f'cat:"cs.AI" AND ti:"{kw}"',
            max_results=current_batch_size
        )

        # 遍历处理
        for paper in papers:
            if len(seen_ids) >= target_total:
                break
            # --- 修复了函数名拼写错误 ---
            download_and_process(paper, seen_ids, meta_list, len(seen_ids))
        # -----------------------------------

        if papers:
            print(f"   💤 本批完成，系统休眠 {REQUEST_INTERVAL} 秒...")
            time.sleep(REQUEST_INTERVAL)

    save_metadata(meta_list)
    print(f"\n🎉 任务结束！最终库容: {len(seen_ids)} 篇")


if __name__ == "__main__":
    main()