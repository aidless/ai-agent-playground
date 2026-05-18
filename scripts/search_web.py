"""Web search & fetch tool — 供 Claude Code 自主调用

在中国网络环境下可用：
  - 搜索：Bing (cn.bing.com) ✅ 已验证
  - 抓取：httpx GET 任意 URL

Usage:
    uv run python scripts/search_web.py search <query> [--max-results N]
    uv run python scripts/search_web.py fetch <url> [--max-chars N]

Output: JSON to stdout (单行，方便解析)
"""

import json
import re
import sys
import argparse

import httpx

SEARCH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
}


def search_bing(query: str, max_results: int = 5) -> list[dict]:
    """Search Bing (cn.bing.com) — 在中国可访问"""
    url = "https://cn.bing.com/search"
    params = {"q": query, "count": max_results}

    try:
        r = httpx.get(url, params=params, headers=SEARCH_HEADERS, timeout=15.0, follow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        return [{"error": f"Bing request failed: {e}"}]

    results = []
    html = r.text

    # Try BeautifulSoup first (better parsing)
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for li in soup.select("li.b_algo"):
            title_el = li.select_one("h2 a")
            snippet_el = li.select_one(".b_caption p")
            if title_el:
                results.append({
                    "title": title_el.get_text(strip=True),
                    "url": title_el.get("href", ""),
                    "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
                })
                if len(results) >= max_results:
                    break
    except ImportError:
        pass

    # Fallback: regex-based extraction
    if not results:
        # Extract from JSON-LD or inline results
        for match in re.finditer(
            r'<h2><a[^>]*href="([^"]*)"[^>]*>(.*?)</a></h2>.*?<p>(.*?)</p>',
            html, re.DOTALL
        ):
            results.append({
                "title": re.sub(r'<[^>]+>', '', match.group(2)).strip(),
                "url": match.group(1),
                "snippet": re.sub(r'<[^>]+>', '', match.group(3)).strip(),
            })
            if len(results) >= max_results:
                break

    # Last resort: extract from <a> tags with cite
    if not results:
        for match in re.finditer(
            r'<cite[^>]*>(.*?)</cite>.*?<a[^>]*href="([^"]*)"[^>]*>(.*?)</a>',
            html, re.DOTALL
        ):
            results.append({
                "title": re.sub(r'<[^>]+>', '', match.group(3)).strip(),
                "url": match.group(2),
                "snippet": "",
            })
            if len(results) >= max_results:
                break

    return results if results else [{"error": "No results parsed from Bing response"}]


def fetch_page(url: str, max_chars: int = 5000) -> dict:
    """Fetch a URL and extract readable text content."""
    try:
        r = httpx.get(url, headers=SEARCH_HEADERS, timeout=20.0, follow_redirects=True)
        r.raise_for_status()

        text = r.text
        # Strip script/style
        text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        # Strip HTML
        text = re.sub(r'<[^>]+>', ' ', text)
        # Collapse whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return {"url": url, "content": text[:max_chars], "status": r.status_code}
    except Exception as e:
        return {"url": url, "error": str(e), "status": 0}


if __name__ == "__main__":
    # Fix GBK encoding issue on Windows
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')

    parser = argparse.ArgumentParser(description="Search web or fetch URL")
    sub = parser.add_subparsers(dest="mode", required=True)

    # search
    sp = sub.add_parser("search", help="Search the web")
    sp.add_argument("query", help="Search query")
    sp.add_argument("--max-results", type=int, default=5)

    # fetch
    fp = sub.add_parser("fetch", help="Fetch a URL")
    fp.add_argument("url", help="URL to fetch")
    fp.add_argument("--max-chars", type=int, default=5000)

    args = parser.parse_args()

    if args.mode == "search":
        results = search_bing(args.query, args.max_results)
        print(json.dumps({"ok": "error" not in results[0], "data": results}, ensure_ascii=False))
    elif args.mode == "fetch":
        result = fetch_page(args.url, args.max_chars)
        print(json.dumps({"ok": "error" not in result, "data": result}, ensure_ascii=False))
