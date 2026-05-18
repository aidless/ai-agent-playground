"""Web 搜索 + 网页抓取工具"""

import re
import html as html_mod

TOOLS = []

# ── 搜索 ──

SEARCH_DEF = {
    "name": "web_search",
    "description": "搜索互联网，返回标题+链接+摘要。适合查最新信息、文档、新闻。",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "搜索关键词"},
            "count": {"type": "integer", "description": "返回结果数（默认 5）"},
        },
        "required": ["query"],
    },
}


def web_search(query: str, count: int = 5) -> str:
    """DuckDuckGo 搜索（免费，无需 API key）"""
    # 优先用 duckduckgo_search 库
    try:
        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=count))
        if not results:
            return "无搜索结果"
        lines = []
        for i, r in enumerate(results, 1):
            lines.append(f"{i}. {r.get('title', '')}\n   {r.get('href', '')}\n   {r.get('body', '')}")
        return "\n\n".join(lines)
    except ImportError:
        pass
    except Exception as e:
        return f"搜索失败: {e}"

    # 兜底：用 httpx 抓 DuckDuckGo HTML
    try:
        import httpx

        resp = httpx.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
            },
            timeout=15,
        )
        resp.raise_for_status()
        # 简单解析 HTML 提取结果
        results = re.findall(
            r'<a rel="nofollow" href="(.*?)".*?class="result__a".*?>(.*?)</a>.*?class="result__snippet".*?>(.*?)</a>',
            resp.text,
            re.DOTALL,
        )
        if not results:
            return "无搜索结果"
        lines = []
        for i, (url, title, snippet) in enumerate(results[:count], 1):
            lines.append(f"{i}. {html_mod.unescape(re.sub(r'<.*?>', '', title)).strip()}\n   {url}\n   {html_mod.unescape(re.sub(r'<.*?>', '', snippet)).strip()}")
        return "\n\n".join(lines)
    except Exception as e:
        return f"搜索失败: {e}"


TOOLS.append((SEARCH_DEF, web_search))

# ── 网页抓取 ──

FETCH_DEF = {
    "name": "web_fetch",
    "description": "读取指定 URL 的文本内容（Markdown 格式）",
    "parameters": {
        "type": "object",
        "properties": {
            "url": {"type": "string", "description": "网页 URL（需完整 URL，含 https://）"},
        },
        "required": ["url"],
    },
}


def web_fetch(url: str) -> str:
    """抓取网页内容并转为纯文本"""
    if not url.startswith(("http://", "https://")):
        return "错误：URL 必须以 http:// 或 https:// 开头"

    import httpx

    try:
        resp = httpx.get(url, timeout=30, follow_redirects=True)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "")
        if "text" not in content_type and "html" not in content_type:
            return f"[{len(resp.content)} bytes 二进制内容，无法显示文本]"

        text = resp.text
        # 简单清理：去掉 script/style 标签
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        # 截断到 8000 字符
        if len(text) > 8000:
            text = text[:8000] + "\n\n... [内容已截断]"
        return text
    except Exception as e:
        return f"抓取失败: {e}"


TOOLS.append((FETCH_DEF, web_fetch))
