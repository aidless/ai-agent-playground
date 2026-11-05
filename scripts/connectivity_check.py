"""Test internet connectivity from this environment"""
import json
import sys

try:
    import httpx
    # Test basic connectivity
    tests = []

    # 1. Test baidu (should work in China)
    try:
        r = httpx.get("https://www.baidu.com/s?wd=test", timeout=10.0, headers={"User-Agent": "Mozilla/5.0"})
        tests.append({"target": "baidu", "status": r.status_code, "ok": r.status_code == 200})
    except Exception as e:
        tests.append({"target": "baidu", "ok": False, "error": str(e)[:100]})

    # 2. Test bing
    try:
        r = httpx.get("https://cn.bing.com/search?q=test", timeout=10.0, headers={"User-Agent": "Mozilla/5.0"})
        tests.append({"target": "bing", "status": r.status_code, "ok": r.status_code == 200})
    except Exception as e:
        tests.append({"target": "bing", "ok": False, "error": str(e)[:100]})

    # 3. Test if we can reach any search engine
    try:
        r = httpx.get("https://www.google.com/search?q=test", timeout=5.0, headers={"User-Agent": "Mozilla/5.0"})
        tests.append({"target": "google", "status": r.status_code, "ok": r.status_code == 200})
    except Exception as e:
        tests.append({"target": "google", "ok": False, "error": str(e)[:100]})

    print(json.dumps(tests, indent=2, ensure_ascii=False))
except ImportError:
    print(json.dumps({"error": "httpx not installed"}))
