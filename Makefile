# --- acceptance_backfill (EVAL_ACCEPTANCE_STANDARD.md P4) ---
# 证据来源: tests/
# 修正 2026-11-13（A 类）：tests/test_security.py 导入 agent/server.py，而该模块在
# 缺 DEEPSEEK_API_KEY 时于 import 期 raise（agent/server.py:135），导致 collection 报错。
# 与 CI 同法注入占位 key（.github/workflows/ci.yml 的 env 块）；没有测试真调 API。
.PHONY: eval

eval:
	DEEPSEEK_API_KEY=sk-local-placeholder DEEPSEEK_BASE_URL=https://api.deepseek.com python3 -m pytest tests -q
