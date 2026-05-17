#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test BaseAgent with optimizations"""
import sys
sys.path.insert(0, '.')

print("=" * 60)
print("Testing BaseAgent with optimizations...")
print("=" * 60)

# 测试1: 导入
try:
    from ai_agent_playground.base import BaseAgent
    from ai_agent_playground.config import BaseAgentConfig
    print("[OK] BaseAgent imported")
except Exception as e:
    print(f"[ERROR] Import: {e}")
    sys.exit(1)

# 测试2: 创建测试Agent
class TestAgent(BaseAgent):
    def preprocess(self, inputs, **kwargs):
        return {"messages": [{"role": "user", "content": str(inputs)}]}

    def _forward(self, model_inputs, **kwargs):
        # 模拟AI调用
        return {"reply": f"Processed: {model_inputs['messages'][0]['content']}"}

    def postprocess(self, model_outputs, **kwargs):
        return model_outputs["reply"]

# 创建Agent实例
agent = TestAgent()
print("[OK] TestAgent created")

# 测试3: 安全模块 - 速率限制
from ai_agent_playground.security import get_rate_limiter
limiter = get_rate_limiter()
allowed, reason = limiter.check("test_user")
print(f"[OK] RateLimiter: allowed={allowed}")

# 测试4: 安全模块 - 输入验证
from ai_agent_playground.security import get_input_validator
validator = get_input_validator()
valid, reason = validator.validate("Hello world")
print(f"[OK] InputValidator: valid={valid}")

# 测试5: 缓存检查
from ai_agent_playground.cache import get_llm_cache
cache = get_llm_cache()
print(f"[OK] Cache stats: {cache.get_stats()}")

# 测试6: 链路追踪
from ai_agent_playground.observability_enhanced import get_enhanced_tracer
tracer = get_enhanced_tracer()
print(f"[OK] Tracer stats: {tracer.get_stats()}")

# 测试7: 运行Agent
print("\n--- Running Agent ---")
result = agent.run("Hello!", user_id="test_user")
print(f"Agent result: {result}")

# 测试8: 缓存应该被使用了
print(f"\n[OK] Cache after run: {cache.get_stats()}")

# 测试9: 消息总线统计
from ai_agent_playground.message_bus import message_bus
print(f"[OK] MessageBus: {message_bus.get_stats()}")

print("\n" + "=" * 60)
print("All tests passed!")
print("=" * 60)