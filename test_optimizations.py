#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Test all optimization modules"""
import sys
sys.path.insert(0, '.')

print("=" * 50)
print("Testing optimization modules...")
print("=" * 50)

# Test 1: BaseAgent
try:
    from ai_agent_playground.base import BaseAgent
    print("[OK] BaseAgent imported")
except Exception as e:
    print(f"[ERROR] BaseAgent: {e}")
    sys.exit(1)

# Test 2: Security
try:
    from ai_agent_playground.security import get_rate_limiter
    limiter = get_rate_limiter()
    allowed, _ = limiter.check('test_user')
    print(f"[OK] RateLimiter: {'allowed' if allowed else 'blocked'}")
except Exception as e:
    print(f"[ERROR] RateLimiter: {e}")

# Test 3: Cache
try:
    from ai_agent_playground.cache import get_llm_cache
    cache = get_llm_cache()
    stats = cache.get_stats()
    print(f"[OK] LLM Cache: hits={stats['hits']}, misses={stats['misses']}")
except Exception as e:
    print(f"[ERROR] LLM Cache: {e}")

# Test 4: MessageBus
try:
    from ai_agent_playground.message_bus import message_bus
    stats = message_bus.get_stats()
    print(f"[OK] MessageBus: published={stats['published']}")
except Exception as e:
    print(f"[ERROR] MessageBus: {e}")

# Test 5: AgentRegistry
try:
    from ai_agent_playground.agent_registry import agent_registry
    print(f"[OK] AgentRegistry: {len(agent_registry.list_all())} agents")
except Exception as e:
    print(f"[ERROR] AgentRegistry: {e}")

# Test 6: Observability
try:
    from ai_agent_playground.observability_enhanced import get_enhanced_tracer
    tracer = get_enhanced_tracer()
    stats = tracer.get_stats()
    print(f"[OK] Tracer: traces={stats['total_traces']}")
except Exception as e:
    print(f"[ERROR] Tracer: {e}")

# Test 7: Resilience
try:
    from ai_agent_playground.resilience import get_circuit_breaker_manager
    cbm = get_circuit_breaker_manager()
    print(f"[OK] CircuitBreaker: initialized")
except Exception as e:
    print(f"[ERROR] CircuitBreaker: {e}")

# Test 8: Extension
try:
    from ai_agent_playground.extension import get_plugin_manager
    pm = get_plugin_manager()
    print(f"[OK] PluginManager: initialized")
except Exception as e:
    print(f"[ERROR] PluginManager: {e}")

print("")
print("=" * 50)
print("All modules tested!")
print("=" * 50)