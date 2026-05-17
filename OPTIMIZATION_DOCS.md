# AI Agent Playgound 优化模块说明文档

## 📋 概述

本文档详细介绍了 `ai-agent-playground` 项目中添加的优化模块。这些模块经过精心设计，用于提升 Agent 系统的性能、安全性、可观测性和扩展性。

### 优化模块列表

| 模块 | 文件 | 功能 |
|------|------|------|
| 消息总线 | message_bus.py | Agent 之间的统一通信层 |
| Agent注册中心 | agent_registry.py | 动态注册/发现Agent |
| 可观测性 | observability_enhanced.py | 链路追踪 + 实时告警 |
| 容错机制 | resilience.py | 重试 + 熔断器 |
| 扩展性 | extension.py | 配置化 + 插件体系 |
| 安全控制 | security.py | 权限 + 验证 + 限流 |
| LLM缓存 | cache.py | 响应缓存 |
| 测试框架 | testing.py | 单元测试 + Mock |

---

## 🔧 环境要求

- Python 3.11+
- 虚拟环境（推荐使用 .venv）

### 安装依赖

```bash
cd ai-agent-playground
.venv\Scripts\pip install -r requirements.txt
# 或者使用 uv
uv sync
```

---

## 1️⃣ message_bus.py — 消息总线

### 功能说明

消息总线是 Agent 之间的统一通信层，提供以下特性：

- **消息去重**：相同主题的消息在 TTL 内只保留最新
- **批量处理**：聚合多个消息批量投递，减少网络开销
- **增量同步**：只传状态变化，不传全量数据
- **本地缓存**：每个 Agent 独立的缓存，减少重复请求

### 核心类

```python
from ai_agent_playground.message_bus import (
    MessageBus,
    get_message_bus,
    subscribe,
    publish,
    message_bus
)
```

### 使用示例

#### 1. 发布/订阅模式

```python
from ai_agent_playground.message_bus import subscribe, publish

# 订阅消息
@subscribe("task:complete")
def on_task_complete(data):
    print(f"Task完成: {data['task_id']}")

@subscribe("agent:error")
def on_agent_error(data):
    print(f"Agent错误: {data['agent']} - {data['error']}")

# 发布消息
publish("task:complete", {"task_id": "123", "status": "done"})
```

#### 2. 使用本地缓存

```python
from ai_agent_playground.message_bus import message_bus

# 获取指定Agent的缓存
cache = message_bus.get_cache("agent_1")

# 设置缓存
cache.set("config", {"key": "value"}, ttl_ms=10000)

# 获取缓存（不存在则调用 fetcher）
value = cache.get_or_fetch("data", lambda: load_data(), ttl_ms=5000)
```

#### 3. 统计信息

```python
from ai_agent_playground.message_bus import message_bus

# 查看统计
message_bus.print_stats()
# 输出: [MessageBus] published=100 delivered=95 deduped=5 batched=10 errors=0
```

---

## 2️⃣ agent_registry.py — Agent注册中心

### 功能说明

Agent 注册中心提供动态的 Agent 注册和发现功能：

- **动态注册**：支持装饰器和函数式注册
- **服务发现**：通过名字获取 Agent 实例
- **状态管理**：跟踪 Agent 状态（ready/busy/error）
- **统计信息**：调用次数、错误次数统计

### 核心类

```python
from ai_agent_playground.agent_registry import (
    AgentRegistry,
    get_agent_registry,
    register,
    get_agent,
    list_agents,
    agent_registry
)
```

### 使用示例

#### 1. 装饰器方式注册

```python
from ai_agent_playground.agent_registry import register
from ai_agent_playground.base import BaseAgent

@register("planner")
class PlannerAgent(BaseAgent):
    def preprocess(self, inputs, **kwargs):
        return {"task": inputs}

    def _forward(self, model_inputs, **kwargs):
        return {"result": "planned"}

    def postprocess(self, model_outputs, **kwargs):
        return model_outputs["result"]
```

#### 2. 函数式注册

```python
from ai_agent_playground.agent_registry import register
from ai_agent_playground.base import BaseAgent

class ExecutorAgent(BaseAgent):
    ...

register("executor", ExecutorAgent)
```

#### 3. 获取 Agent

```python
from ai_agent_playground.agent_registry import get_agent

planner = get_agent("planner")
result = planner.run("帮我规划一个任务")
```

#### 4. 查看状态

```python
from ai_agent_playground.agent_registry import agent_registry

# 列出所有 Agent
print(agent_registry.list_all())

# 按状态筛选
print(agent_registry.list_by_status("ready"))

# 打印仪表盘
agent_registry.print_dashboard()
```

---

## 3️⃣ observability_enhanced.py — 可观测性

### 功能说明

可观测性模块提供跨 Agent 的链路追踪和实时告警功能：

- **链路追踪**：同一个请求在多个 Agent 间传递 trace_id
- **实时告警**：错误率/延迟超过阈值自动提醒
- **统计面板**：按 Agent 统计耗时、错误率

### 核心类

```python
from ai_agent_playground.observability_enhanced import (
    EnhancedTracer,
    AlertManager,
    get_enhanced_tracer,
    get_alert_manager
)
```

### 使用示例

#### 1. 链路追踪

```python
from ai_agent_playground.observability_enhanced import get_enhanced_tracer

tracer = get_enhanced_tracer()

# 开始追踪
trace_id = tracer.start_trace("user_request")

# Agent 1
with tracer.start_span("planner", parent_id=trace_id) as span:
    result = planner.run(task)
    span.attributes["task_type"] = task.type

# Agent 2（传递 trace_id）
with tracer.start_span("executor", parent_id=trace_id) as span:
    result = executor.run(task, trace_id=trace_id)

# 查看统计
tracer.print_dashboard()
```

#### 2. 告警

```python
from ai_agent_playground.observability_enhanced import get_alert_manager

alerts = get_alert_manager()

# 定义告警回调
def on_alert(alert):
    print(f"🚨 {alert['type']}: {alert['message']}")
    # 可以发送邮件/Slack/钉钉

alerts.add_callback(on_alert)
alerts.start()  # 启动后台检查

# 手动检查
alerts.print_alerts()
```

---

## 4️⃣ resilience.py — 容错机制

### 功能说明

容错机制确保系统在异常情况下仍能正常运行：

- **自动重试**：失败时自动重试，支持指数退避
- **熔断器**：失败过多时暂停，避免级联故障
- **超时控制**：防止 Agent 卡死

### 核心类

```python
from ai_agent_playground.resilience import (
    retry,
    CircuitBreaker,
    CircuitBreakerManager,
    with_timeout,
    get_circuit_breaker_manager
)
```

### 使用示例

#### 1. 重试装饰器

```python
from ai_agent_playground.resilience import retry

@retry(max_attempts=3, base_delay=1.0, exponential=True)
def call_external_api(data):
    # 可能失败的网络调用
    response = requests.post(url, json=data)
    response.raise_for_status()
    return response.json()
```

#### 2. 熔断器

```python
from ai_agent_playground.resilience import CircuitBreaker, get_circuit_breaker_manager

# 获取或创建熔断器
cbm = get_circuit_breaker_manager()
cb = cbm.get("external_api", failure_threshold=3, recovery_timeout=30)

# 使用熔断器
try:
    with cb:
        result = call_external_api(data)
except CircuitOpenError:
    print("服务暂时不可用，请稍后重试")
```

#### 3. 超时控制

```python
from ai_agent_playground.resilience import with_timeout

# 30秒超时，超时返回 "timeout"
result = with_timeout(agent.run, timeout=30, default="timeout")
```

#### 4. 熔断器状态

```python
from ai_agent_playground.resilience import get_circuit_breaker_manager

cbm = get_circuit_breaker_manager()
cbm.print_dashboard()
```

---

## 5️⃣ security.py — 安全控制

### 功能说明

安全模块提供全面的安全保护：

- **权限控制**：基于角色/用户的访问控制
- **输入验证**：防止 prompt injection、恶意输入
- **速率限制**：防止 API 滥用

### 核心类

```python
from ai_agent_playground.security import (
    PermissionManager,
    InputValidator,
    RateLimiter,
    get_permission_manager,
    get_input_validator,
    get_rate_limiter
)
```

### 使用示例

#### 1. 权限管理

```python
from ai_agent_playground.security import get_permission_manager

pm = get_permission_manager()

# 授予权限
pm.grant("user1", "executor")
pm.grant("user1", "planner")

# 角色分配
pm.assign_role("admin_user", "admin")

# 检查权限
if pm.check_with_role("user1", "executor"):
    print("允许访问")
else:
    print("拒绝访问")
```

#### 2. 输入验证

```python
from ai_agent_playground.security import get_input_validator

validator = get_input_validator()

# 验证输入
valid, reason = validator.validate(user_input)
if not valid:
    raise ValueError(f"Invalid input: {reason}")

# 净化输入
clean_input = validator.sanitize(user_input)
```

#### 3. 速率限制

```python
from ai_agent_playground.security import get_rate_limiter

limiter = get_rate_limiter(max_requests=100, window_sec=60)

# 检查限制
allowed, reason = limiter.check("user_id")
if not allowed:
    raise PermissionError(f"Rate limited: {reason}")

# 获取剩余次数
remaining = limiter.get_remaining("user_id")
```

---

## 6️⃣ cache.py — LLM缓存

### 功能说明

LLM 缓存模块缓存 LLM 响应，减少 API 调用：

- **请求缓存**：相同请求不重复调用 LLM
- **LRU 驱逐**：自动清理最老的缓存
- **命中率统计**：监控缓存效果

### 核心类

```python
from ai_agent_playground.cache import (
    LLMCache,
    get_llm_cache,
    get_embedding_cache
)
```

### 使用示例

```python
from ai_agent_playground.cache import get_llm_cache

cache = get_llm_cache()

# 检查缓存
messages = [{"role": "user", "content": "Hello"}]
model = "deepseek-v4-pro[1m]"

cached = cache.get(messages, model)
if cached:
    print("使用缓存")
    return cached

# 调用 LLM
response = llm.send(messages, model=model)

# 存入缓存
cache.set(messages, model, response)

# 查看统计
cache.print_stats()
```

---

## 7️⃣ extension.py — 扩展性

### 功能说明

扩展性模块支持配置化和插件体系：

- **YAML配置**：通过配置文件定义 Agent
- **插件加载**：动态加载 Agent 模块
- **Worker池**：多线程任务队列

### 核心类

```python
from ai_agent_playground.extension import (
    load_config,
    register_from_config,
    get_plugin_manager,
    get_worker_pool
)
```

### 使用示例

#### 1. YAML 配置

```yaml
# agents.yaml
agents:
  - name: planner
    class_path: my_agents.planner.PlannerAgent
    enabled: true
    config:
      model: deepseek-v4
      max_tokens: 2048
```

```python
from ai_agent_playground.extension import load_config, register_from_config

config = load_config("agents.yaml")
register_from_config(config)
```

#### 2. 插件加载

```python
from ai_agent_playground.extension import get_plugin_manager

pm = get_plugin_manager("plugins")
count = pm.load_all("*.py")
print(f"加载了 {count} 个插件")
```

#### 3. Worker 池

```python
from ai_agent_playground.extension import get_worker_pool

pool = get_worker_pool(num_workers=4)
pool.start()

# 提交任务
pool.submit(agent.run, task)
```

---

## 8️⃣ testing.py — 测试框架

### 功能说明

测试框架帮助开发者测试 Agent：

- **Mock LLM**：不需要真实 API 的测试
- **测试套件**：批量运行测试用例
- **测试报告**：详细的测试结果

### 核心类

```python
from ai_agent_playground.testing import (
    MockLLM,
    create_test_agent,
    AgentTestSuite,
    AgentTestCase
)
```

### 使用示例

```python
from ai_agent_playground.testing import MockLLM, create_test_agent

# 创建 Mock LLM
mock = MockLLM()
mock.add_response("hello", "Hi there!")
mock.add_response("how are you", "I'm doing well, thanks!")

# 创建测试 Agent
agent = create_test_agent(MyAgent, mock_llm=mock)

# 运行测试
result = agent.run("hello")
print(result)  # 输出: Hi there!
```

---

## BaseAgent 集成

`BaseAgent` 已经自动集成了所有优化模块。以下是完整的运行流程：

```python
from ai_agent_playground.base import BaseAgent

class MyAgent(BaseAgent):
    def preprocess(self, inputs, **kwargs):
        return {"messages": [{"role": "user", "content": inputs}]}

    def _forward(self, model_inputs, **kwargs):
        return {"reply": self.llm.send(model_inputs["messages"])}

    def postprocess(self, model_outputs, **kwargs):
        return model_outputs["reply"]

# 创建并运行
agent = MyAgent()
result = agent.run("Hello!", user_id="user1")
```

**自动完成的操作**：

1. ✅ 速率限制检查
2. ✅ 输入验证
3. ✅ 缓存检查
4. ✅ 链路追踪
5. ✅ 自动重试（3次）
6. ✅ 结果缓存
7. ✅ 消息发布

---

## 运行测试

```bash
# 测试所有模块
python test_optimizations.py

# 测试 Agent
python test_agent.py

# 测试特定模块
python -c "from ai_agent_playground.message_bus import message_bus; print(message_bus.get_stats())"
```

---

## 常见问题

### Q: 如何禁用某个模块？

A: 创建自定义的 `BaseAgent` 子类，覆盖 `run` 方法：

```python
class SimpleAgent(BaseAgent):
    def run(self, inputs, **kwargs):
        # 不使用任何优化
        model_inputs = self.preprocess(inputs)
        model_outputs = self._forward(model_inputs)
        return self.postprocess(model_outputs)
```

### Q: 如何添加自定义的优化？

A: 在 `BaseAgent` 子类中覆盖相应方法：

```python
class CustomAgent(BaseAgent):
    def _run_with_retry(self, inputs, **kwargs):
        # 自定义重试逻辑
        ...
```

### Q: 如何查看实时统计？

A: 调用各个模块的统计方法：

```python
from ai_agent_playground.message_bus import message_bus
from ai_agent_playground.cache import get_llm_cache
from ai_agent_playground.agent_registry import agent_registry

message_bus.print_stats()
get_llm_cache().print_stats()
agent_registry.print_dashboard()
```

---

## 🔧 故障排除

### 1. 导入错误

#### ❌ `ModuleNotFoundError: No module named 'ai_agent_playground'`

**原因**：当前目录不在 Python 路径中

**解决**：
```bash
cd C:\Users\Administrator\Desktop\ai-agent-playground
.venv\Scripts\python.exe test.py
```

#### ❌ `NameError: name 'Callable' is not defined`

**原因**：Python 版本过旧或导入语句错误

**解决**：
```python
# 在文件开头添加
from collections.abc import Callable
# 或
from typing import Callable
```

---

### 2. 运行错误

#### ❌ `PermissionError: Rate limited`

**原因**：用户请求频率超过限制

**解决**：
```python
# 检查剩余次数
limiter = get_rate_limiter()
remaining = limiter.get_remaining("user_id")
print(f"剩余请求次数: {remaining}")

# 或调整限制
limiter = get_rate_limiter(max_requests=1000, window_sec=60)
```

#### ❌ `ValueError: Invalid input`

**原因**：输入包含恶意内容或过长

**解决**：
```python
validator = get_input_validator()

# 先验证
valid, reason = validator.validate(user_input)
if not valid:
    user_input = validator.sanitize(user_input)  # 净化后重试
```

#### ❌ `CircuitOpenError: CircuitBreaker is OPEN`

**原因**：服务失败次数过多，熔断器已打开

**解决**：
```python
# 等待恢复（默认30秒）
# 或手动重置
cbm = get_circuit_breaker_manager()
cb = cbm.get("service_name")
cb._state.status = "closed"
cb._state.failure_count = 0
```

---

### 3. 性能问题

#### ❌ 消息队列堵塞

**原因**：消息批量太大或处理太慢

**解决**：
```python
# 减小批量大小
bus = MessageBus(batch_size=5, batch_interval_ms=50)

# 或使用异步处理
import asyncio
```

#### ❌ 缓存命中率低

**原因**：请求内容变化大，无法复用

**解决**：
```python
# 检查缓存统计
cache = get_llm_cache()
stats = cache.get_stats()
print(f"命中率: {stats['hit_rate']}")

# 调整 TTL
cache.set(messages, model, response, ttl=7200)  # 2小时
```

---

### 4. 环境问题

#### ❌ `.venv\Scripts\python.exe: not found`

**原因**：虚拟环境未创建

**解决**：
```bash
cd C:\Users\Administrator\Desktop\ai-agent-playground
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

#### ❌ 依赖版本冲突

**原因**：第三方库版本不兼容

**解决**：
```bash
# 清理并重新安装
.venv\Scripts\pip uninstall -y -r requirements.txt
.venv\Scripts\pip install -r requirements.txt
```

---

### 5. 网络问题

#### ❌ LLM 调用超时

**原因**：网络延迟或 API 限流

**解决**：
```python
from ai_agent_playground.resilience import with_timeout

# 添加超时
result = with_timeout(llm.send, timeout=60, default="timeout")
```

#### ❌ 缓存无法连接

**原因**：Redis 等外部缓存未启动

**解决**：
```bash
# 使用内存缓存（默认）
# 或启动 Redis
docker-compose up -d redis
```

---

### 6. 调试技巧

#### 启用详细日志

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# 或在模块中设置
import ai_agent_playground.message_bus as mb
mb._debug = True
```

#### 查看所有统计

```python
from ai_agent_playground.message_bus import message_bus
from ai_agent_playground.cache import get_llm_cache
from ai_agent_playground.agent_registry import agent_registry
from ai_agent_playground.resilience import get_circuit_breaker_manager

message_bus.print_stats()
get_llm_cache().print_stats()
agent_registry.print_dashboard()
get_circuit_breaker_manager().print_dashboard()
```

#### 断点调试

```python
# 在需要调试的地方添加
import pdb; pdb.set_trace()

# 或使用 IDE 的调试功能
```

---

## 📝 更新日志

### v1.0.0 (2024-05-17)

- 初始版本
- 添加 8 个优化模块
- 集成到 BaseAgent

---

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

---

*文档生成时间: 2024-05-17*