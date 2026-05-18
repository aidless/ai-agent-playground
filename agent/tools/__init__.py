"""agent.tools — ToolRegistry 与内置工具

自动发现所有工具模块，批量注册到 registry。

支持两种发现策略:
    1. AST 解析（优先）— 检测 registry.register() 调用，不加载模块代码
    2. import 导入（回退）— import 模块后用 TOOLS 列表注册
"""

import ast
import importlib
import logging
import pkgutil
from pathlib import Path

logger = logging.getLogger(__name__)


def _has_tool_pattern(filepath: str) -> bool:
    """AST 检测：文件是否包含工具注册模式（借鉴 Hermes Agent）

    检测两种模式:
        1. registry.register(...) — Hermes Agent 标准
        2. TOOLS.append(...) — 传统 TOOLS 列表模式
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            source = f.read()
        tree = ast.parse(source, filename=filepath)
    except (OSError, SyntaxError):
        return False

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            # 模式 1: registry.register( 或 .register(
            if node.func.attr == "register":
                return True
            # 模式 2: TOOLS.append(
            if node.func.attr == "append":
                # 检查调用者是否是 TOOLS (或类似变量)
                if isinstance(node.func.value, ast.Name):
                    name = node.func.value.id.upper()
                    if "TOOL" in name or "REGISTRY" in name:
                        return True
    return False


def register_all(registry):
    """自动发现 tools/ 目录下的工具模块并注册"""
    import agent.tools as pkg

    pkg_path = Path(pkg.__path__[0]) if pkg.__path__ else None
    if not pkg_path:
        return

    for _, mod_name, _ in pkgutil.iter_modules([str(pkg_path)]):
        if mod_name == "registry":
            continue

        module_path = pkg_path / f"{mod_name}.py"

        # 策略 1: AST 检测（不执行模块代码）
        has_tool = _has_tool_pattern(str(module_path))

        if has_tool:
            # AST 已确认含工具模式 → 安全导入
            try:
                mod = importlib.import_module(f"agent.tools.{mod_name}")
            except ImportError as e:
                logger.warning("跳过 agent.tools.%s: %s", mod_name, e)
                continue

            tool_list = getattr(mod, "TOOLS", None)
            if isinstance(tool_list, list):
                for definition, func in tool_list:
                    registry.register(
                        name=definition["name"],
                        description=definition["description"],
                        parameters=definition["parameters"],
                        func=func,
                    )
                    logger.info("注册工具: %s (from %s)", definition["name"], mod_name)
        else:
            # 无 register 调用 → 跳过（如纯工具脚本）
            logger.debug("跳过 %s: 未检测到工具注册模式", mod_name)
