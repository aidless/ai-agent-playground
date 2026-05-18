"""计算工具：加减乘除，供 ToolRegistry 注册使用"""

import builtins
import math
import re
from typing import Union

TOOL_DEFINITION = {
    "name": "calculator",
    "description": "执行数学运算（加减乘除、幂、平方根等）",
    "parameters": {
        "type": "object",
        "properties": {
            "expression": {
                "type": "string",
                "description": "数学表达式，如 '2 + 3 * 4' 或 'sqrt(16)'",
            }
        },
        "required": ["expression"],
    },
}


def calculator(expression: str) -> Union[float, str]:
    """安全的数学计算器，仅允许白名单运算"""
    # 只允许数字、运算符、括号、数学函数名
    safe_pattern = r"^[\d\s\+\-\*\/\(\)\.\,\%]+$"
    safe = re.match(safe_pattern, expression.strip())

    if safe:
        try:
            result = eval(expression, {"__builtins__": {}}, {"math": math})
            return round(float(result), 6)
        except Exception as e:
            return f"计算错误: {e}"
    else:
        # 含数学函数的表达式
        allowed = {"abs", "round", "int", "float", "min", "max", "sum"}
        try:
            # 检查是否只包含允许的调用
            tokens = set(re.findall(r"[a-zA-Z_]\w*", expression))
            extra = tokens - allowed - {"math"}
            if extra:
                return f"不支持的函数: {', '.join(extra)}"
            result = eval(expression, {"__builtins__": {}}, {"math": math, **{f: getattr(builtins, f) for f in allowed}})
            return round(float(result), 6)
        except Exception as e:
            return f"计算错误: {e}"
