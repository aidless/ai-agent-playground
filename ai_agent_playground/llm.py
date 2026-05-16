"""
LLM 客户端 —— 所有 Agent 共享的"电话线"。

想象你办公室里有 5 个销售，他们不需要每人装一条电话线。
只需要一个总机，谁需要打电话就拿起话机。

这个文件就是这个"总机"——一个 Python 进程里只创建一个 Anthropic 客户端，
所有 Agent 共用。节省内存、避免重复加载配置。

这个模式来自 HuggingFace Transformers：模型只加载一次，所有 Pipeline 共享。
"""

import os
from pathlib import Path

# Anthropic 官方 Python SDK
# 虽然我们用 DeepSeek 的 API，但因为 DeepSeek 提供了 Anthropic 兼容接口，
# 所以代码里用 Anthropic SDK 就行（就像充电器接口一样，Type-C 统一了）
from anthropic import Anthropic
from anthropic.types import TextBlock  # TextBlock = AI 回复里的"文字块"
from dotenv import load_dotenv  # 从 .env 文件加载 API Key（避免把密码写在代码里）

# ---- 模块级别的"是否已经加载过 .env"标记 ----
# 为什么需要这个？因为 Python 模块只会被导入一次。
# 但如果有多个文件都调了 _ensure_dotenv()，我们不想重复加载 .env。
# 这个标记就是"我已经加载过了"的记号。
_load_dotenv_done = False


def _ensure_dotenv():
    """
    确保 .env 文件被加载了。如果已经加载过，直接跳过。

    .env 文件长这样：
      DEEPSEEK_API_KEY=sk-xxxxx
      DEEPSEEK_BASE_URL=https://api.deepseek.com/anthropic

    这样做的目的是：API Key 不写在代码里（代码要上传 GitHub 的！）。
    写在 .env 里，.gitignore 会阻止上传，Key 不会泄露。
    """
    global _load_dotenv_done
    if not _load_dotenv_done:
        # 往上级目录找 .env（从 ai_agent_playground/ 找到项目根目录）
        load_dotenv(Path(__file__).parent.parent / ".env")
        _load_dotenv_done = True


class LLMClient:
    """
    薄薄一层的包装——只做三件事：
      1. 认证（你是谁？用 API Key 证明身份）
      2. 发送消息给 AI
      3. 从 AI 回复里提取纯文本

    为什么叫"薄包装"？因为它几乎不加逻辑——就是帮大家省去重复写那几行代码的麻烦。
    """

    def __init__(self):
        """
        初始化：读 API Key，建连接。

        就像：插上电话线，拨号，确认能打通。
        """
        # 确保 .env 加载了（API Key 在里面）
        _ensure_dotenv()

        # 从环境变量拿 Key（os.getenv 比 os.environ[...] 安全：取不到返回 None，不报错）
        base_url = os.getenv("DEEPSEEK_BASE_URL")
        api_key = os.getenv("DEEPSEEK_API_KEY")

        # 如果 Key 没设置，直接报错——比后面再报错更容易定位问题
        if not base_url or not api_key:
            raise RuntimeError(
                "DEEPSEEK_BASE_URL and DEEPSEEK_API_KEY must be set in .env file. "
                "Copy .env.example to .env and fill in your keys."
            )

        # 创建真实的 API 客户端——这就是那根"电话线"
        self._client = Anthropic(base_url=base_url, api_key=api_key)

    def send(
        self,
        messages: list[dict],
        *,
        model: str = "deepseek-v4-pro[1m]",
        max_tokens: int = 2048,
        system: str = "",
    ) -> str:
        """
        发一条消息给 AI，拿回纯文本回复。

        参数:
          messages: 对话列表，每个元素 {"role": "user", "content": "你好"}
          model: 用哪个模型（就像选"用哪个翻译官"）
          max_tokens: AI 最多回复多少 token
          system: 系统提示（"你是客服"、"你是程序员"...）
          *: 后面的参数必须用关键字传（model=xxx），不能用位置传

        返回:
          AI 的纯文本回复（已经帮你把思考过程过滤掉了）
        """
        # 调用 API——这一行是整个系统最"贵"的操作（耗时+花钱）
        response = self._client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=messages,
        )
        # 从回复里提取纯文本（跳过模型的"思考过程"）
        return self._extract_text(response)

    @staticmethod
    def _extract_text(response) -> str:
        """
        从 AI 回复里提取纯文本。

        为什么需要这个？DeepSeek V4 是一个"推理模型"——
        它会先"思考"再回答（就像人先说"让我想想..."再给出答案）。
        思考内容存在 ThinkingBlock 里，答案存在 TextBlock 里。
        我们只需要答案，所以跳过 ThinkingBlock。

        就像：朋友说"我想了想...（3分钟思考）...答案是42"。
        你只需要"答案是42"。
        """
        parts = []
        for block in response.content:  # response.content 是一个列表，可能有多个块
            if isinstance(block, TextBlock):  # "这是文字块吗？"——是的话就收集
                parts.append(block.text)
        # 把所有文字块拼起来返回。如果没有文字块，返回占位符
        return "\n".join(parts) if parts else "[No text in response]"


# ============================================================
#  全局单例 —— "整个系统只有这一根电话线"
#
#  模块级变量 _client 只存一个实例。
#  get_client() 第一次调用时创建，之后永远返回同一个。
#
#  这避免了 5 个 Agent 创建 5 个连接——浪费内存、浪费网络连接。
# ============================================================

# 一开始是空的（还没人需要打电话）
_client: LLMClient | None = None


def get_client() -> LLMClient:
    """
    获取共享的 LLM 客户端。第一次调用时创建，之后直接返回。

    就像：第一次进办公室，装好电话；以后再进来，直接用那部电话。
    """
    global _client  # 声明：我要修改模块级的 _client 变量
    if _client is None:
        _client = LLMClient()  # 第一次：创建
    return _client  # 之后：直接返回已有的
