"""
Agent 基类 —— 所有 AI Agent 的"骨架"。

想象你在经营一家餐厅。每个服务员的工作流程都是一样的：
  1. 接单（preprocess）：客人说"我要一份牛排"→ 写成小票
  2. 做菜（_forward）：把小票给厨房 → 厨房做菜
  3. 上菜（postprocess）：厨房出菜 → 端给客人

这个文件定义的就是"服务员工作流程"——三步走，永远不变。
具体的 Agent（HelloAgent、CodeReviewAgent...）只需要告诉这三步分别做什么。

这个设计模式来自 HuggingFace Transformers 源码的 Pipeline 类：
  preprocess（预处理）→ _forward（模型推理）→ postprocess（后处理）
"""

from abc import ABC, abstractmethod  # ABC = 抽象基类，用来定义"骨架"，不能直接实例化
from typing import Any  # Any = 任意类型，"管它什么类型，先拿着"

from .config import BaseAgentConfig  # 配置盒
from .llm import get_client  # 共享的 AI 客户端（所有 Agent 用同一个）


class BaseAgent(ABC):
    """
    所有 Agent 的"骨架"。

    继承这个类的 Agent 只需要实现 3 个方法：
      preprocess  → 把用户输入变成 AI 能理解的样子
      _forward    → 调用 AI
      postprocess → 把 AI 的回复变成用户能理解的样子

    这就像填空游戏：模板我已经写好了，你把三个空填上就行。

    使用示例：
      class MyAgent(BaseAgent):
          def preprocess(self, inputs):   # 填空 1：怎么准备数据？
              return {"messages": [...]}

          def _forward(self, model_inputs):  # 填空 2：怎么调 AI？
              return {"reply": self.llm.send(...)}

          def postprocess(self, model_outputs):  # 填空 3：怎么格式化回复？
              return model_outputs["reply"]
    """

    # ---- 类变量：子类会覆盖这个 ----
    # 就像"默认配置型号"——子类会说"我用 HelloAgentConfig"，而不是"我用 BaseAgentConfig"
    config_class: type[BaseAgentConfig] = BaseAgentConfig

    def __init__(self, config: BaseAgentConfig | None = None):
        """
        创建一个 Agent。

        参数:
          config: 可选的配置盒。如果不给，就自动用默认配置。
                  就像"你可以选 256GB 的 iPhone，也可以直接用默认的 128GB 版本"。
        """
        # 如果用户给了配置就用它，否则创建一个默认配置
        # "is not None" 比 "or" 安全——因为空的配置盒 {} 也是 truthy
        self.config = config if config is not None else self.config_class()

        # 拿到共享的 AI 客户端（所有 Agent 用同一根电话线打给 AI）
        self.llm = get_client()

    # ============================================================
    #  核心方法：这就是"服务员三步走"
    #  run() 是唯一的对外接口——外部只调这个，不管内部怎么实现的
    # ============================================================

    def run(self, inputs: Any, **kwargs) -> Any:
        """
        跑一遍完整流程：准备 → AI 推理 → 格式化 → 返回结果。

        这是 Agent 的"一键启动"按钮。
        外部代码只调这个，不需要知道内部有三步。

        就像咖啡机上的"一键出咖啡"按钮——
        你不知道它内部先磨豆再加热再萃取，你只知道按了就出咖啡。
        """
        # 第1步：把用户输入变成 AI 能理解的格式
        # 比如：用户说 "hello" → {"messages": [{"role": "user", "content": "hello"}]}
        model_inputs = self.preprocess(inputs, **kwargs)

        # 第2步：调用 AI（这是最核心的一步，前面都是准备，后面都是整理）
        model_outputs = self._forward(model_inputs, **kwargs)

        # 第3步：把 AI 返回的原始数据变成用户能看懂的格式
        # 比如：{"reply": "Hello! How can I help?"} → "Hello! How can I help?"
        return self.postprocess(model_outputs, **kwargs)

    # ============================================================
    #  抽象方法：子类必须实现这三个
    #  "抽象"的意思：我只定义了"要有这些方法"，但没说"具体怎么做"
    #  就像老板说"要有个接单流程"，但每个服务员可以有自己的接单方式
    # ============================================================

    @abstractmethod
    def preprocess(self, inputs: Any, **kwargs) -> dict[str, Any]:
        """
        第1步：准备数据——把用户的原始输入变成 AI 能理解的格式。

        就像翻译官：用户说中文 → 翻译成 AI 能理解的"API 调用格式"。

        输入: 任何东西（字符串、文件、字典...取决于 Agent 类型）
        输出: 一个字典，包含调用 AI 需要的所有东西（消息、模型名、参数等）
        """
        ...  # ... 是 Python 的"占位符"，表示"这里留给子类实现"

    @abstractmethod
    def _forward(self, model_inputs: dict[str, Any], **kwargs) -> dict[str, Any]:
        """
        第2步：调用 AI——把准备好的数据发过去，拿回原始结果。

        这是整个流程中最关键的一步，也是最"贵"的一步（耗时 + 消耗 API 额度）。
        _forward 前面的下划线表示"这是内部方法，外部别直接调"。

        输入: preprocess 的输出（一个字典）
        输出: 一个字典，包含 AI 的原始回复
        """
        ...

    @abstractmethod
    def postprocess(self, model_outputs: dict[str, Any], **kwargs) -> Any:
        """
        第3步：格式化——把 AI 的原始回复变成用户喜欢的样子。

        就像：厨房出了一盘菜 → 服务员摆盘、加装饰 → 端上桌。

        输入: _forward 的输出（AI 的原始回答）
        输出: 任意格式（字符串、列表、字典...取决于 Agent 需要什么）
        """
        ...
