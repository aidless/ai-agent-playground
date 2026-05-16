"""
配置系统 —— 所有 Agent 的"设置面板"。

想象你在买手机，每个手机都有：
  - 屏幕大小（比如 6.1 寸）
  - 存储空间（比如 256GB）
  - 颜色（比如黑色）

这些就是"配置项"——描述一个东西的参数，但还不是东西本身。

在代码里，配置 = 一个装满默认值的盒子。
Agent 从盒子里拿参数用，不需要自己到处找。

这个文件定义的是"基础配置盒"。每个 Agent 继承它，然后添加自己需要的参数。
就像 iPhone 15 继承"手机"的基础配置，再加上"灵动岛"这个特有的配置。
"""

# dataclass 是 Python 的一键工具：你写字段名+类型，Python 自动生成 __init__、__repr__ 等方法
# 省去写重复的 def __init__(self, model, max_tokens, ...):
from dataclasses import dataclass, fields
from typing import ClassVar  # ClassVar = 属于类的变量，不属于实例（所有 Agent 共享同一个 agent_type）


@dataclass  # <- 这个装饰器告诉 Python："帮我把下面的类变成一个配置盒"
class BaseAgentConfig:
    """
    所有 Agent 配置的"老祖宗"。

    就像"手机"这个抽象概念：所有手机都有屏幕、电池、操作系统。
    但具体的 iPhone 还是小米，决定了这些参数的具体值。

    使用方式：
      class MyAgentConfig(BaseAgentConfig):  # 继承老祖宗
          agent_type = "my-agent"           # 给自己起个名字
          model = "gpt-4"                   # 覆盖默认模型
          custom_param: int = 42            # 添加自己的参数
    """

    # ---- 类级别的变量（所有 Agent 配置共享） ----
    # ClassVar 的意思是：这个变量贴在"类"身上，不在"实例"身上
    # 就像"人类的物种名是智人"——贴在"人类"这个概念上，不是贴在你我个人身上
    agent_type: ClassVar[str] = "base"

    # ---- 实例级别的变量（每个 Agent 可以不同） ----
    # 下面这些是"默认值"——如果你不指定，就用这些

    # 用哪个 AI 模型？就像选"用哪个翻译官"
    model: str = "deepseek-v4-pro[1m]"

    # AI 最多回复多少个 token？token ≈ 半个汉字或 3/4 个英文单词
    # 2048 大概能回复 1000-1500 个汉字
    max_tokens: int = 2048

    # 系统提示词——告诉 AI "你是谁、怎么说话、有什么规则"
    # 这就像给客服的"话术手册"
    system_prompt: str = "You are a helpful AI assistant."

    # ============================================================
    #  工具方法：把配置变成字典 / 从字典恢复配置
    #  用途：保存配置到文件、从文件加载配置
    # ============================================================

    def to_dict(self) -> dict:
        """
        把配置"打包"成字典。就像把手机参数写在一张纸上。

        返回示例：
          {"agent_type": "hello", "model": "deepseek-v4-pro[1m]", "max_tokens": 1024}
        """
        result = {"agent_type": self.agent_type}
        for f in fields(self):
            # 跳过 agent_type 本身（因为它是 ClassVar，不在 fields 里）
            # 这个 if 只是安全检查
            if f.name not in ("agent_type",):
                result[f.name] = getattr(self, f.name)
        return result

    @classmethod
    def from_dict(cls, data: dict) -> "BaseAgentConfig":
        """
        从字典"恢复"配置。就像看着手机参数纸，造一个配置盒。

        如果字典里有不认识的参数（比如老版本的参数被删了）——直接跳过，不会报错。
        这是"向前兼容"：新代码能读旧配置。
        """
        valid_keys = {f.name for f in fields(cls)}  # 收集所有"合法参数名"
        kwargs = {k: v for k, v in data.items() if k in valid_keys}  # 只取认识的
        return cls(**kwargs)  # 用合法参数创建配置实例
