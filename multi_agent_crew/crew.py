"""
Crew 编排器 —— 把 4 个 AI Agent 串成一个"虚拟开发团队"。

这是你写的第 4 个项目，展示了多 Agent 协作的核心模式：
  产品经理 → 开发者 → 测试工程师 → 运维工程师

就像一条工厂流水线：
  1. PM（产品经理）：  "客户要什么？" → 拆成具体任务
  2. Dev（开发者）：   "我来实现" → 写代码
  3. QA（测试）：      "我来检查" → 审查代码质量
  4. DevOps（运维）：   "怎么部署" → 出部署方案

每个角色都是一个独立的 AI Agent，有自己的 system prompt（角色定位）。
Crew 的工作就是按顺序调用他们，把上一个的输出传给下一个。

这个设计模式来自 Transformers 的 generate() 方法：
  generate() 自己不动手，编排 9 个 helper 方法完成工作。
"""

from dataclasses import dataclass, field, replace

from ai_agent_playground.config import BaseAgentConfig
from ai_agent_playground.llm import LLMClient, get_client

from .config import CrewConfig
from .roles import (
    DeveloperAgent,    # AI 开发者——写代码
    DevOpsAgent,       # AI 运维——出部署方案
    ProductManagerAgent, # AI 产品经理——拆任务
    QAAgent,           # AI 测试——审查代码
)


@dataclass
class CrewResult:
    """
    一次完整 Crew 运行的结果。

    就像一个项目文件夹，里面有：
      - 需求文档（requirement）
      - 任务清单（tasks）
      - 代码文件（code）
      - 测试报告（qa_report）
      - 部署配置（devops_config）
    """
    requirement: str                                    # 用户原始需求
    tasks: list[dict] = field(default_factory=list)     # PM 拆出来的任务列表
    code: dict[str, str] = field(default_factory=dict)   # task_id → 生成的代码
    qa_report: str = ""                                  # QA 审查报告
    devops_config: str = ""                              # DevOps 部署配置


class Crew:
    """
    虚拟开发团队——"一句话需求进去，完整项目出来"。

    使用示例：
      crew = Crew()
      result = crew.run("做一个 Todo App 的 REST API")

    result.tasks → ["搭建项目", "实现增删改查", "加验证", "写测试"]
    result.code  → {"T-1": "import fastapi...", "T-2": "@app.post..."}
    result.qa_report → "发现 3 个安全问题..."
    result.devops_config → "FROM python:3.11..."

    这一切只用了 4 次 API 调用！不需要人类开发者写一行代码。
    """

    def __init__(self, config: CrewConfig | None = None, llm: LLMClient | None = None):
        # 配置（决定哪些阶段启用、用哪个模型）
        self.config = config or CrewConfig()
        # 共享的 AI 连接（四个 Agent 用同一根电话线）
        self.llm = llm or get_client()

        # ---- 创建四个团队成员 ----
        # 每个成员有不同的 system prompt（"你是产品经理" "你是开发者" ...）
        # _role_config 方法从 CrewConfig 里取出对应角色的提示词
        self.pm = ProductManagerAgent(self._role_config("pm_prompt"))
        self.dev = DeveloperAgent(self._role_config("dev_prompt"))
        self.qa = QAAgent(self._role_config("qa_prompt"))
        self.devops = DevOpsAgent(self._role_config("devops_prompt"))

    def run(self, requirement: str) -> CrewResult:
        """
        启动整个开发团队！按顺序执行四个阶段。

        就像一个项目启动会议：
          1. PM 说"我们要做这些"
          2. 开发者逐个实现
          3. QA 检查质量
          4. DevOps 准备部署

        整个过程是"串行"的——上一步没做完，下一步不会开始。
        这是最简单的多 Agent 协作模式，但已经非常有用。
        """
        result = CrewResult(requirement=requirement)

        # ============================================================
        #  阶段 1：产品经理拆任务
        # ============================================================
        # PM Agent 拿到一句话需求 → 拆成 3-5 个具体技术任务
        # 比如 "做电商网站" → ["用户注册", "商品列表", "购物车", "支付"]
        print("=" * 60)
        print("  Phase 1: PM Agent — Breaking down requirement")
        print("=" * 60)
        print(f"  Requirement: {requirement}\n")

        result.tasks = self.pm.run(requirement)
        for t in result.tasks:
            print(f"  [{t['id']}] {t['priority']:6s} | {t['title']}")
        print(f"\n  → {len(result.tasks)} tasks defined\n")

        if not result.tasks:
            print("  PM didn't generate any tasks. Stopping.")
            return result

        # ============================================================
        #  阶段 2：开发者逐个实现任务
        # ============================================================
        # Dev Agent 拿到每个任务 → 写代码
        # 每个任务是独立的——你可以并行处理，但这里我们串行（更稳定）
        print("=" * 60)
        print("  Phase 2: Dev Agent — Implementing tasks")
        print("=" * 60)

        for task in result.tasks:
            print(f"\n  [{task['id']}] {task['title']} ...", end=" ", flush=True)
            # 告诉 Dev："这里是标题，这里是描述，请写代码"
            code_result = self.dev.run({
                "title": task["title"],
                "description": task["description"],
            })
            result.code[task["id"]] = code_result["code"]
            print(f"done ({len(code_result['code'])} chars)")

        # 把所有的代码拼成一个大字符串（给 QA 和 DevOps 看的）
        all_code = "\n\n".join(
            f"## Task: {tid}\n{code}"
            for tid, code in result.code.items()
        )

        # ============================================================
        #  阶段 3：QA 审查（可选）
        # ============================================================
        # QA Agent 拿到全部代码 → 审查质量
        # 输出：严重问题（critical）、警告（warning）、建议（info）
        if self.config.enable_qa:
            print("\n" + "=" * 60)
            print("  Phase 3: QA Agent — Reviewing code")
            print("=" * 60)
            result.qa_report = self.qa.run(all_code)
            preview = result.qa_report[:200] + "..." if len(result.qa_report) > 200 else result.qa_report
            print(f"  {preview}\n")

        # ============================================================
        #  阶段 4：DevOps 出部署方案（可选）
        # ============================================================
        # DevOps Agent 拿到全部代码 → 生成 Dockerfile + docker-compose + 部署清单
        if self.config.enable_devops:
            print("=" * 60)
            print("  Phase 4: DevOps Agent — Deployment config")
            print("=" * 60)
            result.devops_config = self.devops.run(all_code)
            print("  Done.\n")

        return result

    def _role_config(self, prompt_field: str) -> CrewConfig:
        """
        创建一个角色专属的配置。

        所有角色共享同一个基础配置（model、max_tokens...），
        但 system_prompt 不同（PM 的 prompt 是"你是产品经理"，
        Dev 的 prompt 是"你是开发者"）。

        dataclasses.replace() 创建一个副本，只改指定字段。
        就像：复印一份表格，只改"岗位"那一栏，其他不变。
        """
        return replace(
            self.config,
            system_prompt=getattr(self.config, prompt_field),
        )
