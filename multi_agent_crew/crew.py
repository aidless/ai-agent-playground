"""Crew orchestrator: chains PM → Dev → QA → DevOps in sequence.

Like generate() method: orchestrates multiple steps, each is a focused call.
"""

from dataclasses import dataclass, field

from ai_agent_playground.config import BaseAgentConfig
from ai_agent_playground.llm import LLMClient, get_client

from .config import CrewConfig
from .roles import DeveloperAgent, DevOpsAgent, ProductManagerAgent, QAAgent


@dataclass
class CrewResult:
    requirement: str
    tasks: list[dict] = field(default_factory=list)
    code: dict[str, str] = field(default_factory=dict)  # task_id → code
    qa_report: str = ""
    devops_config: str = ""


class Crew:
    """Orchestrates a team of AI agents through a sequential workflow.

    Workflow:
        1. PM Agent:   requirement → task list
        2. Dev Agent:  each task → code
        3. QA Agent:   all code → review report (optional)
        4. DevOps Agent: all code → deploy config (optional)
    """

    def __init__(self, config: CrewConfig | None = None, llm: LLMClient | None = None):
        self.config = config or CrewConfig()
        self.llm = llm or get_client()
        self.pm = ProductManagerAgent(self._role_config("pm_prompt"))
        self.dev = DeveloperAgent(self._role_config("dev_prompt"))
        self.qa = QAAgent(self._role_config("qa_prompt"))
        self.devops = DevOpsAgent(self._role_config("devops_prompt"))

    def run(self, requirement: str) -> CrewResult:
        """Execute the full crew workflow."""
        result = CrewResult(requirement=requirement)

        # Phase 1: PM breaks down requirement
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

        # Phase 2: Dev implements each task
        print("=" * 60)
        print("  Phase 2: Dev Agent — Implementing tasks")
        print("=" * 60)

        for task in result.tasks:
            print(f"\n  [{task['id']}] {task['title']} ...", end=" ", flush=True)
            code_result = self.dev.run({
                "title": task["title"],
                "description": task["description"],
            })
            result.code[task["id"]] = code_result["code"]
            print(f"done ({len(code_result['code'])} chars)")

        all_code = "\n\n".join(
            f"## Task: {tid}\n{code}"
            for tid, code in result.code.items()
        )

        # Phase 3: QA review (optional)
        if self.config.enable_qa:
            print("\n" + "=" * 60)
            print("  Phase 3: QA Agent — Reviewing code")
            print("=" * 60)
            result.qa_report = self.qa.run(all_code)
            print(f"  {result.qa_report[:200]}...\n" if len(result.qa_report) > 200
                  else f"  {result.qa_report}\n")

        # Phase 4: DevOps deploy config (optional)
        if self.config.enable_devops:
            print("=" * 60)
            print("  Phase 4: DevOps Agent — Deployment config")
            print("=" * 60)
            result.devops_config = self.devops.run(all_code)
            print("  Done.\n")

        return result

    def _role_config(self, prompt_field: str) -> CrewConfig:
        """Return a config with the role-specific system prompt."""
        from dataclasses import replace
        return replace(
            self.config,
            system_prompt=getattr(self.config, prompt_field),
        )
