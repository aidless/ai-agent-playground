"""Skills Bootstrapping — auto-generate missing tools from reflection insights.

When the agent reflects and discovers a capability gap ("I need X but don't
have it"), this module:

  1. Uses LLM to generate Python tool code
  2. Validates syntax via compile()
  3. Writes to the skills/ directory
  4. Registers in ToolRegistry for immediate use next time

This is the "HYPERAGENTS" paper's self-patching loop made concrete:
reflection → code generation → validation → registration → reuse.
"""

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

BOOTSTRAP_DIR = Path(__file__).resolve().parent.parent / "skills" / "bootstrapped"


BOOTSTRAP_PROMPT = (
    "You are a tool code generator. Generate a Python function that implements "
    "the requested capability. The function must:\n"
    "1. Be named with snake_case\n"
    "2. Accept (params: dict) -> str signature\n"
    "3. Be safe — no system calls, no file deletion, no network to internal hosts\n"
    "4. Return a string result\n"
    "5. Include a one-line docstring\n"
    "Output ONLY the Python function code, nothing else."
)


@dataclass
class BootstrappedTool:
    name: str
    code: str
    description: str
    created_at: str
    source_reflection: str
    validated: bool = False
    registered: bool = False
    error: str = ""


class BootstrapEngine:
    """Auto-generates missing tool code from reflection insights.

    Usage:
        engine = BootstrapEngine(llm_client, model="deepseek-chat")
        tool = await engine.generate_from_reflection(
            "I need a tool to parse markdown tables but don't have one",
            "markdown_table_parser",
        )
        engine.register_tool(tool, registry)
    """

    def __init__(self, client, model: str = "deepseek-chat"):
        self.client = client
        self.model = model
        self._tools: dict[str, BootstrappedTool] = {}
        self._load_existing()

    def _load_existing(self):
        BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
        for f in BOOTSTRAP_DIR.glob("*.json"):
            try:
                data = json.loads(f.read_text(encoding="utf-8"))
                self._tools[data["name"]] = BootstrappedTool(**data)
            except Exception:
                pass

    async def generate_from_reflection(self, reflection: str, suggested_name: str) -> BootstrappedTool:
        """Generate a tool from a reflection about missing capability."""
        user_prompt = (
            f"I need a tool called '{suggested_name}'. "
            f"Context from reflection: {reflection}\n"
            f"Generate the Python function implementation."
        )

        code = ""
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": BOOTSTRAP_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=1024,
                temperature=0.3,
            )
            code = response.choices[0].message.content or ""
        except Exception as e:
            logger.error("Bootstrap code generation failed: %s", e)
            return BootstrappedTool(
                name=suggested_name, code="", description=reflection[:200],
                created_at=datetime.now(timezone.utc).isoformat(),
                source_reflection=reflection, error=str(e),
            )

        # Extract code block if wrapped in markdown
        code_match = re.search(r"```(?:python)?\s*\n(.*?)```", code, re.DOTALL)
        if code_match:
            code = code_match.group(1).strip()
        else:
            code = code.strip()

        tool = BootstrappedTool(
            name=suggested_name,
            code=code,
            description=reflection[:200],
            created_at=datetime.now(timezone.utc).isoformat(),
            source_reflection=reflection,
        )

        # Validate syntax
        tool.validated = self._validate_syntax(code, suggested_name)
        if not tool.validated:
            tool.error = "Syntax validation failed"
            return tool

        # Persist
        self._save_tool(tool)
        logger.info("Bootstrapped tool generated: %s (%d chars, validated=%s)",
                     suggested_name, len(code), tool.validated)
        return tool

    def _validate_syntax(self, code: str, name: str) -> bool:
        """Validate Python syntax and check for unsafe patterns."""
        try:
            compile(code, f"<bootstrap:{name}>", "exec")
        except SyntaxError as e:
            logger.warning("Bootstrap syntax error in %s: %s", name, e)
            return False

        # AST-level safety check
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Block os.system / subprocess with shell=True
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Attribute):
                        if node.func.attr in ("system", "popen", "call"):
                            logger.warning("Bootstrap blocked unsafe call: %s", ast.dump(node))
                            return False
                # Block imports of dangerous modules
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in ("os", "subprocess", "shutil", "socket", "ctypes"):
                            logger.warning("Bootstrap blocked unsafe import: %s", alias.name)
                            return False
                if isinstance(node, ast.ImportFrom):
                    if node.module in ("os", "subprocess", "shutil", "socket", "ctypes"):
                        logger.warning("Bootstrap blocked unsafe import from: %s", node.module)
                        return False
        except SyntaxError:
            pass

        # Ensure it's a function with correct signature
        if "def " not in code or "params" not in code:
            logger.warning("Bootstrap code missing function or params parameter")
            return False

        return True

    def _save_tool(self, tool: BootstrappedTool):
        BOOTSTRAP_DIR.mkdir(parents=True, exist_ok=True)
        path = BOOTSTRAP_DIR / f"{tool.name}.json"
        path.write_text(json.dumps({
            "name": tool.name,
            "code": tool.code,
            "description": tool.description,
            "created_at": tool.created_at,
            "source_reflection": tool.source_reflection,
            "validated": tool.validated,
            "registered": tool.registered,
            "error": tool.error,
        }, indent=2, ensure_ascii=False), encoding="utf-8")
        self._tools[tool.name] = tool

    def register_tool(self, tool: BootstrappedTool, registry) -> bool:
        """Register a validated bootstrapped tool into the ToolRegistry.

        Wraps the generated `func(params: dict)` callable to match
        ToolRegistry's `func(**kwargs)` calling convention.
        """
        if not tool.validated or not tool.code:
            return False

        try:
            # Create a callable from the code string
            namespace = {}
            exec(tool.code, namespace)
            raw_func = None
            for name, obj in namespace.items():
                if callable(obj) and name != "__builtins__" and not name.startswith("_"):
                    raw_func = obj
                    break

            if raw_func is None:
                tool.error = "No callable found in generated code"
                return False

            # Wrap: convert kwargs dict to the params dict the function expects
            def _wrapped(**kwargs):
                return raw_func(kwargs)

            _wrapped.__name__ = tool.name

            # Register in the tool registry
            if hasattr(registry, "register"):
                registry.register(
                    tool.name,
                    tool.description,
                    {"properties": {}, "required": []},
                    _wrapped,
                )
                tool.registered = True
                self._save_tool(tool)
                logger.info("Tool registered: %s", tool.name)
                return True
            else:
                tool.error = "Registry does not support dynamic registration"
                return False
        except Exception as e:
            tool.error = str(e)
            logger.error("Failed to register bootstrapped tool %s: %s", tool.name, e)
            return False

    def list_bootstrapped(self) -> list[dict]:
        return [
            {"name": t.name, "validated": t.validated, "registered": t.registered}
            for t in self._tools.values()
        ]

    def get_tool(self, name: str) -> Optional[BootstrappedTool]:
        return self._tools.get(name)
