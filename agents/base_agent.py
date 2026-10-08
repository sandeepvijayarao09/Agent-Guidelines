import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from config import GUIDELINES_DIR, MAX_TOKENS_AGENT, MAX_TOOL_ROUNDS, MODEL
from memory.memory_store import MemoryStore
from tools.registry import ToolSpec


class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.

    Public entry point is `run_task(task, context)` which enforces the strict
    agent flow:
        1. execute()               -- domain work (implemented by subclass)
        2. _post_task_memory()     -- dedicated Gemini call to update {AgentName}-memory.md
        3. memory.log()            -- append to session log

    Subclasses must implement: name, description, domain_system_prompt, execute.
    They may override `tools` to expose deterministic Python functions to
    Gemini; _call_llm() then runs a function-calling loop over them.
    They must NOT call memory.log() or write memory themselves.
    """

    def __init__(self, memory: MemoryStore, client: genai.Client):
        self.memory = memory
        self.client = client
        self._guidelines_text = self._load_guidelines()
        self._memory_dir = Path(memory.store_path).parent
        self._memory_dir.mkdir(parents=True, exist_ok=True)
        self.tool_calls: list[dict] = []

    # ── Abstract interface ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def domain_system_prompt(self) -> str: ...

    @abstractmethod
    def execute(self, task: str, context: dict) -> str: ...

    @property
    def tools(self) -> list[ToolSpec]:
        """Tools this agent may call. Default: none (prompt-only)."""
        return []

    # ── Public entry point (enforced agent flow) ──────────────────────────────

    def run_task(self, task: str, context: dict) -> str:
        result = self.execute(task, context)
        self._post_task_memory(task, result)
        self.memory.log(self.name, f"Completed: {task[:80]}")
        return result

    # ── Markdown memory ───────────────────────────────────────────────────────

    @property
    def _md_memory_path(self) -> Path:
        return self._memory_dir / f"{self.name}-memory.md"

    def _load_md_memory(self) -> str:
        if self._md_memory_path.exists():
            return self._md_memory_path.read_text().strip()
        return ""

    def _save_md_memory(self, content: str) -> None:
        self._md_memory_path.write_text(content.strip() + "\n")

    def _post_task_memory(self, task: str, result: str) -> None:
        """Dedicated post-task call: rewrite the agent's markdown memory file."""
        current = self._load_md_memory()
        system = (
            f"You are **{self.name}**. Your only job is to update your personal memory file.\n\n"
            f"## Your Current Memory\n{current if current else '*(empty — first task)*'}\n\n"
            "Rules:\n"
            "- Rewrite the full memory in well-structured markdown.\n"
            "- Retain all important prior information unless clearly outdated.\n"
            "- Add what you learned: user preferences, decisions made, key context.\n"
            "- Reply with ONLY the updated markdown — no preamble, no commentary."
        )
        response = self.client.models.generate_content(
            model=MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=(
                        f"Task I just completed:\n{task}\n\n"
                        f"My response:\n{result[:1500]}\n\n"
                        "Write my updated memory file now."
                    ))],
                )
            ],
            config=types.GenerateContentConfig(
                system_instruction=system,
                max_output_tokens=1024,
            ),
        )
        updated = response.text or ""
        if updated.strip():
            self._save_md_memory(updated.strip())

    # ── System prompt ─────────────────────────────────────────────────────────

    def _load_guidelines(self) -> str:
        path = GUIDELINES_DIR / "agent_guidelines.md"
        return path.read_text() if path.exists() else (
            "Follow all instructions carefully. Produce complete, accurate output."
        )

    def _build_system(self) -> str:
        """Return a system instruction string for Gemini."""
        current_memory = self._load_md_memory()
        memory_section = (
            f"\n\n## Your Memory\n{current_memory}"
            if current_memory
            else "\n\n## Your Memory\n*(empty — no prior interactions recorded)*"
        )
        return (
            self._guidelines_text
            + f"\n\n## Your Identity\nYou are **{self.name}**.\n\n"
            + self.domain_system_prompt
            + memory_section
        )

    def _call_llm(self, messages: list[dict]) -> str:
        """
        Call Gemini and return the final text.

        If the agent has tools, this runs the same function-calling loop as
        MasterAgent: execute each requested tool locally, send the results
        back, and repeat until Gemini answers in plain text.
        """
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            if isinstance(content, str):
                contents.append(
                    types.Content(role=role, parts=[types.Part(text=content)])
                )

        specs = {spec.name: spec for spec in self.tools}
        config = types.GenerateContentConfig(
            system_instruction=self._build_system(),
            max_output_tokens=MAX_TOKENS_AGENT,
            tools=[types.Tool(function_declarations=[s.declaration() for s in specs.values()])] if specs else None,
        )
        self.tool_calls = []

        for _ in range(MAX_TOOL_ROUNDS + 1):
            response = self.client.models.generate_content(
                model=MODEL,
                contents=contents,
                config=config,
            )
            candidate = response.candidates[0] if response.candidates else None
            parts = (candidate.content.parts or []) if candidate and candidate.content else []
            function_calls = [p.function_call for p in parts if p.function_call]
            if not function_calls:
                return "".join(p.text for p in parts if p.text) or (response.text or "")

            contents.append(candidate.content)
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part(
                            function_response=types.FunctionResponse(
                                name=fc.name,
                                response=self._run_tool(specs, fc.name, dict(fc.args or {})),
                            )
                        )
                        for fc in function_calls
                    ],
                )
            )

        return f"[{self.name}] stopped after {MAX_TOOL_ROUNDS} tool rounds without a final answer."

    def _run_tool(self, specs: dict[str, ToolSpec], name: str, args: dict) -> dict:
        """Execute one tool call; errors go back to the model instead of crashing the agent."""
        spec = specs.get(name)
        if spec is None:
            output = {"error": f"Unknown tool: {name}"}
        else:
            try:
                output = {"result": spec.function(**args)}
            except (TypeError, ValueError, KeyError) as exc:
                output = {"error": f"{type(exc).__name__}: {exc}"}
        self.tool_calls.append({"tool": name, "args": args, "ok": "error" not in output})
        self.memory.log(self.name, f"Tool {name}({json.dumps(args)[:80]})" + ("" if "error" not in output else " -> error"))
        return output

    # ── MemoryStore helpers ───────────────────────────────────────────────────

    def _remember(self, key: str, value: Any) -> None:
        self.memory.set_agent_memory(self.name, key, value)

    def _recall(self, key: str, default: Any = None) -> Any:
        return self.memory.get_agent_memory(self.name, key, default)

    def _send_to(self, to_agent: str, message: Any) -> None:
        self.memory.send_message(self.name, to_agent, message)

    def _read_inbox(self) -> list:
        return self.memory.read_messages(self.name, unread_only=True)
