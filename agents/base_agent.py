import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types

from config import MODEL, MAX_TOKENS_AGENT
from memory.memory_store import MemoryStore


class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.

    Public entry point is `run_task(task, context)` which enforces the strict
    agent flow:
        1. execute()               -- domain work (implemented by subclass)
        2. _post_task_memory()     -- dedicated Gemini call to update {AgentName}-memory.md
        3. memory.log()            -- append to session log

    Subclasses must implement: name, description, domain_system_prompt, execute.
    They must NOT call memory.log() or write memory themselves.
    """

    def __init__(self, memory: MemoryStore, client: genai.Client):
        self.memory = memory
        self.client = client
        self._guidelines_text = self._load_guidelines()
        Path("memory").mkdir(exist_ok=True)

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

    # ── Public entry point (enforced agent flow) ──────────────────────────────

    def run_task(self, task: str, context: dict) -> str:
        result = self.execute(task, context)
        self._post_task_memory(task, result)
        self.memory.log(self.name, f"Completed: {task[:80]}")
        return result

    # ── Markdown memory ───────────────────────────────────────────────────────

    @property
    def _md_memory_path(self) -> Path:
        return Path(f"memory/{self.name}-memory.md")

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
        path = Path("guidelines/agent_guidelines.md")
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

    def _call_claude(self, messages: list[dict], extra_kwargs: dict | None = None) -> str:
        """Call Gemini with the given messages and return the text response."""
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            if isinstance(content, str):
                contents.append(
                    types.Content(role=role, parts=[types.Part(text=content)])
                )

        config = types.GenerateContentConfig(
            system_instruction=self._build_system(),
            max_output_tokens=MAX_TOKENS_AGENT,
        )
        response = self.client.models.generate_content(
            model=MODEL,
            contents=contents,
            config=config,
        )
        return response.text or ""

    # ── MemoryStore helpers ───────────────────────────────────────────────────

    def _remember(self, key: str, value: Any) -> None:
        self.memory.set_agent_memory(self.name, key, value)

    def _recall(self, key: str, default: Any = None) -> Any:
        return self.memory.get_agent_memory(self.name, key, default)

    def _send_to(self, to_agent: str, message: Any) -> None:
        self.memory.send_message(self.name, to_agent, message)

    def _read_inbox(self) -> list:
        return self.memory.read_messages(self.name, unread_only=True)
