import anthropic
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from config import MODEL, MAX_TOKENS_AGENT
from memory.memory_store import MemoryStore


class BaseAgent(ABC):
    """
    Abstract base for all specialist agents.

    Subclasses define `name`, `description`, `domain_system_prompt`, and `execute`.
    Guidelines are loaded from disk and injected into the system prompt with
    prompt caching so repeated calls don't re-tokenise the static rules.
    """

    def __init__(self, memory: MemoryStore, client: anthropic.Anthropic):
        self.memory = memory
        self.client = client
        self._guidelines_text = self._load_guidelines()

    # ── Abstract interface ───────────────────────────────────────────────────

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used in logs and memory keys."""

    @property
    @abstractmethod
    def description(self) -> str:
        """One-sentence description shown to the master agent."""

    @property
    @abstractmethod
    def domain_system_prompt(self) -> str:
        """Domain-specific instructions appended after the shared guidelines."""

    @abstractmethod
    def execute(self, task: str, context: dict) -> str:
        """Run the task and return a plain-text or JSON-string result."""

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _load_guidelines(self) -> str:
        path = Path("guidelines/agent_guidelines.md")
        return path.read_text() if path.exists() else (
            "Follow all instructions carefully. Produce complete, accurate output."
        )

    def _build_system(self) -> list[dict]:
        """
        System prompt with two blocks:
        1. Shared guidelines — marked ephemeral so they are cached across calls.
        2. Domain-specific instructions — not cached (shorter, agent-unique).
        """
        return [
            {
                "type": "text",
                "text": self._guidelines_text,
                "cache_control": {"type": "ephemeral"},
            },
            {
                "type": "text",
                "text": (
                    f"\n\n## Your Identity\nYou are **{self.name}**.\n\n"
                    f"{self.domain_system_prompt}"
                ),
            },
        ]

    def _call_claude(self, messages: list[dict], extra_kwargs: dict | None = None) -> str:
        kwargs = dict(
            model=MODEL,
            max_tokens=MAX_TOKENS_AGENT,
            system=self._build_system(),
            messages=messages,
        )
        if extra_kwargs:
            kwargs.update(extra_kwargs)
        response = self.client.messages.create(**kwargs)
        return next(
            (b.text for b in response.content if b.type == "text"), ""
        )

    def _remember(self, key: str, value: Any) -> None:
        self.memory.set_agent_memory(self.name, key, value)

    def _recall(self, key: str, default: Any = None) -> Any:
        return self.memory.get_agent_memory(self.name, key, default)
