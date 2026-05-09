import json
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import uuid


class MemoryStore:
    """Persistent JSON-based memory for long-running agent sessions."""

    def __init__(self, store_path: str = "memory/state.json"):
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            with open(self.store_path) as f:
                self._state = json.load(f)
        else:
            self._state = {
                "task_queue": [],
                "completed_tasks": [],
                "active_task": None,
                "context": {},
                "agent_memory": {},
                "session_log": [],
                "created_at": datetime.utcnow().isoformat(),
                "updated_at": datetime.utcnow().isoformat(),
            }
            self._save()

    def _save(self) -> None:
        self._state["updated_at"] = datetime.utcnow().isoformat()
        with open(self.store_path, "w") as f:
            json.dump(self._state, f, indent=2)

    # ── Task Queue ───────────────────────────────────────────────────────────

    def enqueue_task(self, task: dict, task_id: Optional[str] = None) -> str:
        """Add a task to the end of the sequential queue."""
        task_id = task_id or str(uuid.uuid4())[:8]
        record = {
            "id": task_id,
            "status": "queued",
            "created_at": datetime.utcnow().isoformat(),
            **task,
        }
        self._state["task_queue"].append(record)
        self._save()
        return task_id

    def get_next_task(self) -> Optional[dict]:
        """Pop the front of the queue and mark it active."""
        if not self._state["task_queue"]:
            return None
        task = self._state["task_queue"].pop(0)
        task["status"] = "in_progress"
        task["started_at"] = datetime.utcnow().isoformat()
        self._state["active_task"] = task
        self._save()
        return task

    def complete_task(self, task_id: str, result: Any) -> None:
        active = self._state.get("active_task")
        if active and active["id"] == task_id:
            active.update(
                status="completed",
                result=result,
                completed_at=datetime.utcnow().isoformat(),
            )
            self._state["completed_tasks"].append(active)
            self._state["active_task"] = None
            self._save()

    def fail_task(self, task_id: str, error: str) -> None:
        active = self._state.get("active_task")
        if active and active["id"] == task_id:
            active.update(
                status="failed",
                error=error,
                failed_at=datetime.utcnow().isoformat(),
            )
            self._state["completed_tasks"].append(active)
            self._state["active_task"] = None
            self._save()

    def queue_status(self) -> dict:
        return {
            "queued": len(self._state["task_queue"]),
            "active": self._state.get("active_task"),
            "completed": len(self._state["completed_tasks"]),
        }

    # ── Shared Context ───────────────────────────────────────────────────────

    def set_context(self, key: str, value: Any) -> None:
        self._state["context"][key] = value
        self._save()

    def get_context(self, key: str, default: Any = None) -> Any:
        return self._state["context"].get(key, default)

    # ── Per-Agent Memory ─────────────────────────────────────────────────────

    def set_agent_memory(self, agent_name: str, key: str, value: Any) -> None:
        self._state["agent_memory"].setdefault(agent_name, {})[key] = value
        self._save()

    def get_agent_memory(self, agent_name: str, key: str, default: Any = None) -> Any:
        return self._state["agent_memory"].get(agent_name, {}).get(key, default)

    def get_all_agent_memory(self, agent_name: str) -> dict:
        return self._state["agent_memory"].get(agent_name, {})

    # ── Session Log ──────────────────────────────────────────────────────────

    def log(self, source: str, message: str) -> None:
        self._state["session_log"].append(
            {
                "timestamp": datetime.utcnow().isoformat(),
                "source": source,
                "message": message,
            }
        )
        self._save()

    def get_log(self, limit: int = 20) -> list:
        return self._state["session_log"][-limit:]

    def get_task_history(self, limit: int = 10) -> list:
        return self._state["completed_tasks"][-limit:]

    def reset_queue(self) -> None:
        """Clear the task queue (completed tasks are preserved)."""
        self._state["task_queue"] = []
        self._state["active_task"] = None
        self._save()

    # ── Inter-Agent Messaging ────────────────────────────────────────────────

    def send_message(self, from_agent: str, to_agent: str, message: Any) -> None:
        """Post a message from one agent into another agent's inbox."""
        inbox = self._state.setdefault("agent_messages", {})
        inbox.setdefault(to_agent, []).append(
            {
                "from": from_agent,
                "message": message,
                "timestamp": datetime.utcnow().isoformat(),
                "read": False,
            }
        )
        self._save()

    def read_messages(self, agent_name: str, unread_only: bool = True) -> list:
        """Return messages in an agent's inbox and mark them as read."""
        inbox = self._state.get("agent_messages", {}).get(agent_name, [])
        results = [m for m in inbox if not m["read"]] if unread_only else list(inbox)
        for m in inbox:
            m["read"] = True
        self._save()
        return results

    def clear_inbox(self, agent_name: str) -> None:
        self._state.get("agent_messages", {}).pop(agent_name, None)
        self._save()
