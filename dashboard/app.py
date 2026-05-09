"""
Dashboard server -- real-time visual monitor for the agent orchestration system.

Run from the project root:
    uvicorn dashboard.app:app --host 0.0.0.0 --port 8765 --reload

Or directly:
    python dashboard/app.py
"""

import asyncio
import json
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

ROOT = Path(__file__).parent.parent
MEMORY_DIR = ROOT / "memory"
STATE_PATH = MEMORY_DIR / "state.json"
INDEX_HTML = Path(__file__).parent / "index.html"

AGENTS = [
    "AmazonAgent",
    "DoorDashAgent",
    "ShoppingAgent",
    "PlannerAgent",
    "DailyRoutineAgent",
]

POLL_INTERVAL = 0.5

app = FastAPI(title="Agent Dashboard")


class ConnectionManager:
    def __init__(self) -> None:
        self._connections: list[WebSocket] = []

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        self._connections.append(ws)

    def disconnect(self, ws: WebSocket) -> None:
        if ws in self._connections:
            self._connections.remove(ws)

    async def broadcast(self, data: dict) -> None:
        dead: list[WebSocket] = []
        for ws in list(self._connections):
            try:
                await ws.send_text(json.dumps(data))
            except Exception:
                dead.append(ws)
        for ws in dead:
            if ws in self._connections:
                self._connections.remove(ws)

    @property
    def count(self) -> int:
        return len(self._connections)


manager = ConnectionManager()


def build_state() -> dict:
    try:
        raw = json.loads(STATE_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        raw = {}

    active_task = raw.get("active_task")
    active_agent = active_task.get("agent") if active_task else None

    completed_by_agent: dict[str, list] = {}
    for t in raw.get("completed_tasks", []):
        completed_by_agent.setdefault(t.get("agent", ""), []).append(t)

    queued_agents = {t.get("agent") for t in raw.get("task_queue", [])}

    agents = []
    for name in AGENTS:
        if name == active_agent:
            status = "active"
        elif name in queued_agents:
            status = "queued"
        elif name in completed_by_agent:
            status = "completed"
        else:
            status = "idle"

        md_path = MEMORY_DIR / f"{name}-memory.md"
        agents.append(
            {
                "name": name,
                "status": status,
                "task_count": len(completed_by_agent.get(name, [])),
                "has_memory": md_path.exists(),
                "memory_bytes": md_path.stat().st_size if md_path.exists() else 0,
            }
        )

    return {
        "agents": agents,
        "active_task": active_task,
        "task_queue": raw.get("task_queue", []),
        "completed_tasks": raw.get("completed_tasks", [])[-15:],
        "session_log": raw.get("session_log", [])[-60:],
        "agent_messages": {
            k: v for k, v in raw.get("agent_messages", {}).items() if v
        },
        "master_status": "running" if active_task else "idle",
        "updated_at": raw.get("updated_at", ""),
        "queue_status": {
            "queued": len(raw.get("task_queue", [])),
            "completed": len(raw.get("completed_tasks", [])),
        },
    }


@app.get("/", response_class=HTMLResponse)
def serve_ui() -> HTMLResponse:
    return HTMLResponse(content=INDEX_HTML.read_text())


@app.get("/api/state")
def get_state() -> dict:
    return build_state()


@app.get("/api/memory/{agent_name}")
def get_agent_memory(agent_name: str) -> dict:
    if agent_name not in AGENTS:
        return {"error": "Unknown agent", "content": ""}
    path = MEMORY_DIR / f"{agent_name}-memory.md"
    if not path.exists():
        return {"content": "*(no memory recorded yet)*", "bytes": 0}
    content = path.read_text()
    return {"content": content, "bytes": len(content.encode())}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket) -> None:
    await manager.connect(ws)
    await ws.send_text(json.dumps(build_state()))
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(ws)


@app.on_event("startup")
async def start_poller() -> None:
    asyncio.create_task(_poll_state())


async def _poll_state() -> None:
    last_ts = ""
    while True:
        await asyncio.sleep(POLL_INTERVAL)
        try:
            state = build_state()
            if state["updated_at"] != last_ts and manager.count > 0:
                last_ts = state["updated_at"]
                await manager.broadcast(state)
        except Exception:
            pass


if __name__ == "__main__":
    uvicorn.run("dashboard.app:app", host="0.0.0.0", port=8765, reload=True)
