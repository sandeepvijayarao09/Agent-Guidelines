import json

from fastapi.testclient import TestClient

import dashboard.app as dash


def test_dashboard_serves_ui_and_state(tmp_path, monkeypatch):
    state = {
        "task_queue": [{"id": "b", "agent": "DoorDashAgent", "task": "lunch"}],
        "active_task": {"id": "a", "agent": "PlannerAgent", "task": "plan"},
        "completed_tasks": [{"id": "z", "agent": "AmazonAgent", "status": "completed"}],
        "session_log": [],
        "updated_at": "2026-05-09T10:00:00+00:00",
    }
    (tmp_path / "state.json").write_text(json.dumps(state))
    (tmp_path / "PlannerAgent-memory.md").write_text("# Planner memory\n")
    monkeypatch.setattr(dash, "STATE_PATH", tmp_path / "state.json")
    monkeypatch.setattr(dash, "MEMORY_DIR", tmp_path)

    client = TestClient(dash.app)
    assert "<html" in client.get("/").text.lower()

    data = client.get("/api/state").json()
    status = {a["name"]: a["status"] for a in data["agents"]}
    assert status == {
        "AmazonAgent": "completed",
        "DoorDashAgent": "queued",
        "ShoppingAgent": "idle",
        "PlannerAgent": "active",
        "DailyRoutineAgent": "idle",
    }
    assert data["master_status"] == "running"
    assert client.get("/api/memory/PlannerAgent").json()["content"] == "# Planner memory\n"
    assert client.get("/api/memory/Nobody").json()["error"] == "Unknown agent"


def test_dashboard_handles_missing_state(tmp_path, monkeypatch):
    monkeypatch.setattr(dash, "STATE_PATH", tmp_path / "missing.json")
    data = TestClient(dash.app).get("/api/state").json()
    assert data["master_status"] == "idle" and len(data["agents"]) == 5
