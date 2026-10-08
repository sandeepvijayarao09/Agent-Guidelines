import json

from memory.memory_store import MemoryStore


def test_task_lifecycle_persists_to_disk(tmp_path):
    path = tmp_path / "state.json"
    store = MemoryStore(str(path))
    first = store.enqueue_task({"agent": "PlannerAgent", "task": "plan"})
    second = store.enqueue_task({"agent": "ShoppingAgent", "task": "shop"})
    assert store.queue_status()["queued"] == 2

    assert store.get_next_task()["id"] == first
    store.complete_task(first, "done")
    assert store.get_next_task()["id"] == second
    store.fail_task(second, "boom")

    reloaded = MemoryStore(str(path))
    history = reloaded.get_task_history()
    assert [(t["id"], t["status"]) for t in history] == [(first, "completed"), (second, "failed")]
    assert history[0]["result"] == "done" and history[1]["error"] == "boom"
    assert reloaded.queue_status() == {"queued": 0, "active": None, "completed": 2}
    assert json.loads(path.read_text())["updated_at"]


def test_complete_ignores_wrong_task_id(tmp_path):
    store = MemoryStore(str(tmp_path / "s.json"))
    task_id = store.enqueue_task({"agent": "A", "task": "t"})
    store.get_next_task()
    store.complete_task("other", "x")
    assert store.queue_status()["active"]["id"] == task_id


def test_messages_are_marked_read(tmp_path):
    store = MemoryStore(str(tmp_path / "s.json"))
    store.send_message("PlannerAgent", "DoorDashAgent", "lunch at 12:30")
    unread = store.read_messages("DoorDashAgent")
    assert [m["message"] for m in unread] == ["lunch at 12:30"]
    assert store.read_messages("DoorDashAgent") == []
    assert len(store.read_messages("DoorDashAgent", unread_only=False)) == 1
    store.clear_inbox("DoorDashAgent")
    assert store.read_messages("DoorDashAgent", unread_only=False) == []


def test_context_agent_memory_and_log(tmp_path):
    store = MemoryStore(str(tmp_path / "s.json"))
    store.set_context("city", "Boston")
    store.set_agent_memory("PlannerAgent", "wake", "06:30")
    for i in range(25):
        store.log("Test", f"entry {i}")
    assert store.get_context("city") == "Boston"
    assert store.get_agent_memory("PlannerAgent", "wake") == "06:30"
    assert store.get_agent_memory("PlannerAgent", "missing", "d") == "d"
    assert store.get_all_agent_memory("PlannerAgent") == {"wake": "06:30"}
    log = store.get_log()
    assert len(log) == 20 and log[-1]["message"] == "entry 24"
