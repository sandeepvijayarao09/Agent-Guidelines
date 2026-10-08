import json

from scripts.offline_demo import STUB, main


def test_offline_demo_runs_end_to_end(tmp_path, capsys):
    path = tmp_path / "demo" / "state.json"
    answer = main(str(path))
    assert answer.startswith(STUB)

    state = json.loads(path.read_text())
    agents = [(t["agent"], t["status"]) for t in state["completed_tasks"]]
    assert agents == [("PlannerAgent", "completed"), ("DoorDashAgent", "completed"), ("AmazonAgent", "completed")]
    out = capsys.readouterr().out
    assert "ok=False" not in out
    assert "Brightdesk LED Desk Lamp: Walmart $34.99" in out
