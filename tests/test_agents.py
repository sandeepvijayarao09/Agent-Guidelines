import pytest

from agents import AmazonAgent, DailyRoutineAgent, DoorDashAgent, PlannerAgent, ShoppingAgent
from memory.memory_store import MemoryStore
from tests.fakes import FakeClient, call_response, function_responses, text_response

# (agent class, expected tool names, a tool call to script, a check on the tool's real output)
CASES = [
    (
        PlannerAgent,
        {"build_schedule", "date_info"},
        ("build_schedule", {"day_start": "09:00", "day_end": "12:00", "tasks": [{"name": "Deep work", "minutes": 90, "priority": 1}]}),
        lambda r: r["blocks"][0] == {"start": "09:00", "end": "10:30", "label": "Deep work", "kind": "task"},
    ),
    (
        DailyRoutineAgent,
        {"routine_timeline", "habit_streaks"},
        ("routine_timeline", {"anchor_time": "06:30", "steps": [{"name": "Stretch", "minutes": 15}, {"name": "Shower", "minutes": 10}]}),
        lambda r: r["ends_at"] == "06:55",
    ),
    (
        AmazonAgent,
        {"search_products", "compare_prices"},
        ("search_products", {"query": "air fryer"}),
        lambda r: r["products"][0]["best_retailer"] == "Amazon" and r["products"][0]["best_price"] == 89.99,
    ),
    (
        ShoppingAgent,
        {"search_products", "compare_prices", "allocate_budget"},
        ("compare_prices", {"product_id": "p010"}),
        lambda r: r["cheapest"] == "Walmart",
    ),
    (
        DoorDashAgent,
        {"search_restaurants", "estimate_order_total", "latest_order_time"},
        ("estimate_order_total", {"restaurant_id": "r05", "items": [{"item_id": "r05-1", "quantity": 1}], "tip_percent": 0, "tax_rate": 0}),
        lambda r: r["total"] == round(11.50 + 0.99 + 1.15, 2),
    ),
]


@pytest.fixture
def memory(tmp_path):
    return MemoryStore(str(tmp_path / "state.json"))


@pytest.mark.parametrize("agent_cls, tool_names, tool_call, check", CASES, ids=[c[0].__name__ for c in CASES])
def test_execute_runs_tool_loop_with_real_tool_output(memory, agent_cls, tool_names, tool_call, check):
    client = FakeClient(agent_script=[call_response(*tool_call), text_response("Here is your answer.")])
    agent = agent_cls(memory, client)

    result = agent.run_task("do the thing", {"session_goal": "test"})

    assert result == "Here is your answer."
    first, second = client.calls_of("agent")
    declared = {d.name for d in first["config"].tools[0].function_declarations}
    assert declared == tool_names
    (response,) = function_responses(second)
    assert response.name == tool_call[0]
    assert "error" not in response.response
    assert check(response.response["result"])
    assert agent.tool_calls == [{"tool": tool_call[0], "args": tool_call[1], "ok": True}]

    # Enforced flow: memory file rewritten, completion logged.
    md = memory.store_path.parent / f"{agent.name}-memory.md"
    assert md.read_text().startswith("# Memory")
    messages = [e["message"] for e in memory.get_log(50)]
    assert any(m.startswith(f"Tool {tool_call[0]}") for m in messages)
    assert messages[-1] == "Completed: do the thing"


def test_tool_errors_are_returned_to_the_model(memory):
    client = FakeClient(
        agent_script=[
            call_response("compare_prices", {"product_id": "does-not-exist"}),
            call_response("not_a_tool", {}),
            text_response("Sorry, I couldn't find that product."),
        ]
    )
    agent = ShoppingAgent(memory, client)
    assert agent.execute("compare", {}) == "Sorry, I couldn't find that product."
    calls = client.calls_of("agent")
    assert "No product" in function_responses(calls[1])[-1].response["error"]
    assert function_responses(calls[2])[-1].response == {"error": "Unknown tool: not_a_tool"}
    assert [c["ok"] for c in agent.tool_calls] == [False, False]


def test_tool_loop_is_bounded(memory, monkeypatch):
    import agents.base_agent as base

    monkeypatch.setattr(base, "MAX_TOOL_ROUNDS", 2)
    loop = [call_response("date_info", {"target_date": "2026-05-09"}) for _ in range(3)]
    agent = PlannerAgent(memory, FakeClient(agent_script=loop))
    assert "stopped after 2 tool rounds" in agent.execute("loop forever", {})


def test_plain_text_answer_without_tool_calls(memory):
    agent = DailyRoutineAgent(memory, FakeClient(agent_script=[text_response("Drink water first.")]))
    assert agent.execute("morning tip", {}) == "Drink water first."


def test_system_prompt_includes_guidelines_identity_and_memory(memory):
    agent = PlannerAgent(memory, FakeClient())
    (memory.store_path.parent / "PlannerAgent-memory.md").write_text("- Wakes at 06:30\n")
    system = agent._build_system()
    assert "Specialist Agent Guidelines" in system
    assert "You are **PlannerAgent**" in system
    assert "Wakes at 06:30" in system
