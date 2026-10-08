import json

import pytest

from master_agent import AGENT_TOOLS, MasterAgent
from memory.memory_store import MemoryStore
from tests.fakes import FakeClient, call_response, function_responses, text_response


@pytest.fixture
def memory(tmp_path):
    return MemoryStore(str(tmp_path / "state.json"))


def test_specialists_are_exposed_as_tools():
    names = {d.name for d in AGENT_TOOLS.function_declarations}
    assert names == {
        "run_amazon_agent",
        "run_doordash_agent",
        "run_shopping_agent",
        "run_planner_agent",
        "run_daily_routine_agent",
        "read_agent_messages",
        "get_queue_status",
    }


def test_delegates_to_specialist_and_forwards_result(memory):
    client = FakeClient(
        master_script=[
            call_response("run_planner_agent", {"task": "Plan my morning", "notify_agent": "DoorDashAgent"}),
            call_response("read_agent_messages", {"agent_name": "DoorDashAgent"}),
            call_response("get_queue_status", {}),
            text_response("Your morning is planned."),
        ],
        agent_script=[
            call_response("build_schedule", {"day_start": "08:00", "day_end": "12:00", "tasks": [{"name": "Study", "minutes": 60}]}),
            text_response("08:00-09:00 Study"),
        ],
    )
    master = MasterAgent(session_context={"session_goal": "test"}, client=client, memory=memory)

    assert master.run("Plan my morning") == "Your morning is planned."

    master_calls = client.calls_of("master")
    assert len(master_calls) == 4
    # Specialist result is fed back to the master as the function response.
    (planner_resp,) = function_responses(master_calls[1])[-1:]
    assert planner_resp.name == "run_planner_agent"
    assert planner_resp.response == {"result": "08:00-09:00 Study"}
    # notify_agent delivered the result to DoorDashAgent's inbox.
    inbox = json.loads(function_responses(master_calls[2])[-1].response["result"])
    assert inbox[0]["from"] == "PlannerAgent" and inbox[0]["message"] == "08:00-09:00 Study"
    status = json.loads(function_responses(master_calls[3])[-1].response["result"])
    assert status["completed"] == 1 and status["queued"] == 0

    (task,) = memory.get_task_history()
    assert task["agent"] == "PlannerAgent" and task["status"] == "completed"
    log = [e["message"] for e in memory.get_log(50)]
    assert "Dispatched PlannerAgent: Plan my morning" in log
    assert log[-1] == "Assistant: Your morning is planned."


def test_chat_history_persists_across_turns(memory):
    client = FakeClient(master_script=[text_response("Hi."), text_response("Still here.")])
    master = MasterAgent(client=client, memory=memory)
    master.run("hello")
    master.run("again")
    second_call = client.calls_of("master")[1]
    texts = [p.text for c in second_call["contents"] for p in c.parts if p.text]
    assert texts == ["hello", "Hi.", "again"]


def test_specialist_failure_is_recorded_and_reported(memory):
    class Broken(Exception):
        pass

    client = FakeClient(
        master_script=[call_response("run_amazon_agent", {"task": "find headphones"}), text_response("That failed.")],
    )
    master = MasterAgent(client=client, memory=memory)

    def explode(task, context):
        raise Broken("quota exceeded")

    master._agents["AmazonAgent"].run_task = explode
    assert master.run("headphones") == "That failed."
    resp = function_responses(client.calls_of("master")[1])[-1].response["result"]
    assert json.loads(resp) == {"error": "quota exceeded", "agent": "AmazonAgent"}
    assert memory.get_task_history()[-1]["status"] == "failed"


def test_unknown_tool(memory):
    master = MasterAgent(client=FakeClient(), memory=memory)
    assert json.loads(master._execute_tool("nope", {})) == {"error": "Unknown tool: nope"}
