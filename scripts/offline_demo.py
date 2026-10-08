"""
Offline demo: run the full MasterAgent -> specialist -> tool pipeline with a
scripted stand-in for Gemini. No API key, no network.

What is real: the orchestration loop, task queue, inter-agent messaging,
memory store, every tool call and its output, and the dashboard state.
What is scripted: the model's decisions (which tool to call with which
arguments) and its prose, which only reformats the real tool results.

    python scripts/offline_demo.py
    AGENT_MEMORY_DIR=memory/demo uvicorn dashboard.app:app --port 8765
"""

import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("AGENT_MEMORY_DIR", str(ROOT / "memory" / "demo"))

from config import MEMORY_PATH  # noqa: E402
from master_agent import MasterAgent  # noqa: E402
from memory.memory_store import MemoryStore  # noqa: E402
from tests.fakes import FakeClient, call_response, text_response  # noqa: E402

STUB = "[scripted stub, not Gemini]"


def tool_results(contents) -> dict:
    """Latest result per tool name from the function responses sent to the model."""
    out = {}
    for content in contents:
        for part in content.parts or []:
            if part.function_response:
                out[part.function_response.name] = part.function_response.response.get("result")
    return out


def summarise(render):
    return lambda contents: text_response(f"{STUB}\n{render(tool_results(contents))}")


def render_schedule(r):
    sched = r["build_schedule"]
    lines = [f"{b['start']}-{b['end']}  {b['label']}" for b in sched["blocks"]]
    if sched["unscheduled"]:
        lines.append("Did not fit: " + ", ".join(t["name"] for t in sched["unscheduled"]))
    return "\n".join(lines)


def render_lunch(r):
    est, when = r["estimate_order_total"], r["latest_order_time"]
    items = ", ".join(f"{l['quantity']}x {l['name']}" for l in est["lines"])
    return (
        f"{est['restaurant']}: {items}. Estimated total ${est['total']:.2f} "
        f"(subtotal ${est['subtotal']:.2f}). Order by {when['order_by']} for {when['meal_time']}. "
        f"{est['notice']}"
    )


def render_lamp(r):
    cmp = r["compare_prices"]
    offers = ", ".join(f"{o['retailer']} ${o['total']:.2f}" for o in cmp["offers"])
    return f"{cmp['product']}: {offers}. Cheapest: {cmp['cheapest']}. {cmp['notice']}"


PLAN_ARGS = {
    "day_start": "08:00",
    "day_end": "17:00",
    "tasks": [
        {"name": "Study: distributed systems", "minutes": 180, "priority": 1},
        {"name": "Gym", "minutes": 60, "priority": 2},
        {"name": "Laundry", "minutes": 30, "priority": 3},
    ],
    "fixed_events": [{"name": "Lunch", "start": "12:30", "end": "13:15"}],
}

MASTER_SCRIPT = [
    call_response("run_planner_agent", {"task": "Plan Saturday 08:00-17:00: 3h study, gym, laundry, lunch at 12:30", "notify_agent": "DoorDashAgent"}),
    call_response("read_agent_messages", {"agent_name": "DoorDashAgent"}),
    call_response("run_doordash_agent", {"task": "Vegan lunch delivered by 12:30, under 40 minutes delivery"}),
    call_response("run_amazon_agent", {"task": "Desk lamp under $40"}),
    text_response(f"{STUB}\nSaturday is planned, lunch is priced, and the lamp is compared. See each agent's result above."),
]

AGENT_SCRIPT = [
    # PlannerAgent
    call_response("build_schedule", PLAN_ARGS),
    summarise(render_schedule),
    # DoorDashAgent
    call_response("search_restaurants", {"dietary": ["vegan"], "max_delivery_minutes": 40}),
    call_response("estimate_order_total", {"restaurant_id": "r02", "items": [{"item_id": "r02-3", "quantity": 1}, {"item_id": "r02-4", "quantity": 1}], "tip_percent": 18}),
    call_response("latest_order_time", {"meal_time": "12:30", "restaurant_id": "r02"}),
    summarise(render_lunch),
    # AmazonAgent
    call_response("search_products", {"query": "desk lamp", "max_price": 40}),
    call_response("compare_prices", {"product_id": "p018"}),
    summarise(render_lamp),
]


def main(memory_path: str = MEMORY_PATH) -> str:
    memory_dir = Path(memory_path).parent
    if memory_dir.exists() and memory_dir.name == "demo":
        shutil.rmtree(memory_dir)  # start each demo from a clean slate
    client = FakeClient(agent_script=AGENT_SCRIPT, master_script=MASTER_SCRIPT, memory_text=f"# Memory\n- {STUB} memory entry")
    master = MasterAgent(session_context={"session_goal": "Plan my Saturday"}, client=client, memory=MemoryStore(memory_path))

    request = "Plan my Saturday (3h study, gym, laundry), get me a vegan lunch for 12:30, and find a desk lamp under $40."
    print(f"You: {request}\n")
    answer = master.run(request)

    for task in master.memory.get_task_history():
        print(f"--- {task['agent']} ({task['status']})")
        print(task.get("result") or task.get("error"))
        print()
    print(f"MasterAgent: {answer}\n")
    print("Tool calls made:")
    for name, agent in master._agents.items():
        for call in agent.tool_calls:
            print(f"  {name}.{call['tool']}  ok={call['ok']}")
    print(f"\nState written to {memory_path}")
    return answer


if __name__ == "__main__":
    main()
