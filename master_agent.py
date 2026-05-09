"""
Master agent orchestrator.

Exposes each specialist agent as a Claude tool and runs the tool-use agentic
loop until Claude decides the overall session goal is satisfied. Tasks are
processed sequentially via the MemoryStore task queue -- one at a time, in order.
"""

import json
from datetime import datetime
from pathlib import Path

import anthropic

from config import MODEL, MAX_TOKENS_MASTER, MEMORY_PATH
from memory.memory_store import MemoryStore
from agents import (
    AmazonAgent,
    DoorDashAgent,
    ShoppingAgent,
    PlannerAgent,
    DailyRoutineAgent,
)


AGENT_TOOLS = [
    {
        "name": "run_amazon_agent",
        "description": (
            "Delegate an Amazon shopping task to the AmazonAgent specialist. "
            "Use for product searches, price comparisons, deal hunting, and "
            "purchase guidance on Amazon."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Full description of the Amazon shopping task."},
                "notify_agent": {"type": "string", "description": "Optional. Agent name to forward the result to."},
            },
            "required": ["task"],
        },
    },
    {
        "name": "run_doordash_agent",
        "description": (
            "Delegate a food delivery task to the DoorDashAgent specialist. "
            "Use for restaurant discovery, meal recommendations, dietary filtering, "
            "and DoorDash order planning."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Full description of the food delivery task."},
                "notify_agent": {"type": "string", "description": "Optional. Agent name to forward the result to."},
            },
            "required": ["task"],
        },
    },
    {
        "name": "run_shopping_agent",
        "description": (
            "Delegate a general (non-Amazon) shopping task to the ShoppingAgent. "
            "Use for multi-platform price comparisons, budget planning, gift "
            "recommendations, coupon awareness, and wishlist management."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Full description of the shopping task."},
                "notify_agent": {"type": "string", "description": "Optional. Agent name to forward the result to."},
            },
            "required": ["task"],
        },
    },
    {
        "name": "run_planner_agent",
        "description": (
            "Delegate a scheduling or prioritisation task to the PlannerAgent. "
            "Use for building time-blocked daily schedules, applying the Eisenhower "
            "Matrix, resolving calendar conflicts, and productivity coaching."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Full description of the planning task."},
                "notify_agent": {"type": "string", "description": "Optional. Agent name to forward the result to."},
            },
            "required": ["task"],
        },
    },
    {
        "name": "run_daily_routine_agent",
        "description": (
            "Delegate a habit or routine task to the DailyRoutineAgent. "
            "Use for designing morning/evening rituals, habit stacking, streak "
            "tracking, daily check-ins, and routine optimisation."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "task": {"type": "string", "description": "Full description of the habit or routine task."},
                "notify_agent": {"type": "string", "description": "Optional. Agent name to forward the result to."},
            },
            "required": ["task"],
        },
    },
    {
        "name": "read_agent_messages",
        "description": (
            "Read any inter-agent messages waiting in an agent's inbox. "
            "Use this before dispatching a task to an agent so it can incorporate "
            "context forwarded by a previously executed agent."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "agent_name": {"type": "string", "description": "The agent whose inbox to read."},
            },
            "required": ["agent_name"],
        },
    },
    {
        "name": "get_queue_status",
        "description": "Return the current task queue status (queued, active, completed counts).",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
]


def _load_master_guidelines() -> str:
    path = Path("guidelines/master_guidelines.md")
    return path.read_text() if path.exists() else (
        "Orchestrate specialist agents in sequence. Never skip tasks. "
        "Verify each result before proceeding to the next."
    )


def _build_master_system(guidelines: str) -> list[dict]:
    return [
        {
            "type": "text",
            "text": guidelines,
            "cache_control": {"type": "ephemeral"},
        },
        {
            "type": "text",
            "text": (
                "\n\n## Your Role\n"
                "You are the **MasterAgent** orchestrator. You receive a session goal "
                "and decompose it into sequential sub-tasks, one per specialist agent. "
                "You MUST use the provided tools to delegate -- never answer domain "
                "questions yourself. Execute tasks in order; wait for each result "
                "before proceeding. When agents need to share information, use "
                "`notify_agent` in the tool call and `read_agent_messages` before "
                "the receiving agent's turn.\n\n"
                "Available specialists:\n"
                "- `run_amazon_agent` -- Amazon product search & purchase guidance\n"
                "- `run_doordash_agent` -- DoorDash restaurant discovery & order planning\n"
                "- `run_shopping_agent` -- General multi-platform shopping & gifting\n"
                "- `run_planner_agent` -- Day scheduling, priorities, calendar\n"
                "- `run_daily_routine_agent` -- Habits, streaks, morning/evening routines\n"
                "- `read_agent_messages` -- Check an agent's inbox before dispatching\n"
                "- `get_queue_status` -- Inspect the task queue at any point\n"
            ),
        },
    ]


class MasterAgent:
    def __init__(self, session_context: dict | None = None):
        self.client = anthropic.Anthropic()
        self.memory = MemoryStore(MEMORY_PATH)
        self._guidelines = _load_master_guidelines()
        self.context: dict = {
            "session_goal": "",
            "date": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
            **(session_context or {}),
        }
        self._agents = {
            "AmazonAgent": AmazonAgent(self.memory, self.client),
            "DoorDashAgent": DoorDashAgent(self.memory, self.client),
            "ShoppingAgent": ShoppingAgent(self.memory, self.client),
            "PlannerAgent": PlannerAgent(self.memory, self.client),
            "DailyRoutineAgent": DailyRoutineAgent(self.memory, self.client),
        }

    def _dispatch_agent(self, tool_name: str, inputs: dict) -> str:
        tool_to_agent = {
            "run_amazon_agent": "AmazonAgent",
            "run_doordash_agent": "DoorDashAgent",
            "run_shopping_agent": "ShoppingAgent",
            "run_planner_agent": "PlannerAgent",
            "run_daily_routine_agent": "DailyRoutineAgent",
        }
        agent_name = tool_to_agent[tool_name]
        agent = self._agents[agent_name]

        pending = self.memory.read_messages(agent_name, unread_only=True)
        exec_context = dict(self.context)
        if pending:
            exec_context["incoming_messages"] = pending

        task_text = inputs["task"]
        task_id = self.memory.enqueue_task({"agent": agent_name, "task": task_text})
        self.memory.get_next_task()

        try:
            result = agent.run_task(task_text, exec_context)
            self.memory.complete_task(task_id, result)
        except Exception as exc:
            error_msg = str(exc)
            self.memory.fail_task(task_id, error_msg)
            result = json.dumps({"error": error_msg, "agent": agent_name})

        notify = inputs.get("notify_agent", "").strip()
        if notify and notify in self._agents:
            self.memory.send_message(agent_name, notify, result)

        self.memory.log("MasterAgent", f"Dispatched {agent_name}: {task_text[:60]}")
        return result

    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        if tool_name in (
            "run_amazon_agent", "run_doordash_agent", "run_shopping_agent",
            "run_planner_agent", "run_daily_routine_agent",
        ):
            return self._dispatch_agent(tool_name, tool_input)
        if tool_name == "read_agent_messages":
            messages = self.memory.read_messages(tool_input["agent_name"], unread_only=True)
            return json.dumps(messages) if messages else "No unread messages."
        if tool_name == "get_queue_status":
            return json.dumps(self.memory.queue_status())
        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    def run(self, user_request: str) -> str:
        self.context["session_goal"] = user_request
        self.memory.set_context("session_goal", user_request)
        self.memory.log("MasterAgent", f"Session started: {user_request[:120]}")

        system = _build_master_system(self._guidelines)
        messages = [{"role": "user", "content": user_request}]

        while True:
            response = self.client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS_MASTER,
                system=system,
                tools=AGENT_TOOLS,
                messages=messages,
            )
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason == "end_turn":
                final = next((b.text for b in response.content if hasattr(b, "text")), "")
                self.memory.log("MasterAgent", "Session completed.")
                return final

            if response.stop_reason != "tool_use":
                return next(
                    (b.text for b in response.content if hasattr(b, "text")),
                    f"Stopped unexpectedly: {response.stop_reason}",
                )

            tool_results = []
            for block in response.content:
                if block.type != "tool_use":
                    continue
                result_text = self._execute_tool(block.name, block.input)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result_text}
                )
            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    import sys
    session_ctx: dict = {}
    for arg in sys.argv[1:]:
        if "=" in arg:
            k, v = arg.split("=", 1)
            session_ctx[k.strip()] = v.strip()

    master = MasterAgent(session_context=session_ctx)
    print("Multi-Agent Orchestrator ready. Type your request (Ctrl-C to exit).\n")
    while True:
        try:
            request = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye.")
            break
        if not request:
            continue
        print("\nMasterAgent: processing...\n")
        answer = master.run(request)
        print(f"MasterAgent:\n{answer}\n")
