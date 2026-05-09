"""
Master agent orchestrator.

Exposes each specialist agent as a Gemini tool and runs the function-calling
agentic loop until Gemini decides the overall session goal is satisfied.
Tasks are processed sequentially via the MemoryStore task queue.
"""

import json
import os
from datetime import datetime, timezone
from pathlib import Path

from google import genai
from google.genai import types

from config import MODEL, MAX_TOKENS_MASTER, MEMORY_PATH
from memory.memory_store import MemoryStore
from agents import (
    AmazonAgent,
    DoorDashAgent,
    ShoppingAgent,
    PlannerAgent,
    DailyRoutineAgent,
)


# ── Tool schemas exposed to the master Gemini call ───────────────────────────

AGENT_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="run_amazon_agent",
            description=(
                "Delegate an Amazon shopping task to the AmazonAgent specialist. "
                "Use for product searches, price comparisons, deal hunting, and "
                "purchase guidance on Amazon."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "task": types.Schema(
                        type=types.Type.STRING,
                        description="Full description of the Amazon shopping task.",
                    ),
                    "notify_agent": types.Schema(
                        type=types.Type.STRING,
                        description=(
                            "Optional. Name of another agent to send the result to "
                            "(e.g. 'PlannerAgent'). Leave empty if not needed."
                        ),
                    ),
                },
                required=["task"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_doordash_agent",
            description=(
                "Delegate a food delivery task to the DoorDashAgent specialist. "
                "Use for restaurant discovery, meal recommendations, dietary filtering, "
                "and DoorDash order planning."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "task": types.Schema(
                        type=types.Type.STRING,
                        description="Full description of the food delivery task.",
                    ),
                    "notify_agent": types.Schema(
                        type=types.Type.STRING,
                        description="Optional. Agent name to forward the result to.",
                    ),
                },
                required=["task"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_shopping_agent",
            description=(
                "Delegate a general (non-Amazon) shopping task to the ShoppingAgent. "
                "Use for multi-platform price comparisons, budget planning, gift "
                "recommendations, coupon awareness, and wishlist management."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "task": types.Schema(
                        type=types.Type.STRING,
                        description="Full description of the shopping task.",
                    ),
                    "notify_agent": types.Schema(
                        type=types.Type.STRING,
                        description="Optional. Agent name to forward the result to.",
                    ),
                },
                required=["task"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_planner_agent",
            description=(
                "Delegate a scheduling or prioritisation task to the PlannerAgent. "
                "Use for building time-blocked daily schedules, applying the Eisenhower "
                "Matrix, resolving calendar conflicts, and productivity coaching."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "task": types.Schema(
                        type=types.Type.STRING,
                        description="Full description of the planning task.",
                    ),
                    "notify_agent": types.Schema(
                        type=types.Type.STRING,
                        description="Optional. Agent name to forward the result to.",
                    ),
                },
                required=["task"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_daily_routine_agent",
            description=(
                "Delegate a habit or routine task to the DailyRoutineAgent. "
                "Use for designing morning/evening rituals, habit stacking, streak "
                "tracking, daily check-ins, and routine optimisation."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "task": types.Schema(
                        type=types.Type.STRING,
                        description="Full description of the habit or routine task.",
                    ),
                    "notify_agent": types.Schema(
                        type=types.Type.STRING,
                        description="Optional. Agent name to forward the result to.",
                    ),
                },
                required=["task"],
            ),
        ),
        types.FunctionDeclaration(
            name="read_agent_messages",
            description=(
                "Read any inter-agent messages waiting in an agent's inbox. "
                "Use this before dispatching a task to an agent so it can incorporate "
                "context forwarded by a previously executed agent."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "agent_name": types.Schema(
                        type=types.Type.STRING,
                        description="The agent whose inbox to read (e.g. 'PlannerAgent').",
                    ),
                },
                required=["agent_name"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_queue_status",
            description="Return the current task queue status (queued, active, completed counts).",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
    ]
)


# ── Master agent guidelines ──────────────────────────────────────────────────

def _load_master_guidelines() -> str:
    path = Path("guidelines/master_guidelines.md")
    return path.read_text() if path.exists() else (
        "Orchestrate specialist agents in sequence. Never skip tasks. "
        "Verify each result before proceeding to the next."
    )


def _build_master_system(guidelines: str) -> str:
    return (
        guidelines
        + "\n\n## Your Role\n"
        "You are the **MasterAgent** orchestrator. You receive a session goal "
        "and decompose it into sequential sub-tasks, one per specialist agent. "
        "You MUST use the provided tools to delegate — never answer domain "
        "questions yourself. Execute tasks in order; wait for each result "
        "before proceeding. When agents need to share information, use "
        "`notify_agent` in the tool call and `read_agent_messages` before "
        "the receiving agent's turn.\n\n"
        "Available specialists:\n"
        "- `run_amazon_agent` — Amazon product search & purchase guidance\n"
        "- `run_doordash_agent` — DoorDash restaurant discovery & order planning\n"
        "- `run_shopping_agent` — General multi-platform shopping & gifting\n"
        "- `run_planner_agent` — Day scheduling, priorities, calendar\n"
        "- `run_daily_routine_agent` — Habits, streaks, morning/evening routines\n"
        "- `read_agent_messages` — Check an agent's inbox before dispatching\n"
        "- `get_queue_status` — Inspect the task queue at any point\n"
    )


# ── Master agent ─────────────────────────────────────────────────────────────

class MasterAgent:
    def __init__(self, session_context: dict | None = None):
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))
        self.memory = MemoryStore(MEMORY_PATH)
        self._guidelines = _load_master_guidelines()

        self.context: dict = {
            "session_goal": "",
            "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            **(session_context or {}),
        }

        # Persistent chat history — survives across multiple run() calls
        self._chat_history: list[types.Content] = []
        self._config = types.GenerateContentConfig(
            system_instruction=_build_master_system(self._guidelines),
            max_output_tokens=MAX_TOKENS_MASTER,
            tools=[AGENT_TOOLS],
        )

        self._agents = {
            "AmazonAgent": AmazonAgent(self.memory, self.client),
            "DoorDashAgent": DoorDashAgent(self.memory, self.client),
            "ShoppingAgent": ShoppingAgent(self.memory, self.client),
            "PlannerAgent": PlannerAgent(self.memory, self.client),
            "DailyRoutineAgent": DailyRoutineAgent(self.memory, self.client),
        }

    # ── Internal tool dispatch ───────────────────────────────────────────────

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
            "run_amazon_agent",
            "run_doordash_agent",
            "run_shopping_agent",
            "run_planner_agent",
            "run_daily_routine_agent",
        ):
            return self._dispatch_agent(tool_name, tool_input)

        if tool_name == "read_agent_messages":
            agent_name = tool_input["agent_name"]
            messages = self.memory.read_messages(agent_name, unread_only=True)
            return json.dumps(messages) if messages else "No unread messages."

        if tool_name == "get_queue_status":
            return json.dumps(self.memory.queue_status())

        return json.dumps({"error": f"Unknown tool: {tool_name}"})

    # ── Agentic loop ─────────────────────────────────────────────────────────

    def run(self, user_request: str) -> str:
        """Append user message to chat history and run the agentic loop for this turn."""
        self.memory.log("MasterAgent", f"User: {user_request[:120]}")

        self._chat_history.append(
            types.Content(role="user", parts=[types.Part(text=user_request)])
        )

        while True:
            response = self.client.models.generate_content(
                model=MODEL,
                contents=self._chat_history,
                config=self._config,
            )

            candidate = response.candidates[0]
            self._chat_history.append(candidate.content)

            function_calls = [
                p for p in candidate.content.parts if p.function_call
            ]

            if not function_calls:
                final = "".join(
                    p.text for p in candidate.content.parts if p.text
                )
                self.memory.log("MasterAgent", f"Assistant: {final[:120]}")
                return final

            response_parts: list[types.Part] = []
            for part in function_calls:
                fc = part.function_call
                result_text = self._execute_tool(fc.name, dict(fc.args))
                response_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=fc.name,
                            response={"result": result_text},
                        )
                    )
                )

            self._chat_history.append(
                types.Content(role="user", parts=response_parts)
            )


# ── CLI entry point ──────────────────────────────────────────────────────────

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
