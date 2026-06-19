# Agent-Guidelines

> A multi-agent orchestrator built on Gemini function calling, with persistent memory, guideline-driven behavior, and a live web dashboard.

## Overview

This project implements a **master agent** that decomposes a user's request into sub-tasks and delegates each one to a specialist agent. The master uses Gemini's function-calling loop to decide which specialist to invoke, runs tasks sequentially through a task queue, and lets agents pass context to one another via an inbox/message system. Agent and orchestrator behavior is governed by markdown **guideline** files, and all state is persisted to a JSON store that a FastAPI dashboard renders in real time.

## How It Works

1. The `MasterAgent` receives a request and runs an agentic loop against Gemini, with each specialist exposed as a callable tool (`run_amazon_agent`, `run_doordash_agent`, etc.).
2. For each delegated task the master enqueues it, dispatches to the specialist, and stores the result in the shared `MemoryStore`.
3. Specialists extend `BaseAgent`, which enforces a fixed flow: run the domain task, then make a dedicated Gemini call to rewrite that agent's personal markdown memory file, then append to the session log.
4. Agents can forward results to one another with `notify_agent`; the receiving agent reads its inbox before its turn.
5. The dashboard polls the JSON state and pushes updates to the browser over a WebSocket.

## Specialist Agents

| Agent | Domain |
| --- | --- |
| `AmazonAgent` | Amazon product search, price comparison, and purchase guidance |
| `DoorDashAgent` | Restaurant discovery and food-delivery order planning |
| `ShoppingAgent` | General multi-platform shopping, budgeting, and gifting |
| `PlannerAgent` | Day scheduling, prioritization, and calendar planning |
| `DailyRoutineAgent` | Habits, streaks, and morning/evening routines |

## Features

- Gemini function-calling orchestration loop with sequential task execution
- Persistent JSON memory store: task queue, completed tasks, session log, and inter-agent messages
- Per-agent markdown memory files that are rewritten after every completed task
- Guideline-driven behavior via `guidelines/master_guidelines.md` and `guidelines/agent_guidelines.md`
- Inter-agent messaging (`notify_agent` / `read_agent_messages`)
- Real-time FastAPI + WebSocket dashboard showing agent status, the task queue, and memory
- Dev container configured for Python 3.12 with the dashboard port forwarded

## Tech Stack

- Python 3.12
- [`google-genai`](https://pypi.org/project/google-genai/) — Gemini API client (model: `gemini-2.5-flash`)
- FastAPI + Uvicorn — dashboard server and WebSocket
- `python-dotenv` — environment configuration

## Prerequisites

- Python 3.12+
- A Google Gemini API key

## Setup

```bash
pip install -r requirements.txt
export GOOGLE_API_KEY="your-gemini-api-key"
```

## Usage

Run the interactive orchestrator from the project root:

```bash
python master_agent.py
```

You can pass session context as `key=value` arguments, for example:

```bash
python master_agent.py session_goal="Plan my day and order lunch"
```

Then enter requests at the `You:` prompt.

### Dashboard

Launch the live monitor (from the project root) and open it in a browser:

```bash
uvicorn dashboard.app:app --host 0.0.0.0 --port 8765 --reload
# then open http://localhost:8765
```

## Project Structure

```
master_agent.py            -- Master orchestrator and CLI entry point
config.py                  -- Model and token/memory configuration
agents/
  base_agent.py            -- Abstract agent: enforced run/memory/log flow
  amazon_agent.py          -- Amazon shopping specialist
  doordash_agent.py        -- Food delivery specialist
  shopping_agent.py        -- General shopping specialist
  planner_agent.py         -- Day planning specialist
  daily_routine_agent.py   -- Habits & routines specialist
memory/
  memory_store.py          -- Persistent JSON state, task queue, messaging
guidelines/
  master_guidelines.md     -- Orchestrator rules
  agent_guidelines.md      -- Shared specialist rules
dashboard/
  app.py                   -- FastAPI + WebSocket server
  index.html               -- Dashboard UI
```

## Configuration

Defaults live in `config.py`:

- `MODEL` — Gemini model id (`gemini-2.5-flash`)
- `MAX_TOKENS_MASTER` / `MAX_TOKENS_AGENT` — output token limits
- `MEMORY_PATH` — path to the persistent state file (`memory/state.json`)

## Notes

The specialist agents provide guidance and reasoning over their domains; they do not place real orders or make purchases. The system requires network access to the Gemini API.
