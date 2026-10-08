import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL = "gemini-2.5-flash"
MAX_TOKENS_MASTER = 8192
MAX_TOKENS_AGENT = 4096
MAX_TOOL_ROUNDS = 6
GUIDELINES_DIR = ROOT / "guidelines"

# State lives in memory/ by default; AGENT_MEMORY_DIR points the agents and
# the dashboard somewhere else (the offline demo uses memory/demo).
MEMORY_DIR = Path(os.environ.get("AGENT_MEMORY_DIR", ROOT / "memory")).resolve()
MEMORY_PATH = str(MEMORY_DIR / "state.json")
