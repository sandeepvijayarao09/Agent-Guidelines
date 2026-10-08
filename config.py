from pathlib import Path

ROOT = Path(__file__).resolve().parent

MODEL = "gemini-2.5-flash"
MAX_TOKENS_MASTER = 8192
MAX_TOKENS_AGENT = 4096
MAX_TOOL_ROUNDS = 6
MEMORY_PATH = str(ROOT / "memory" / "state.json")
GUIDELINES_DIR = ROOT / "guidelines"
