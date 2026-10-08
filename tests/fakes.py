"""
Offline stand-in for google.genai.Client.

Returns real google.genai response objects from a script, so the agents run
their normal function-calling code paths without any network access.
"""

from google.genai import types

MEMORY_MARKER = "Your only job is to update your personal memory file"
MASTER_MARKER = "**MasterAgent** orchestrator"


def text_response(text: str) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[types.Candidate(content=types.Content(role="model", parts=[types.Part(text=text)]))]
    )


def call_response(name: str, args: dict) -> types.GenerateContentResponse:
    return types.GenerateContentResponse(
        candidates=[
            types.Candidate(
                content=types.Content(
                    role="model",
                    parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))],
                )
            )
        ]
    )


class _Models:
    def __init__(self, owner: "FakeClient"):
        self._owner = owner

    def generate_content(self, model, contents, config):
        return self._owner._respond(model, list(contents), config)


class FakeClient:
    """
    master_script: responses for MasterAgent calls, in order.
    agent_script:  responses for specialist execute() calls, in order.
    Post-task memory rewrites are answered automatically.
    """

    def __init__(self, agent_script=None, master_script=None, memory_text="# Memory\n- Prefers short answers"):
        self.agent_script = list(agent_script or [])
        self.master_script = list(master_script or [])
        self.memory_text = memory_text
        self.calls: list[dict] = []
        self.models = _Models(self)

    def _respond(self, model, contents, config):
        system = config.system_instruction or ""
        kind = "memory" if MEMORY_MARKER in system else "master" if MASTER_MARKER in system else "agent"
        self.calls.append({"kind": kind, "model": model, "contents": contents, "config": config})
        if kind == "memory":
            return text_response(self.memory_text)
        script = self.master_script if kind == "master" else self.agent_script
        if not script:
            raise AssertionError(f"Unexpected {kind} call: script exhausted")
        return script.pop(0)

    def calls_of(self, kind: str) -> list[dict]:
        return [c for c in self.calls if c["kind"] == kind]


def function_responses(call: dict) -> list[types.FunctionResponse]:
    """Function responses sent back to the model in a recorded call."""
    return [p.function_response for c in call["contents"] for p in (c.parts or []) if p.function_response]
