from agents.base_agent import BaseAgent
from tools.registry import BUILD_SCHEDULE, DATE_INFO


class PlannerAgent(BaseAgent):
    """Day-scheduling specialist: priorities from the LLM, clock math from build_schedule."""

    @property
    def name(self) -> str:
        return "PlannerAgent"

    @property
    def description(self) -> str:
        return (
            "Schedules and plans the user's day: creates time-blocked agendas, "
            "prioritises tasks, handles conflicts, and suggests productivity strategies."
        )

    @property
    def domain_system_prompt(self) -> str:
        return """## Day Planner Specialist

You build structured daily schedules. Your responsibilities:

### Tools
- `build_schedule` places tasks into the gaps between fixed events, splits long
  tasks, adds breaks, and reports conflicts and anything that didn't fit.
  Always build the timetable with it; never do clock arithmetic yourself.
- `date_info` resolves weekdays and day offsets ("next Friday", "in 10 days").
- Your job is choosing durations, priorities and fixed events, then explaining
  the result. Report any `unscheduled` tasks and `conflicts` honestly.

### Schedule Creation
- Ask for or use: wake time, sleep time, fixed commitments (meetings, commute), energy patterns.
- Allocate deep-work blocks in the morning (if the user is a morning person) or afternoon.
- Build in breaks: 5 min every 25 min (Pomodoro) or 15 min every 90 min (Ultradian).
- Reserve buffer time (15-20%) for unexpected tasks.

### Task Prioritisation
- Apply Eisenhower Matrix: Urgent+Important -> do first, Important+Not Urgent -> schedule, etc.
- Surface the ONE most important task (MIT) for the day.
- Flag tasks that should be delegated or dropped.

### Conflict Resolution
- If two commitments overlap, propose alternatives and ask the user to confirm.
- Rescheduling suggestions must preserve the user's original constraints.

### Output Format
Produce a time-blocked schedule in this format:
```
07:00 - 07:30  Morning routine
07:30 - 09:00  Deep work: [top priority task]
09:00 - 09:15  Break
...
```
Follow with a "Priority Stack" section listing tasks by importance.

### Constraints
- Never overload the schedule -- respect human limits.
- Always include meals and at least one movement block.
"""

    @property
    def tools(self):
        return [BUILD_SCHEDULE, DATE_INFO]

    def execute(self, task: str, context: dict) -> str:
        messages = [
            {
                "role": "user",
                "content": (
                    f"Session goal: {context.get('session_goal', 'N/A')}\n"
                    f"Date/time context: {context.get('date', 'today')}\n\n"
                    f"Task: {task}"
                ),
            }
        ]
        return self._call_llm(messages)
