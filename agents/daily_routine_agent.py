import anthropic
from agents.base_agent import BaseAgent
from memory.memory_store import MemoryStore


class DailyRoutineAgent(BaseAgent):
    """Daily habits and routine specialist: morning/evening rituals, habit tracking, streaks."""

    @property
    def name(self) -> str:
        return "DailyRoutineAgent"

    @property
    def description(self) -> str:
        return (
            "Manages the user's daily routines and habits: designs morning/evening rituals, "
            "tracks habit streaks, identifies bottlenecks, and suggests micro-improvements."
        )

    @property
    def domain_system_prompt(self) -> str:
        return """## Daily Routine & Habit Specialist

You help users build and maintain healthy daily routines. Your responsibilities:

### Routine Design
- Gather: wake time, sleep time, work schedule, fitness goals, family commitments.
- Design a morning ritual (≤60 min) and an evening wind-down (≤45 min).
- Sequence habits using "habit stacking": anchor each new habit to an existing one.
- Distinguish between keystone habits (high leverage) and supporting habits.

### Habit Tracking
- Maintain a habit list with daily completion status in memory.
- Report current streaks and longest streaks per habit.
- When the user checks in, ask which habits were completed today and update records.
- Flag habits not completed for 3+ days in a row and suggest a reset plan.

### Check-in Format
When the user does a daily check-in, respond with:
```
📅 Check-in: [date]
✅ Completed: [habit list]
❌ Missed: [habit list]
🔥 Streaks: [habit → N days]
💡 Tip: [one micro-improvement for tomorrow]
```

### Routine Optimisation
- Identify friction points (habits frequently skipped) and suggest simplifications.
- Recommend the "2-minute rule" for habits with <40% completion rate.
- Suggest habit pairing (e.g., listen to podcast while commuting).

### Output Format
- For new routines: numbered step-by-step morning and evening sequences with time estimates.
- For habit audits: table with habit name, frequency target, current streak, 7-day completion %.
- Always end with one actionable "quick win" for tomorrow.

### Memory
- Persist the habit list, streaks, and last check-in date across sessions.
- Recall the user's stated goals to contextualise suggestions.

### Constraints
- Never suggest routines that exceed available time windows.
- Do not prescribe medical or dietary regimens beyond general wellness advice.
- Keep routines realistic — fewer, consistent habits beat ambitious, inconsistent ones.
"""

    def execute(self, task: str, context: dict) -> str:
        # Recall persistent routine data
        habits = self._recall("habits", [])
        streaks = self._recall("streaks", {})
        last_checkin = self._recall("last_checkin", None)
        goals = self._recall("goals", [])

        habit_note = f"\n\nTracked habits: {habits}" if habits else ""
        streak_note = f"\nCurrent streaks: {streaks}" if streaks else ""
        checkin_note = f"\nLast check-in: {last_checkin}" if last_checkin else ""
        goals_note = f"\nUser goals: {goals}" if goals else ""

        messages = [
            {
                "role": "user",
                "content": (
                    f"Session goal: {context.get('session_goal', 'N/A')}\n"
                    f"Date/time context: {context.get('date', 'today')}\n\n"
                    f"Task: {task}{habit_note}{streak_note}{checkin_note}{goals_note}"
                ),
            }
        ]
        result = self._call_claude(messages)

        # Update last check-in timestamp if this looks like a check-in
        if any(kw in task.lower() for kw in ("check in", "check-in", "checkin", "completed", "done today")):
            from datetime import datetime
            self._remember("last_checkin", datetime.utcnow().isoformat())

        self.memory.log(self.name, f"Completed: {task[:80]}")
        return result
