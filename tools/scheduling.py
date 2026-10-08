"""
Deterministic scheduling tools used by PlannerAgent.

The LLM decides *what* matters; these functions do the clock math so the
schedule it returns never has overlapping blocks or arithmetic mistakes.
"""

from datetime import date

from tools.timeutil import as_int, parse_date, to_hhmm, to_minutes

MIN_CHUNK = 15  # don't create work fragments shorter than this


def date_info(target_date: str, reference_date: str | None = None) -> dict:
    """Weekday, ISO week and day offset for a date, relative to reference_date (default today)."""
    target = parse_date(target_date)
    reference = parse_date(reference_date) if reference_date else date.today()
    delta = (target - reference).days
    return {
        "date": target.isoformat(),
        "weekday": target.strftime("%A"),
        "iso_week": target.isocalendar().week,
        "is_weekend": target.weekday() >= 5,
        "reference_date": reference.isoformat(),
        "days_from_reference": delta,
    }


def _free_windows(day_start: int, day_end: int, fixed: list[dict]) -> list[tuple[int, int]]:
    windows = []
    cursor = day_start
    for event in fixed:
        event_start = min(event["_start"], day_end)
        if event_start > cursor:
            windows.append((cursor, event_start))
        cursor = max(cursor, min(event["_end"], day_end))
    if cursor < day_end:
        windows.append((cursor, day_end))
    return windows


def build_schedule(
    day_start: str,
    day_end: str,
    tasks: list[dict],
    fixed_events: list[dict] | None = None,
    break_minutes: int = 10,
    max_block_minutes: int = 90,
) -> dict:
    """
    Build a time-blocked day.

    tasks:        [{"name": str, "minutes": int, "priority": 1|2|3}]  (1 = highest)
    fixed_events: [{"name": str, "start": "HH:MM", "end": "HH:MM"}]  (meetings, meals...)

    Tasks are placed in priority order (ties keep input order) into the gaps
    between fixed events. Long tasks are split into blocks of at most
    max_block_minutes with a break after each block when there is room.
    Anything that doesn't fit is returned under "unscheduled".
    """
    start, end = to_minutes(day_start), to_minutes(day_end)
    if end <= start:
        raise ValueError("day_end must be after day_start (same day)")
    break_minutes = as_int(break_minutes, "break_minutes")
    max_block = as_int(max_block_minutes, "max_block_minutes")
    if max_block < MIN_CHUNK:
        raise ValueError(f"max_block_minutes must be at least {MIN_CHUNK}")

    fixed = []
    for event in fixed_events or []:
        fs, fe = to_minutes(event["start"]), to_minutes(event["end"])
        if fe <= fs:
            raise ValueError(f"Fixed event {event.get('name')!r} ends before it starts")
        fixed.append({"name": event.get("name", "Fixed"), "_start": fs, "_end": fe})
    fixed.sort(key=lambda e: e["_start"])

    conflicts = []
    for a, b in zip(fixed, fixed[1:]):
        if b["_start"] < a["_end"]:
            conflicts.append(
                f"{a['name']} ({to_hhmm(a['_start'])}-{to_hhmm(a['_end'])}) overlaps "
                f"{b['name']} ({to_hhmm(b['_start'])}-{to_hhmm(b['_end'])})"
            )
    for event in fixed:
        if event["_start"] < start or event["_end"] > end:
            conflicts.append(f"{event['name']} falls outside {day_start}-{day_end}")

    queue = []
    for index, task in enumerate(tasks):
        minutes = as_int(task["minutes"], "minutes")
        if minutes <= 0:
            raise ValueError(f"Task {task.get('name')!r} needs a positive duration")
        priority = as_int(task.get("priority", 2), "priority")
        queue.append({"name": task["name"], "remaining": minutes, "priority": priority, "order": index})
    queue.sort(key=lambda t: (t["priority"], t["order"]))

    blocks = [
        {"start": to_hhmm(e["_start"]), "end": to_hhmm(e["_end"]), "label": e["name"], "kind": "fixed"}
        for e in fixed
    ]

    for w_start, w_end in _free_windows(start, end, fixed):
        cursor = w_start
        while queue and w_end - cursor >= MIN_CHUNK:
            task = queue[0]
            room = w_end - cursor
            chunk = min(task["remaining"], max_block, room)
            # Avoid a tiny leftover fragment: only split if both halves are usable.
            if chunk < task["remaining"] and chunk < MIN_CHUNK:
                break
            blocks.append(
                {"start": to_hhmm(cursor), "end": to_hhmm(cursor + chunk), "label": task["name"], "kind": "task"}
            )
            cursor += chunk
            task["remaining"] -= chunk
            if task["remaining"] == 0:
                queue.pop(0)
            if queue and break_minutes and w_end - cursor >= break_minutes + MIN_CHUNK:
                blocks.append(
                    {"start": to_hhmm(cursor), "end": to_hhmm(cursor + break_minutes), "label": "Break", "kind": "break"}
                )
                cursor += break_minutes

    blocks.sort(key=lambda b: (to_minutes(b["start"]), b["kind"] != "fixed"))
    task_minutes = sum(to_minutes(b["end"]) - to_minutes(b["start"]) for b in blocks if b["kind"] == "task")
    fixed_minutes = sum(e["_end"] - e["_start"] for e in fixed)
    return {
        "blocks": blocks,
        "unscheduled": [{"name": t["name"], "minutes_remaining": t["remaining"]} for t in queue],
        "conflicts": conflicts,
        "scheduled_task_minutes": task_minutes,
        "fixed_minutes": fixed_minutes,
        "day_minutes": end - start,
    }
