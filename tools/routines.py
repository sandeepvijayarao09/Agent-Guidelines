"""
Deterministic routine and habit tools used by DailyRoutineAgent.
"""

from datetime import date, timedelta

from tools.timeutil import as_int, parse_date, to_hhmm, to_minutes


def routine_timeline(
    anchor_time: str,
    steps: list[dict],
    anchor: str = "start",
    max_minutes: int | None = None,
) -> dict:
    """
    Turn an ordered list of routine steps into clock times.

    steps:  [{"name": str, "minutes": int}]
    anchor: "start" -> the first step begins at anchor_time (e.g. wake-up)
            "end"   -> the last step finishes at anchor_time (e.g. lights-out)
    max_minutes: optional time budget; the result reports whether it fits.
    """
    if anchor not in ("start", "end"):
        raise ValueError("anchor must be 'start' or 'end'")
    if not steps:
        raise ValueError("steps must not be empty")

    durations = []
    for step in steps:
        minutes = as_int(step["minutes"], "minutes")
        if minutes <= 0:
            raise ValueError(f"Step {step.get('name')!r} needs a positive duration")
        durations.append(minutes)
    total = sum(durations)

    cursor = to_minutes(anchor_time) if anchor == "start" else to_minutes(anchor_time) - total
    timeline = []
    for step, minutes in zip(steps, durations):
        timeline.append({"start": to_hhmm(cursor), "end": to_hhmm(cursor + minutes), "step": step["name"], "minutes": minutes})
        cursor += minutes

    result = {
        "timeline": timeline,
        "starts_at": timeline[0]["start"],
        "ends_at": timeline[-1]["end"],
        "total_minutes": total,
    }
    if max_minutes is not None:
        budget = as_int(max_minutes, "max_minutes")
        result["max_minutes"] = budget
        result["fits"] = total <= budget
        result["over_by_minutes"] = max(0, total - budget)
    return result


def habit_streaks(check_ins: dict[str, list[str]], today: str | None = None) -> dict:
    """
    Compute streak stats from completion dates.

    check_ins: {"habit name": ["2026-05-01", "2026-05-02", ...]}
    A current streak counts consecutive days ending today, or ending
    yesterday if today hasn't been checked in yet.
    """
    end = parse_date(today) if today else date.today()
    report = {}
    for habit, raw_dates in check_ins.items():
        days = sorted({parse_date(d) for d in raw_dates if parse_date(d) <= end})

        longest = run = 0
        previous = None
        for day in days:
            run = run + 1 if previous and day - previous == timedelta(days=1) else 1
            longest = max(longest, run)
            previous = day

        done = set(days)
        cursor = end if end in done else end - timedelta(days=1)
        current = 0
        while cursor in done:
            current += 1
            cursor -= timedelta(days=1)

        last7 = {end - timedelta(days=i) for i in range(7)}
        completion = round(100 * len(done & last7) / 7)
        last_done = days[-1] if days else None
        report[habit] = {
            "current_streak": current,
            "longest_streak": longest,
            "completion_7d_pct": completion,
            "last_completed": last_done.isoformat() if last_done else None,
            "days_since_last": (end - last_done).days if last_done else None,
            "needs_reset": last_done is None or (end - last_done).days >= 3,
        }
    return {"as_of": end.isoformat(), "habits": report}
