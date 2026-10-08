"""Small helpers for HH:MM clock arithmetic shared by the scheduling tools."""

from datetime import date


def to_minutes(hhmm: str) -> int:
    """'07:30' -> 450. Accepts 'H:MM' or 'HH:MM' on a 24-hour clock."""
    try:
        hours, minutes = str(hhmm).strip().split(":")
        h, m = int(hours), int(minutes)
    except ValueError:
        raise ValueError(f"Expected a 24-hour time like '07:30', got {hhmm!r}")
    if not (0 <= h <= 24 and 0 <= m < 60) or (h == 24 and m != 0):
        raise ValueError(f"Time out of range: {hhmm!r}")
    return h * 60 + m


def to_hhmm(minutes: int) -> str:
    """450 -> '07:30'. Wraps past midnight (1500 -> '01:00')."""
    minutes = int(minutes) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def as_int(value, name: str) -> int:
    """LLM function-call args often arrive as floats (30.0); normalise to int."""
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        raise ValueError(f"{name} must be a number, got {value!r}")
    return number


def parse_date(value: str) -> date:
    try:
        return date.fromisoformat(str(value).strip())
    except ValueError:
        raise ValueError(f"Expected an ISO date like '2026-05-09', got {value!r}")
