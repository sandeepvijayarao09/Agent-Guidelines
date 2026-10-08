import pytest

from tools.scheduling import build_schedule, date_info
from tools.timeutil import to_hhmm, to_minutes


def test_clock_helpers_round_trip():
    assert to_minutes("07:30") == 450
    assert to_hhmm(450) == "07:30"
    assert to_hhmm(24 * 60 + 60) == "01:00"
    with pytest.raises(ValueError):
        to_minutes("7.30")
    with pytest.raises(ValueError):
        to_minutes("25:00")


def test_date_info_weekday_and_offset():
    info = date_info("2026-05-15", reference_date="2026-05-09")
    assert info["weekday"] == "Friday"
    assert info["days_from_reference"] == 6
    assert info["is_weekend"] is False
    assert date_info("2026-05-09", "2026-05-15")["days_from_reference"] == -6


def _minutes(block):
    return to_minutes(block["end"]) - to_minutes(block["start"])


def test_build_schedule_places_tasks_by_priority_around_fixed_events():
    result = build_schedule(
        day_start="09:00",
        day_end="13:00",
        tasks=[
            {"name": "Email", "minutes": 30, "priority": 3},
            {"name": "Write report", "minutes": 120, "priority": 1},
        ],
        fixed_events=[{"name": "Standup", "start": "10:00", "end": "10:30"}],
        break_minutes=10,
        max_block_minutes=60,
    )
    labels = [(b["start"], b["end"], b["label"]) for b in result["blocks"]]
    # Highest priority first; it fills the 60-minute gap before standup exactly.
    assert labels[0] == ("09:00", "10:00", "Write report")
    assert ("10:00", "10:30", "Standup") in labels
    report_minutes = sum(_minutes(b) for b in result["blocks"] if b["label"] == "Write report")
    assert report_minutes == 120
    assert any(b["label"] == "Email" for b in result["blocks"])
    assert result["unscheduled"] == []
    assert result["conflicts"] == []


def test_build_schedule_blocks_never_overlap_and_stay_in_day():
    result = build_schedule(
        "08:00",
        "18:00",
        tasks=[{"name": f"T{i}", "minutes": 75, "priority": i % 3 + 1} for i in range(6)],
        fixed_events=[
            {"name": "Lunch", "start": "12:00", "end": "13:00"},
            {"name": "1:1", "start": "15:00", "end": "15:30"},
        ],
    )
    blocks = sorted(result["blocks"], key=lambda b: to_minutes(b["start"]))
    for a, b in zip(blocks, blocks[1:]):
        assert to_minutes(a["end"]) <= to_minutes(b["start"])
    assert to_minutes(blocks[0]["start"]) >= to_minutes("08:00")
    assert to_minutes(blocks[-1]["end"]) <= to_minutes("18:00")
    assert all(_minutes(b) <= 90 for b in blocks if b["kind"] == "task")


def test_build_schedule_reports_overflow_and_conflicts():
    result = build_schedule(
        "09:00",
        "10:00",
        tasks=[{"name": "Big project", "minutes": 180, "priority": 1}],
        fixed_events=[
            {"name": "A", "start": "09:15", "end": "09:45"},
            {"name": "B", "start": "09:30", "end": "10:30"},
        ],
    )
    assert result["unscheduled"] == [{"name": "Big project", "minutes_remaining": 165}]
    assert any("overlaps" in c for c in result["conflicts"])
    assert any("outside" in c for c in result["conflicts"])


def test_build_schedule_accepts_float_args_from_llm():
    result = build_schedule("09:00", "10:00", [{"name": "Read", "minutes": 30.0, "priority": 1.0}])
    assert result["scheduled_task_minutes"] == 30


def test_build_schedule_rejects_bad_input():
    with pytest.raises(ValueError):
        build_schedule("10:00", "09:00", [])
    with pytest.raises(ValueError):
        build_schedule("09:00", "10:00", [{"name": "X", "minutes": 0}])
