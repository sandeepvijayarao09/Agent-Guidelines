import pytest

from tools.routines import habit_streaks, routine_timeline

STEPS = [{"name": "Water", "minutes": 5}, {"name": "Stretch", "minutes": 15}, {"name": "Journal", "minutes": 10}]


def test_routine_timeline_forward_from_wake_time():
    result = routine_timeline("06:30", STEPS, anchor="start", max_minutes=45)
    assert [(s["start"], s["end"]) for s in result["timeline"]] == [("06:30", "06:35"), ("06:35", "06:50"), ("06:50", "07:00")]
    assert result["total_minutes"] == 30
    assert result["fits"] is True and result["over_by_minutes"] == 0


def test_routine_timeline_backward_from_bedtime_crosses_midnight():
    result = routine_timeline("00:15", STEPS, anchor="end", max_minutes=20)
    assert result["starts_at"] == "23:45"
    assert result["ends_at"] == "00:15"
    assert result["fits"] is False and result["over_by_minutes"] == 10


def test_routine_timeline_validates():
    with pytest.raises(ValueError):
        routine_timeline("06:00", STEPS, anchor="middle")
    with pytest.raises(ValueError):
        routine_timeline("06:00", [])


def test_habit_streaks():
    report = habit_streaks(
        {
            "meditate": ["2026-05-01", "2026-05-02", "2026-05-03", "2026-05-06", "2026-05-07", "2026-05-08"],
            "run": ["2026-05-01", "2026-05-02"],
            "read": [],
        },
        today="2026-05-09",
    )["habits"]
    # Not checked in today yet, so the streak ending yesterday still counts.
    assert report["meditate"]["current_streak"] == 3
    assert report["meditate"]["longest_streak"] == 3
    assert report["meditate"]["completion_7d_pct"] == round(100 * 4 / 7)
    assert report["meditate"]["needs_reset"] is False
    assert report["run"]["current_streak"] == 0
    assert report["run"]["days_since_last"] == 7
    assert report["run"]["needs_reset"] is True
    assert report["read"]["last_completed"] is None


def test_habit_streaks_ignores_future_and_duplicate_dates():
    report = habit_streaks({"x": ["2026-05-09", "2026-05-09", "2026-05-10"]}, today="2026-05-09")
    assert report["habits"]["x"]["current_streak"] == 1
    assert report["habits"]["x"]["longest_streak"] == 1
