"""Torch-free coverage for wake/sleep coordination."""

import pytest

from sleep.schedule import Phase, PhasicSleepSchedule


def test_session_boundary_requests_sleep_for_a_partial_wake_phase():
    events = []
    schedule = PhasicSleepSchedule(wake_ticks=5)
    schedule.act(lambda: events.append("act"))

    assert schedule.request_sleep() is True
    assert schedule.sleep_due is True
    result = schedule.consolidate(
        lambda: events.append("consolidate") or 4,
        reload_weights=lambda: events.append("reload") or 4,
    )

    assert result.published_version == result.loaded_version == 4
    assert events == ["act", "consolidate", "reload"]
    assert schedule.phase is Phase.WAKE
    assert schedule.ticks_in_phase == 0


def test_idle_session_boundary_does_not_schedule_empty_sleep():
    schedule = PhasicSleepSchedule(wake_ticks=5)

    assert schedule.request_sleep() is False
    assert schedule.sleep_due is False


def test_sleep_cannot_be_requested_during_consolidation():
    schedule = PhasicSleepSchedule(wake_ticks=1)
    schedule.act(lambda: None)

    def sleep_pass():
        with pytest.raises(RuntimeError, match="consolidation is running"):
            schedule.request_sleep()
        return 1

    schedule.consolidate(sleep_pass)
