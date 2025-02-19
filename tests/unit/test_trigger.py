from unittest.mock import MagicMock

import pytest

from openhands.controller.state.state import State
from openhands.memory.condenser.trigger import AlwaysTrigger, HistoryLengthTrigger


@pytest.fixture
def mock_state() -> State:
    """Mocks a State object with the only parameters needed for testing triggers: history."""
    mock_state = MagicMock(spec=State)
    mock_state.history = []
    return mock_state


def test_always_trigger(mock_state):
    """Test that AlwaysTrigger always returns True."""
    trigger = AlwaysTrigger()
    assert trigger.should_fire(mock_state)

    # Should still return True even with history
    mock_state.history = ["event1", "event2"]
    assert trigger.should_fire(mock_state)


def test_history_length_trigger(mock_state):
    """Test that HistoryLengthTrigger fires only when history reaches min_length."""
    min_length = 3
    trigger = HistoryLengthTrigger(min_length=min_length)

    # Should not fire with empty history
    assert not trigger.should_fire(mock_state)

    # Should not fire with history shorter than min_length
    mock_state.history = ["event1"]
    assert not trigger.should_fire(mock_state)

    mock_state.history = ["event1", "event2"]
    assert not trigger.should_fire(mock_state)

    # Should fire when history reaches min_length
    mock_state.history = ["event1", "event2", "event3"]
    assert trigger.should_fire(mock_state)

    # Should fire when history exceeds min_length
    mock_state.history = ["event1", "event2", "event3", "event4"]
    assert trigger.should_fire(mock_state)


def test_history_length_trigger_invalid_config():
    """Test that HistoryLengthTrigger raises error for invalid min_length."""
    with pytest.raises(ValueError, match="min_length must be positive"):
        HistoryLengthTrigger(min_length=0)

    with pytest.raises(ValueError, match="min_length must be positive"):
        HistoryLengthTrigger(min_length=-1)