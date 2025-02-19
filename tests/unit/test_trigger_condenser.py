from unittest.mock import MagicMock

import pytest

from openhands.controller.state.state import State
from openhands.events.event import Event
from openhands.memory.condenser import Condenser
from openhands.memory.condenser.trigger import HistoryLengthTrigger


class SimpleCondenser(Condenser):
    """A simple condenser that just returns the first event."""

    def condense(self, events: list[Event]) -> list[Event]:
        return events[:1]


class SimpleRollingCondenser(SimpleCondenser):
    """A simple rolling condenser that just returns the first event."""

    def __init__(self, trigger=None):
        super().__init__(trigger)
        self._condensation = []
        self._last_history_length = 0

    def condensed_history(self, state: State) -> list[Event]:
        new_events = state.history[self._last_history_length:]
        current_events = self._condensation + new_events

        if not self._trigger.should_fire(state):
            self._condensation = current_events
            self._last_history_length = len(state.history)
            return current_events

        with self.metadata_batch(state):
            results = self.condense(current_events)

        self._condensation = results
        self._last_history_length = len(state.history)

        return results


@pytest.fixture
def mock_state() -> State:
    """Mocks a State object with the only parameters needed for testing condensers: history and extra_data."""
    mock_state = MagicMock(spec=State)
    mock_state.history = []
    mock_state.extra_data = {}
    return mock_state


def test_condenser_with_history_length_trigger(mock_state):
    """Test that condenser only fires when history length trigger condition is met."""
    min_length = 3
    trigger = HistoryLengthTrigger(min_length=min_length)
    condenser = SimpleCondenser(trigger=trigger)

    # Add events one by one
    events = []
    for i in range(5):
        event = MagicMock(spec=Event)
        event._message = f"Event {i}"
        events.append(event)
        mock_state.history = events
        result = condenser.condensed_history(mock_state)

        # Before min_length, should return full history
        if len(events) < min_length:
            assert result == events
        # After min_length, should return condensed history
        else:
            assert result == events[:1]


def test_rolling_condenser_with_history_length_trigger(mock_state):
    """Test that rolling condenser maintains state correctly with trigger."""
    min_length = 3
    trigger = HistoryLengthTrigger(min_length=min_length)
    condenser = SimpleRollingCondenser(trigger=trigger)

    # Add events one by one
    events = []
    for i in range(5):
        event = MagicMock(spec=Event)
        event._message = f"Event {i}"
        events.append(event)
        mock_state.history = events
        result = condenser.condensed_history(mock_state)

        # Before min_length, should return current events
        if len(events) < min_length:
            assert result == events
        # After min_length, should return condensed history
        else:
            # Should be just the first event since our SimpleRollingCondenser
            # always condenses to just the first event
            assert result == events[:1]