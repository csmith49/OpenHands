from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from openhands.controller.state.state import State


class Trigger(ABC):
    """Abstract base class for condenser triggers.
    
    A trigger determines when a condenser should fire based on the current state.
    """
    
    @abstractmethod
    def should_fire(self, state: State) -> bool:
        """Determine if the condenser should fire based on the current state.
        
        Args:
            state: The current state to evaluate.
            
        Returns:
            bool: True if the condenser should fire, False otherwise.
        """


class AlwaysTrigger(Trigger):
    """A trigger that always fires."""
    
    def should_fire(self, state: State) -> bool:
        return True


class HistoryLengthTrigger(Trigger):
    """A trigger that fires when the history reaches a certain length."""
    
    def __init__(self, min_length: int) -> None:
        """Initialize the trigger.
        
        Args:
            min_length: The minimum length of history required to trigger.
        """
        self.min_length = min_length
    
    def should_fire(self, state: State) -> bool:
        return len(state.history) >= self.min_length