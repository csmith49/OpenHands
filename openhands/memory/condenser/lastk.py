from typing import Literal

from pydantic import Field

from openhands.memory.condenser.base import Condenser, CondenserConfig, Event


class LastKCondenserConfig(CondenserConfig):
    """Configuration for LastKCondenser."""
    type: Literal["lastk"] = Field(default="lastk", description="Must be 'lastk'")
    k: int = Field(default=5, description="Number of non-user messages to keep")


class LastKCondenser(Condenser):
    type_name = "lastk"

    def __init__(self, k: int):
        self.k = k

    def condense(self, events: list[Event]) -> list[Event]:
        user_messages = [e for e in events if e.role == 'user']
        other_messages = [e for e in events if e.role != 'user']
        
        if len(other_messages) <= self.k:
            return user_messages + other_messages
        
        return user_messages + other_messages[-self.k:]