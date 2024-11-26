from typing import Literal

from pydantic import Field

from openhands.memory.condenser.base import Condenser, CondenserConfig, Event


class NoOpCondenserConfig(CondenserConfig):
    """Configuration for NoOpCondenser.
    Does not require any additional parameters."""
    type: Literal["noop"] = Field(default="noop", description="Must be 'noop'")


class NoOpCondenser(Condenser):
    type_name = "noop"

    def condense(self, events: list[Event]) -> list[Event]:
        return events