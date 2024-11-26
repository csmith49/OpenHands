from typing import Literal

from pydantic import Field

from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM
from openhands.memory.condenser.base import Condenser, CondenserConfig, Event


class LLMCondenserConfig(CondenserConfig):
    """Configuration for LLMCondenser."""
    type: Literal["llm"] = Field(default="llm", description="Must be 'llm'")
    llm_config: str | None = Field(default=None, description="Name of LLM config to use")


class LLMCondenser(Condenser):
    type_name = "llm"

    def __init__(self, llm: LLM):
        self.llm = llm

    def condense(self, events: list[Event]) -> list[Event]:
        try:
            messages = [{'content': event.content, 'role': event.role} for event in events]
            resp = self.llm.completion(messages=messages)
            summary_response = resp['choices'][0]['message']['content']
            return [Event(content=summary_response, role='assistant')]
        except Exception as e:
            logger.error('Error condensing thoughts: %s', str(e), exc_info=False)
            raise