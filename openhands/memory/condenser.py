from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Type

from pydantic import BaseModel, Field

from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM


@dataclass
class Event:
    content: str
    role: str


class CondenserConfig(BaseModel):
    """Configuration for memory condensers.
    
    Attributes:
        type: The type of condenser to use ('llm', 'noop', 'lastk')
        k: Number of non-user messages to keep for LastKCondenser
        llm_config: The name of the llm config to use for LLMCondenser
    """
    type: str = Field(default="noop", description="Type of condenser to use")
    k: int = Field(default=5, description="Number of non-user messages to keep (for LastKCondenser)")
    llm_config: str | None = Field(default=None, description="Name of LLM config to use (for LLMCondenser)")


class Condenser(ABC):
    _registry: dict[str, Type['Condenser']] = {}

    @abstractmethod
    def condense(self, events: List[Event]) -> List[Event]:
        pass

    @classmethod
    def register(cls, name: str, condenser_cls: Type['Condenser']) -> None:
        """Register a condenser class with the given name.
        
        Args:
            name: Name to register the condenser under
            condenser_cls: The condenser class to register
        
        Raises:
            ValueError: If a condenser is already registered with this name
        """
        if name in cls._registry:
            raise ValueError(f"Condenser already registered with name: {name}")
        cls._registry[name] = condenser_cls

    @classmethod
    def get_cls(cls, name: str) -> Type['Condenser']:
        """Get a registered condenser class by name.
        
        Args:
            name: Name of the condenser to retrieve
        
        Returns:
            The registered condenser class
        
        Raises:
            ValueError: If no condenser is registered with this name
        """
        if name not in cls._registry:
            raise ValueError(f"No condenser registered with name: {name}")
        return cls._registry[name]


class LLMCondenser(Condenser):
    def __init__(self, llm: LLM):
        self.llm = llm

    def condense(self, events: List[Event]) -> List[Event]:
        try:
            messages = [{'content': event.content, 'role': event.role} for event in events]
            resp = self.llm.completion(messages=messages)
            summary_response = resp['choices'][0]['message']['content']
            return [Event(content=summary_response, role='assistant')]
        except Exception as e:
            logger.error('Error condensing thoughts: %s', str(e), exc_info=False)
            raise


class NoOpCondenser(Condenser):
    def condense(self, events: List[Event]) -> List[Event]:
        return events


class LastKCondenser(Condenser):
    def __init__(self, k: int):
        self.k = k

    def condense(self, events: List[Event]) -> List[Event]:
        user_messages = [e for e in events if e.role == 'user']
        other_messages = [e for e in events if e.role != 'user']
        
        if len(other_messages) <= self.k:
            return user_messages + other_messages
        
        return user_messages + other_messages[-self.k:]


# Register the built-in condensers
Condenser.register("llm", LLMCondenser)
Condenser.register("noop", NoOpCondenser)
Condenser.register("lastk", LastKCondenser)
