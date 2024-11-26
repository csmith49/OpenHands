from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Type, Literal

from pydantic import BaseModel, Field

from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM


@dataclass
class Event:
    content: str
    role: str


class CondenserConfig(BaseModel):
    """Base configuration for memory condensers."""
    type: str = Field(..., description="Type of condenser to use")


class Condenser(ABC):
    _registry: dict[str, Type['Condenser']] = {}
    type_name: str

    @abstractmethod
    def condense(self, events: List[Event]) -> List[Event]:
        pass

    @classmethod
    def register(cls, condenser_cls: Type['Condenser']) -> None:
        """Register a condenser class using its type_name.
        
        Args:
            condenser_cls: The condenser class to register
        
        Raises:
            ValueError: If a condenser is already registered with this name
            ValueError: If condenser_cls doesn't have type_name defined
        """
        if not hasattr(condenser_cls, 'type_name'):
            raise ValueError(f"Condenser class {condenser_cls.__name__} must define type_name")
        
        name = condenser_cls.type_name
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
    type_name = "llm"

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
    type_name = "noop"

    def condense(self, events: List[Event]) -> List[Event]:
        return events


class LastKCondenser(Condenser):
    type_name = "lastk"

    def __init__(self, k: int):
        self.k = k

    def condense(self, events: List[Event]) -> List[Event]:
        user_messages = [e for e in events if e.role == 'user']
        other_messages = [e for e in events if e.role != 'user']
        
        if len(other_messages) <= self.k:
            return user_messages + other_messages
        
        return user_messages + other_messages[-self.k:]


class NoOpCondenserConfig(CondenserConfig):
    """Configuration for NoOpCondenser.
    Does not require any additional parameters."""
    type: Literal[NoOpCondenser.type_name] = Field(
        default=NoOpCondenser.type_name,
        description=f"Must be '{NoOpCondenser.type_name}'"
    )


class LastKCondenserConfig(CondenserConfig):
    """Configuration for LastKCondenser."""
    type: Literal[LastKCondenser.type_name] = Field(
        default=LastKCondenser.type_name,
        description=f"Must be '{LastKCondenser.type_name}'"
    )
    k: int = Field(default=5, description="Number of non-user messages to keep")


class LLMCondenserConfig(CondenserConfig):
    """Configuration for LLMCondenser."""
    type: Literal[LLMCondenser.type_name] = Field(
        default=LLMCondenser.type_name,
        description=f"Must be '{LLMCondenser.type_name}'"
    )
    llm_config: str | None = Field(default=None, description="Name of LLM config to use")


# Register the built-in condensers
Condenser.register(LLMCondenser)
Condenser.register(NoOpCondenser)
Condenser.register(LastKCondenser)
