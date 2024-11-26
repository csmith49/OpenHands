from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Type

from pydantic import BaseModel, Field


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