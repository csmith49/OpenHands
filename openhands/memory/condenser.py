from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM


@dataclass
class Event:
    content: str
    role: str


class Condenser(ABC):
    @abstractmethod
    def condense(self, events: List[Event]) -> List[Event]:
        pass


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
