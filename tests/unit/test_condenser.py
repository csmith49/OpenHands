from unittest.mock import Mock, patch

import pytest
from pydantic import ValidationError

from openhands.core.exceptions import LLMResponseError
from openhands.llm.llm import LLM
from openhands.memory.condenser import (
    Event,
    LLMCondenser,
    NoOpCondenser,
    LastKCondenser,
    Condenser,
    NoOpCondenserConfig,
    LastKCondenserConfig,
    LLMCondenserConfig,
)


@pytest.fixture
def mock_llm():
    return Mock(spec=LLM)


@pytest.fixture
def llm_condenser(mock_llm):
    return LLMCondenser(mock_llm)


@pytest.fixture
def events():
    return [
        Event(content="Hello", role="user"),
        Event(content="Hi there", role="assistant"),
        Event(content="How are you?", role="user"),
        Event(content="I'm good", role="assistant"),
    ]


def test_llm_condense_success(llm_condenser, mock_llm, events):
    mock_llm.completion.return_value = {
        'choices': [{'message': {'content': 'Condensed memory'}}]
    }
    result = llm_condenser.condense(events)
    assert len(result) == 1
    assert result[0].content == 'Condensed memory'
    assert result[0].role == 'assistant'
    mock_llm.completion.assert_called_once()


def test_llm_condense_exception(llm_condenser, mock_llm, events):
    mock_llm.completion.side_effect = LLMResponseError('LLM error')
    with pytest.raises(LLMResponseError, match='LLM error'):
        llm_condenser.condense(events)


@patch('openhands.memory.condenser.logger')
def test_llm_condense_logs_error(mock_logger, llm_condenser, mock_llm, events):
    mock_llm.completion.side_effect = LLMResponseError('LLM error')
    with pytest.raises(LLMResponseError):
        llm_condenser.condense(events)
    mock_logger.error.assert_called_once_with(
        'Error condensing thoughts: %s', 'LLM error', exc_info=False
    )


def test_noop_condenser(events):
    condenser = NoOpCondenser()
    result = condenser.condense(events)
    assert result == events


def test_lastk_condenser_keeps_all_user_messages():
    events = [
        Event(content="Hello", role="user"),
        Event(content="Hi there", role="assistant"),
        Event(content="How are you?", role="user"),
        Event(content="I'm good", role="assistant"),
    ]
    condenser = LastKCondenser(k=1)
    result = condenser.condense(events)
    
    user_messages = [e for e in result if e.role == "user"]
    assert len(user_messages) == 2
    assert user_messages[0].content == "Hello"
    assert user_messages[1].content == "How are you?"


def test_lastk_condenser_keeps_k_non_user_messages():
    events = [
        Event(content="Hi there", role="assistant"),
        Event(content="Hello", role="user"),
        Event(content="How can I help?", role="assistant"),
        Event(content="I need help", role="user"),
        Event(content="Sure!", role="assistant"),
    ]
    condenser = LastKCondenser(k=2)
    result = condenser.condense(events)
    
    non_user_messages = [e for e in result if e.role != "user"]
    assert len(non_user_messages) == 2
    assert non_user_messages[0].content == "How can I help?"
    assert non_user_messages[1].content == "Sure!"


def test_lastk_condenser_with_k_larger_than_messages():
    events = [
        Event(content="Hello", role="user"),
        Event(content="Hi there", role="assistant"),
    ]
    condenser = LastKCondenser(k=5)
    result = condenser.condense(events)
    assert result == events


def test_condenser_registry():
    # Test registration
    class TestCondenser(Condenser):
        def condense(self, events: list[Event]) -> list[Event]:
            return events
    
    Condenser.register("test", TestCondenser)
    assert Condenser.get_cls("test") == TestCondenser
    
    # Test duplicate registration
    with pytest.raises(ValueError, match="Condenser already registered with name: test"):
        Condenser.register("test", TestCondenser)
    
    # Test getting non-existent condenser
    with pytest.raises(ValueError, match="No condenser registered with name: nonexistent"):
        Condenser.get_cls("nonexistent")


def test_noop_condenser_config_validation():
    # Test valid config
    config = NoOpCondenserConfig()
    assert config.type == "noop"
    
    # Test invalid type
    with pytest.raises(ValidationError):
        NoOpCondenserConfig(type="invalid")
    
    with pytest.raises(ValidationError):
        NoOpCondenserConfig(type=123)


def test_lastk_condenser_config_validation():
    # Test valid configs
    config = LastKCondenserConfig()
    assert config.type == "lastk"
    assert config.k == 5  # default value
    
    config = LastKCondenserConfig(k=10)
    assert config.type == "lastk"
    assert config.k == 10
    
    # Test invalid type
    with pytest.raises(ValidationError):
        LastKCondenserConfig(type="invalid")
    
    # Test invalid k
    with pytest.raises(ValidationError):
        LastKCondenserConfig(k="invalid")


def test_llm_condenser_config_validation():
    # Test valid configs
    config = LLMCondenserConfig()
    assert config.type == "llm"
    assert config.llm_config is None  # default value
    
    config = LLMCondenserConfig(llm_config="gpt-4")
    assert config.type == "llm"
    assert config.llm_config == "gpt-4"
    
    # Test invalid type
    with pytest.raises(ValidationError):
        LLMCondenserConfig(type="invalid")
    
    # Test invalid llm_config type
    with pytest.raises(ValidationError):
        LLMCondenserConfig(llm_config=123)
