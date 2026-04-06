"""Tests for HippoDidChatMessageHistory."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_hippodid import HippoDidChatMessageHistory


@pytest.fixture()
def mock_client():
    with patch("langchain_hippodid.chat_message_history.HippoDid") as cls:
        client = MagicMock()
        cls.return_value = client
        yield client


def test_init_with_character_id(mock_client):
    history = HippoDidChatMessageHistory(api_key="hd_test", character_id="cid-123")
    assert history.character_id == "cid-123"


def test_init_with_external_id(mock_client):
    char = MagicMock()
    char.id = "resolved-cid"
    mock_client.upsert_by_external_id.return_value = char

    history = HippoDidChatMessageHistory(api_key="hd_test", external_id="user_42")
    assert history.character_id == "resolved-cid"
    mock_client.upsert_by_external_id.assert_called_once_with("user_42")


def test_init_requires_id():
    with pytest.raises(ValueError, match="Either character_id or external_id"):
        HippoDidChatMessageHistory(api_key="hd_test")


def test_init_rejects_both_ids():
    with pytest.raises(ValueError, match="character_id OR external_id"):
        HippoDidChatMessageHistory(api_key="hd_test", character_id="cid-1", external_id="ext-1")


def test_add_human_message(mock_client):
    history = HippoDidChatMessageHistory(api_key="hd_test", character_id="cid-123")
    history.add_message(HumanMessage(content="Hello"))
    mock_client.add_memory.assert_called_once_with("cid-123", "Human: Hello")


def test_add_ai_message(mock_client):
    history = HippoDidChatMessageHistory(api_key="hd_test", character_id="cid-123")
    history.add_message(AIMessage(content="Hi there"))
    mock_client.add_memory.assert_called_once_with("cid-123", "AI: Hi there")


def test_add_system_message(mock_client):
    history = HippoDidChatMessageHistory(api_key="hd_test", character_id="cid-123")
    history.add_message(SystemMessage(content="You are a bot"))
    mock_client.add_memory.assert_called_once_with("cid-123", "You are a bot")


def test_add_messages_combines(mock_client):
    history = HippoDidChatMessageHistory(api_key="hd_test", character_id="cid-123")
    history.add_messages(
        [
            HumanMessage(content="Hi"),
            AIMessage(content="Hello!"),
        ]
    )
    mock_client.add_memory.assert_called_once_with("cid-123", "Human: Hi\nAI: Hello!")


def test_messages_parses_roles(mock_client):
    result_human = MagicMock()
    result_human.content = "Human: What is HippoDid?"
    result_ai = MagicMock()
    result_ai.content = "AI: It's a memory layer for AI agents."
    result_fact = MagicMock()
    result_fact.content = "User prefers Python over JavaScript."

    mock_client.search_memories.return_value = [result_human, result_ai, result_fact]

    history = HippoDidChatMessageHistory(api_key="hd_test", character_id="cid-123")
    msgs = history.messages

    assert len(msgs) == 3
    assert isinstance(msgs[0], HumanMessage)
    assert msgs[0].content == "What is HippoDid?"
    assert isinstance(msgs[1], AIMessage)
    assert msgs[1].content == "It's a memory layer for AI agents."
    assert isinstance(msgs[2], SystemMessage)
    assert msgs[2].content == "User prefers Python over JavaScript."


def test_search_delegates(mock_client):
    mock_client.search_memories.return_value = []
    history = HippoDidChatMessageHistory(api_key="hd_test", character_id="cid-123")
    results = history.search("travel preferences", top_k=5)
    mock_client.search_memories.assert_called_with(
        "cid-123", "travel preferences", top_k=5, categories=None
    )
    assert results == []


def test_get_context_delegates(mock_client):
    ctx = MagicMock()
    ctx.formatted_prompt = "You are Ada. She likes Python."
    mock_client.assemble_context.return_value = ctx

    history = HippoDidChatMessageHistory(api_key="hd_test", character_id="cid-123")
    prompt = history.get_context("user preferences")

    mock_client.assemble_context.assert_called_once_with("cid-123", "user preferences")
    assert prompt == "You are Ada. She likes Python."


def test_clear_is_noop(mock_client):
    history = HippoDidChatMessageHistory(api_key="hd_test", character_id="cid-123")
    history.clear()  # Should not raise
