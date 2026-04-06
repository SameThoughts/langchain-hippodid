"""Tests for HippoDidMemory convenience wrapper."""

from unittest.mock import MagicMock, patch

from langchain_hippodid import HippoDidMemory


@patch("langchain_hippodid.chat_message_history.HippoDid")
def test_memory_creates_history(mock_cls):
    client = MagicMock()
    mock_cls.return_value = client

    memory = HippoDidMemory(api_key="hd_test", character_id="cid-123")
    assert memory.history is not None
    assert memory.history.character_id == "cid-123"


@patch("langchain_hippodid.chat_message_history.HippoDid")
def test_memory_with_external_id(mock_cls):
    client = MagicMock()
    char = MagicMock()
    char.id = "resolved-cid"
    client.upsert_by_external_id.return_value = char
    mock_cls.return_value = client

    memory = HippoDidMemory(api_key="hd_test", external_id="user_42")
    assert memory.history.character_id == "resolved-cid"


@patch("langchain_hippodid.chat_message_history.HippoDid")
def test_get_history_returns_same_instance(mock_cls):
    client = MagicMock()
    mock_cls.return_value = client

    memory = HippoDidMemory(api_key="hd_test", character_id="cid-123")
    assert memory.get_history("any-session") is memory.history


@patch("langchain_hippodid.chat_message_history.HippoDid")
def test_get_history_ignores_session_id(mock_cls):
    client = MagicMock()
    mock_cls.return_value = client

    memory = HippoDidMemory(api_key="hd_test", character_id="cid-123")
    h1 = memory.get_history("session-a")
    h2 = memory.get_history("session-b")
    assert h1 is h2
