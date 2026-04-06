"""LangChain integration for HippoDid — persistent character memory for AI agents."""

from langchain_hippodid._version import __version__
from langchain_hippodid.chat_message_history import HippoDidChatMessageHistory
from langchain_hippodid.memory import HippoDidMemory

__all__ = [
    "__version__",
    "HippoDidChatMessageHistory",
    "HippoDidMemory",
]
