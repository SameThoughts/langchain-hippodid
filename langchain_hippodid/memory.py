"""HippoDidMemory — convenience factory for LangChain's RunnableWithMessageHistory."""

from __future__ import annotations

from typing import Optional

from langchain_hippodid.chat_message_history import HippoDidChatMessageHistory


class HippoDidMemory:
    """Convenience wrapper that creates a HippoDidChatMessageHistory.

    Use with LangChain's ``RunnableWithMessageHistory`` for persistent
    conversation memory::

        from langchain_hippodid import HippoDidMemory
        from langchain_core.runnables.history import RunnableWithMessageHistory

        memory = HippoDidMemory(
            api_key="hd_...",
            character_id="your-character-uuid",
        )

        chain_with_memory = RunnableWithMessageHistory(
            chain,
            lambda session_id: memory.history,
        )

    Or use the history object directly::

        memory = HippoDidMemory(api_key="hd_...", character_id="...")
        memory.history.add_message(HumanMessage(content="Hello"))
        messages = memory.history.messages
    """

    def __init__(
        self,
        api_key: str,
        character_id: Optional[str] = None,
        external_id: Optional[str] = None,
        base_url: str = "https://api.hippodid.com",
        memory_mode: str = "VERBATIM",
        search_top_k: int = 20,
    ):
        self._history = HippoDidChatMessageHistory(
            api_key=api_key,
            character_id=character_id,
            external_id=external_id,
            base_url=base_url,
            memory_mode=memory_mode,
            search_top_k=search_top_k,
        )

    @property
    def history(self) -> HippoDidChatMessageHistory:
        """The underlying chat message history."""
        return self._history

    def get_history(self, session_id: str = "") -> HippoDidChatMessageHistory:
        """Session-keyed accessor for RunnableWithMessageHistory.

        The session_id is ignored — HippoDid scopes memory by character_id.
        This method exists for compatibility with ``RunnableWithMessageHistory``
        which requires a callable ``(session_id) -> BaseChatMessageHistory``.
        """
        return self._history
