"""HippoDid-backed chat message history for LangChain."""

from __future__ import annotations

from typing import List, Optional

from hippodid import HippoDid
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage


class HippoDidChatMessageHistory(BaseChatMessageHistory):
    """Chat message history backed by HippoDid persistent character memory.

    Each conversation is stored as memories on a HippoDid character.
    Messages are stored in VERBATIM mode for exact recall, or EXTRACTED
    mode for AI-structured memory extraction.

    Usage::

        from langchain_hippodid import HippoDidChatMessageHistory

        history = HippoDidChatMessageHistory(
            api_key="hd_...",
            character_id="your-character-uuid",
        )

        # Or auto-create/resolve character by external user ID:
        history = HippoDidChatMessageHistory(
            api_key="hd_...",
            external_id="user_12345",
        )
    """

    def __init__(
        self,
        api_key: str,
        character_id: Optional[str] = None,
        external_id: Optional[str] = None,
        base_url: str = "https://api.hippodid.com",
        memory_mode: str = "VERBATIM",
        search_top_k: int = 20,
        search_query: Optional[str] = None,
    ):
        """Initialize HippoDid chat message history.

        Args:
            api_key: HippoDid API key (starts with hd_).
            character_id: UUID of existing HippoDid character.
                Mutually exclusive with external_id.
            external_id: Your system's user ID. Auto-creates character if not
                found. Mutually exclusive with character_id.
            base_url: HippoDid API base URL.
            memory_mode: How messages are stored:
                - VERBATIM: Exact message text, zero AI cost (recommended for chat history)
                - EXTRACTED: AI extracts structured facts via AUDN pipeline
                - HYBRID: Both verbatim archive and extracted facts
            search_top_k: Number of memories to retrieve when loading history.
            search_query: Custom search query for retrieval. If None, retrieves
                most recent memories.
        """
        if not character_id and not external_id:
            raise ValueError("Either character_id or external_id must be provided")
        if character_id and external_id:
            raise ValueError("Provide character_id OR external_id, not both")

        self._client = HippoDid(api_key=api_key, base_url=base_url)
        self._search_top_k = search_top_k
        self._search_query = search_query
        self._memory_mode = memory_mode

        # Resolve character
        if external_id:
            result = self._client.upsert_by_external_id(external_id)
            self._character_id = result.id
        else:
            self._character_id = character_id  # type: ignore[assignment]

        # Set memory mode (no-op if already in the requested mode)
        if memory_mode:
            try:
                self._client.set_memory_mode(self._character_id, memory_mode)
            except Exception:
                pass  # Tier may not support it, or already set

    @property
    def messages(self) -> List[BaseMessage]:
        """Retrieve messages from HippoDid.

        Uses semantic search to find relevant memories. For VERBATIM mode,
        messages are stored with role prefixes (Human:/AI:) and parsed back.
        For EXTRACTED mode, memories are structured facts returned as
        SystemMessages providing context.
        """
        query = self._search_query or "recent conversation messages"
        results = self._client.search_memories(
            self._character_id,
            query,
            top_k=self._search_top_k,
        )

        messages: List[BaseMessage] = []
        for result in results:
            content = result.content
            if content.startswith("Human: "):
                messages.append(HumanMessage(content=content[7:]))
            elif content.startswith("AI: "):
                messages.append(AIMessage(content=content[4:]))
            else:
                messages.append(SystemMessage(content=content))

        return messages

    def add_message(self, message: BaseMessage) -> None:
        """Store a message in HippoDid.

        In VERBATIM mode: stores exact message with role prefix.
        In EXTRACTED mode: AUDN pipeline extracts structured facts.
        """
        if isinstance(message, HumanMessage):
            content = f"Human: {message.content}"
        elif isinstance(message, AIMessage):
            content = f"AI: {message.content}"
        else:
            content = message.content

        self._client.add_memory(self._character_id, content)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Store multiple messages as a single combined memory."""
        parts = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                parts.append(f"Human: {msg.content}")
            elif isinstance(msg, AIMessage):
                parts.append(f"AI: {msg.content}")
            else:
                parts.append(msg.content)

        combined = "\n".join(parts)
        self._client.add_memory(self._character_id, combined)

    def clear(self) -> None:
        """No-op. HippoDid memories are append-only by design."""

    # --- HippoDid-specific convenience methods ---

    def search(
        self,
        query: str,
        top_k: int = 10,
        categories: Optional[List[str]] = None,
    ) -> list:
        """Semantic search across this character's memories.

        Returns raw HippoDid SearchResult objects with relevance scores.
        """
        return self._client.search_memories(
            self._character_id,
            query,
            top_k=top_k,
            categories=categories,
        )

    def get_context(self, query: str) -> str:
        """Assemble full context (profile + memories) for a query.

        Returns a formatted prompt string ready for injection into a system prompt.
        """
        ctx = self._client.assemble_context(self._character_id, query)
        return ctx.formatted_prompt

    @property
    def character_id(self) -> str:
        """The HippoDid character ID backing this history."""
        return self._character_id
