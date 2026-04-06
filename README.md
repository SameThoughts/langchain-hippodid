# langchain-hippodid

Persistent character memory for LangChain agents — powered by [HippoDid](https://hippodid.com).

Add cloud-persistent, character-scoped memory to any LangChain chain in a few lines.

## Install

```bash
pip install langchain-hippodid
```

## Quick Start

```python
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_hippodid import HippoDidMemory

memory = HippoDidMemory(api_key="hd_...", character_id="your-character-uuid")

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}"),
])

chain_with_memory = RunnableWithMessageHistory(
    prompt | ChatAnthropic(model="claude-sonnet-4-20250514"),
    memory.get_history,
    input_messages_key="input",
    history_messages_key="history",
)

chain_with_memory.invoke(
    {"input": "My name is Yang, I'm building SameThoughts"},
    config={"configurable": {"session_id": "default"}},
)
# Memory persists to HippoDid cloud — survives restarts, redeploys, everything.
```

## Why HippoDid?

| Feature | Default LangChain | HippoDidMemory |
|---|---|---|
| Persistence | In-process only | Cloud-persistent across sessions |
| Scope | Global | Character-scoped (one per user) |
| Retrieval | Flat chat log | Semantic search across structured categories |
| Context assembly | Append all messages | Profile + category summaries + relevant facts |
| LLM provider | N/A | Works with any provider (BYOK) |

## External ID Pattern — One Character Per User

Auto-create and resolve characters by your system's user ID:

```python
from langchain_hippodid import HippoDidMemory

def get_memory(user_id: str) -> HippoDidMemory:
    return HippoDidMemory(
        api_key="hd_...",
        external_id=user_id,  # Auto-creates character if first time
    )

memory = get_memory("user_yang_001")
```

## Memory Modes

| Mode | Behavior | Best For |
|---|---|---|
| `VERBATIM` | Stores exact message text, zero AI cost | Chat history replay |
| `EXTRACTED` | AI extracts structured facts via AUDN pipeline | Long-term user knowledge |
| `HYBRID` | Both verbatim archive and extracted facts | Full fidelity + structured recall |

```python
memory = HippoDidMemory(
    api_key="hd_...",
    character_id="...",
    memory_mode="EXTRACTED",
)
```

## Advanced: Custom Search & Context Assembly

Use `HippoDidChatMessageHistory` directly for fine-grained control:

```python
from langchain_hippodid import HippoDidChatMessageHistory

history = HippoDidChatMessageHistory(
    api_key="hd_...",
    character_id="your-character-uuid",
)

# Semantic search across memories
results = history.search("travel preferences", top_k=5)

# Assemble full context (profile + memories) for a system prompt
context = history.get_context("current user preferences")
```

## Links

- [HippoDid](https://hippodid.com) — Persistent memory for AI agents
- [Documentation](https://docs.hippodid.com/guides/langchain)
- [GitHub](https://github.com/SameThoughts/langchain-hippodid)
- [PyPI](https://pypi.org/project/langchain-hippodid/)
- [hippodid Python SDK](https://pypi.org/project/hippodid/)
