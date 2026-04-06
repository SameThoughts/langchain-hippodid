"""Basic: LangChain conversation with persistent HippoDid memory.

Requirements: pip install langchain-hippodid langchain-anthropic
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_hippodid import HippoDidMemory

# Memory persists across sessions — restart your app and it remembers
memory = HippoDidMemory(
    api_key="hd_...",
    character_id="your-character-uuid",
    memory_mode="VERBATIM",  # Exact chat storage, zero AI cost
)

llm = ChatAnthropic(model="claude-sonnet-4-20250514")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ]
)

chain = prompt | llm

chain_with_memory = RunnableWithMessageHistory(
    chain,
    memory.get_history,
    input_messages_key="input",
    history_messages_key="history",
)

# First session
chain_with_memory.invoke(
    {"input": "My name is Yang and I'm building an AI startup called SameThoughts"},
    config={"configurable": {"session_id": "default"}},
)

# ... restart app, new session, same character_id ...
# Memory survives because it's stored in HippoDid cloud.
