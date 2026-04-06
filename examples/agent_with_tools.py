"""Agent with tools + persistent memory across sessions.

Requirements: pip install langchain-hippodid langchain langchain-anthropic
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_hippodid import HippoDidChatMessageHistory

history = HippoDidChatMessageHistory(
    api_key="hd_...",
    character_id="your-character-uuid",
)

# Use assemble_context for rich system prompt with memories
context = history.get_context("current user preferences and recent interactions")

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", f"You are a helpful assistant.\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm = ChatAnthropic(model="claude-sonnet-4-20250514")
# Add tools, create agent with create_tool_calling_agent, wrap in AgentExecutor, etc.
