"""RAG + persistent memory: retrieve docs AND remember user across sessions.

Requirements: pip install langchain-hippodid langchain langchain-anthropic
"""

from langchain_hippodid import HippoDidChatMessageHistory

history = HippoDidChatMessageHistory(
    api_key="hd_...",
    external_id="user_yang_001",
    memory_mode="EXTRACTED",
)

# Search memories for relevant context about this specific user
user_context = history.search("what does this user prefer for travel?", top_k=5)
for result in user_context:
    print(f"[{result.salience}] {result.content}")

# Combine with RAG retrieval for a personalized answer
full_context = history.get_context("travel recommendations based on user history")
print(f"\nAssembled context ({len(full_context)} chars):")
print(full_context[:200])
