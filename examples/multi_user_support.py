"""Multi-user: One HippoDid character per customer, resolved by external ID.

Requirements: pip install langchain-hippodid langchain langchain-anthropic
"""

from langchain.chains import ConversationChain
from langchain_anthropic import ChatAnthropic

from langchain_hippodid import HippoDidMemory


def get_memory_for_user(user_id: str) -> HippoDidMemory:
    """Each user gets their own persistent memory character."""
    return HippoDidMemory(
        api_key="hd_...",
        external_id=user_id,  # Auto-creates character if first time
        memory_mode="EXTRACTED",  # AI extracts structured facts
    )


# Different users, different memories, same API
llm = ChatAnthropic(model="claude-sonnet-4-20250514")

memory_yang = get_memory_for_user("user_yang_001")
chain_yang = ConversationChain(llm=llm, memory=memory_yang)
chain_yang.predict(input="I prefer Python and dark mode")

memory_sarah = get_memory_for_user("user_sarah_002")
chain_sarah = ConversationChain(llm=llm, memory=memory_sarah)
chain_sarah.predict(input="I'm a frontend developer who loves TypeScript")

# Each user's preferences persist independently
