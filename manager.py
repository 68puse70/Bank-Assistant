from typing import List, Dict, Any
from indic_assistant.utils.logger import logger

class MemoryManager:
    """Manages short-term conversation context and handles long-term memory retrieval."""

    def __init__(self, max_short_term_turns: int = 10):
        self.max_short_term_turns = max_short_term_turns
        self.history: List[Dict[str, str]] = []
        self.user_profile: Dict[str, Any] = {}
        logger.info(f"Initialized MemoryManager with max short term turns: {self.max_short_term_turns}")

    def add_interaction(self, role: str, content: str):
        """Adds a message to the short-term context window."""
        self.history.append({"role": role, "content": content})
        # Keep only the last N turns (1 turn = 1 user + 1 assistant message)
        if len(self.history) > self.max_short_term_turns * 2:
            self.history = self.history[-(self.max_short_term_turns * 2):]

    def get_context(self) -> List[Dict[str, str]]:
        """Returns the current short-term conversation history."""
        return self.history

    def clear(self):
        """Clears short-term context."""
        self.history.clear()
        logger.info("Short-term conversation history cleared.")

    def set_long_term_profile(self, profile_data: Dict[str, Any]):
        """Sets user details for long-term personalization."""
        self.user_profile.update(profile_data)
        logger.info("Long-term profile updated.")

    def get_long_term_summary(self) -> str:
        """Retrieves a formatted summary of user profile/facts for LLM system prompt injection."""
        if not self.user_profile:
            return ""
        
        profile_str = "\n".join([f"- {k}: {v}" for k, v in self.user_profile.items()])
        return f"\nRelevant User Profile Facts:\n{profile_str}\n"
