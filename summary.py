from typing import List, Dict
from indic_assistant.llm.client import LLMClient
from indic_assistant.utils.logger import logger

class SummaryGenerator:
    """Generates a concise summary of the conversation session on exit."""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate_summary(self, history: List[Dict[str, str]]) -> str:
        """Uses LLM to summarize the conversation session history."""
        if not history:
            return "No conversation occurred in this session."

        logger.info("Generating session summary...")
        
        # Build prompt formatting history
        history_text = ""
        for idx, turn in enumerate(history):
            role = "User" if turn["role"] == "user" else "Assistant"
            history_text += f"{role}: {turn['content']}\n"
            
        prompt = (
            "Summarize the following conversation history concisely in 2-3 sentences. "
            "Focus on the main topics discussed and any outstanding questions:\n\n"
            f"{history_text}\nSummary:"
        )
        
        try:
            # We bypass regular history when generating session summaries
            summary = self.llm_client.query(prompt, history=[])
            logger.info(f"Generated session summary: {summary}")
            return summary
        except Exception as e:
            logger.error(f"Failed to generate summary: {e}")
            return "Failed to generate conversation summary."
