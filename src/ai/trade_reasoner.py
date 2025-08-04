"""Use an LLM to analyze potential trades."""
from .llm_client import OllamaClient
from . import prompt_templates


class TradeReasoner:
    def __init__(self):
        self.client = OllamaClient()

    def analyze_entry(self, context):
        prompt = prompt_templates.TRADE_ENTRY_PROMPT.format(**context)
        return self.client.generate(prompt)
