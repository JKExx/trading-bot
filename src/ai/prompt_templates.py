"""Prompt templates for LLM interactions."""

TRADE_ENTRY_PROMPT = (
    "Analyze the following trade setup for entry decisions: {details}"
)

TRADE_EXIT_PROMPT = (
    "Review the current position and advise on exit: {details}"
)
