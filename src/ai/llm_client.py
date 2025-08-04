"""Client for interacting with a local LLM server."""
from typing import Dict
import requests


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral:7b-instruct"):
        self.base_url = base_url
        self.model = model

    def generate(self, prompt: str, context: Dict | None = None) -> str:
        """Send a prompt to the LLM and return the response."""
        payload = {"model": self.model, "prompt": prompt, "context": context or {}}
        response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
        return response.json().get("response", "")
