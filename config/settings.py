"""Application settings and environment variable loading."""
from dotenv import load_dotenv
import os

load_dotenv()

BROKER_API_KEY = os.getenv("BROKER_API_KEY", "")
