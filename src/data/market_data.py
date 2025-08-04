"""Utilities for fetching and preparing market data."""
from typing import Any
import yfinance as yf


def fetch_ohlc(symbol: str, period: str = "1d") -> Any:
    """Fetch OHLC data for a symbol."""
    return yf.download(symbol, period=period)
