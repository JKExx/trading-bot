"""Simple backtesting engine."""
from typing import Callable, Sequence
import pandas as pd


def run_backtest(data: pd.DataFrame, strategy: Callable[[pd.DataFrame], Sequence[int]]) -> pd.Series:
    """Apply strategy signals to data and calculate returns."""
    signals = strategy(data)
    return data['close'].pct_change().shift(-1) * signals
