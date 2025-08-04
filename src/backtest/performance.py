"""Performance metric helpers."""
import pandas as pd


def cumulative_returns(returns: pd.Series) -> pd.Series:
    """Calculate cumulative returns."""
    return (1 + returns.fillna(0)).cumprod() - 1
