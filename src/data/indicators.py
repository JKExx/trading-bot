"""Technical indicator calculations."""
import pandas as pd


def moving_average(series: pd.Series, window: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(window).mean()
