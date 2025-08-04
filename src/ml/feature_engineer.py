"""Feature engineering utilities."""
import pandas as pd


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple indicators to DataFrame."""
    df = df.copy()
    df['sma_10'] = df['close'].rolling(10).mean()
    return df
