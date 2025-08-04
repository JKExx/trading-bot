"""Data validation helpers."""
import pandas as pd


def validate_dataframe(df: pd.DataFrame) -> bool:
    """Check for missing values."""
    return not df.isnull().any().any()
