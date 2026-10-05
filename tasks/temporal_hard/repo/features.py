import pandas as pd

FEATURES = ["dow", "promo", "sales_lag_1", "sales_lag_7", "sales_7d_avg"]


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calendar features plus recent sales history. Expects rows in time order."""
    df = df.copy()
    df["dow"] = pd.to_datetime(df["date"], dayfirst=True).dt.dayofweek
    df["sales_lag_1"] = df["sales"].shift(1)
    df["sales_lag_7"] = df["sales"].shift(7)
    df["sales_7d_avg"] = df["sales"].shift(1).rolling(7).mean()
    return df.dropna(subset=FEATURES)
