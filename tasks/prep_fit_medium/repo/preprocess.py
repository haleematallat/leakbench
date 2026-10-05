import pandas as pd
from sklearn.utils import resample

TARGET = "fraud"


def balance(df: pd.DataFrame, seed: int = 0) -> pd.DataFrame:
    """Upsample the fraud class so the model sees as many frauds as normal transactions."""
    majority = df[df[TARGET] == 0]
    minority = df[df[TARGET] == 1]
    upsampled = resample(minority, replace=True, n_samples=len(majority), random_state=seed)
    return pd.concat([majority, upsampled]).sample(frac=1, random_state=seed)


def load(path) -> pd.DataFrame:
    return balance(pd.read_csv(path))
