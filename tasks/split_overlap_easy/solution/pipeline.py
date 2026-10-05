from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "data" / "customers.csv"
TARGET = "churned"


def load() -> pd.DataFrame:
    return pd.read_csv(DATA)


def split(df: pd.DataFrame):
    df = df.drop_duplicates()
    return train_test_split(df, test_size=0.3, random_state=0)


def evaluate() -> float:
    train, test = split(load())
    features = [c for c in train.columns if c != TARGET]
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(train[features], train[TARGET])
    return float((model.predict(test[features]) == test[TARGET]).mean())
