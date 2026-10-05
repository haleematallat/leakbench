from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

DATA = Path(__file__).parent / "data" / "snapshots.csv"
TARGET = "churned"
FEATURES = [f"f{i}" for i in range(6)]


def evaluate() -> float:
    # NOTE: the same customer shows up many times (one row per monthly snapshot)
    df = pd.read_csv(DATA)
    splitter = GroupShuffleSplit(test_size=0.3, random_state=0)
    train_idx, test_idx = next(splitter.split(df, groups=df["customer_id"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(train[FEATURES], train[TARGET])
    return float((model.predict(test[FEATURES]) == test[TARGET]).mean())
