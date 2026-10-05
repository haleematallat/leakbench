from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

DATA = Path(__file__).parent / "data" / "visits.csv"
TARGET = "diagnosis"
FEATURES = [f"x{i}" for i in range(8)]


def evaluate() -> float:
    df = pd.read_csv(DATA)
    train_idx, test_idx = next(GroupShuffleSplit(test_size=0.3, random_state=0).split(df, groups=df["patient_id"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(train[FEATURES], train[TARGET])
    return float((model.predict(test[FEATURES]) == test[TARGET]).mean())
