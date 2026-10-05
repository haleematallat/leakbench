from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from metrics import overall_accuracy

DATA = Path(__file__).parent / "data" / "scans.csv"
TARGET = "positive"
FEATURES = [f"f{i}" for i in range(5)]


def evaluate() -> float:
    df = pd.read_csv(DATA)
    train, test = train_test_split(df, test_size=0.3, random_state=0, stratify=df["site"])
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(train[FEATURES], train[TARGET])
    return overall_accuracy(test, model.predict(test[FEATURES]), TARGET)
