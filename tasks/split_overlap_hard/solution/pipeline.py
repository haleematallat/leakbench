from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from preprocess import augment, clean

DATA = Path(__file__).parent / "data" / "readings.csv"
TARGET = "fault"


def evaluate() -> float:
    df = clean(pd.read_csv(DATA))
    train, test = train_test_split(df, test_size=0.3, random_state=0)
    train = augment(train)
    features = [c for c in df.columns if c != TARGET]
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(train[features], train[TARGET])
    return float((model.predict(test[features]) == test[TARGET]).mean())
