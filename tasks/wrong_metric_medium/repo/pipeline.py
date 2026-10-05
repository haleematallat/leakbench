from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from metrics import accuracy

DATA = Path(__file__).parent / "data" / "customers.csv"
TARGET = "churned"


def evaluate() -> float:
    df = pd.read_csv(DATA)
    features = [c for c in df.columns if c != TARGET]
    train, test = train_test_split(df, test_size=0.3, random_state=0)
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(train[features], train[TARGET])
    return accuracy(test[TARGET], model.predict(test[features]))
