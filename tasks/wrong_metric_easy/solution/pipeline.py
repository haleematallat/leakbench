from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

DATA = Path(__file__).parent / "data" / "customers.csv"
TARGET = "churned"


def evaluate() -> float:
    df = pd.read_csv(DATA)
    features = [c for c in df.columns if c != TARGET]
    X_train, X_test, y_train, y_test = train_test_split(df[features], df[TARGET], test_size=0.3, random_state=0)
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return float((preds == y_test).mean())
