import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from data_utils import load_customers

TARGET = "churned"
FEATURES = [f"f{i}" for i in range(8)]


def evaluate() -> float:
    df = load_customers()
    train, test = train_test_split(df, test_size=0.3, random_state=0)
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(train[FEATURES], train[TARGET])
    return float((model.predict(test[FEATURES]) == test[TARGET]).mean())
