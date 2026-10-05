from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from config import FEATURES, TARGET
from split import grouped_split

DATA = Path(__file__).parent / "data" / "visits.csv"


def evaluate() -> float:
    train, test = grouped_split(pd.read_csv(DATA))
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(train[FEATURES], train[TARGET])
    return float((model.predict(test[FEATURES]) == test[TARGET]).mean())
