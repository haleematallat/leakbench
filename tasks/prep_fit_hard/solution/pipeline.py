from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline

DATA = Path(__file__).parent / "data" / "patients.csv"
TARGET = "readmitted"


def evaluate() -> float:
    df = pd.read_csv(DATA)
    train, test = train_test_split(df, test_size=0.3, random_state=0)
    features = [c for c in df.columns if c != TARGET]
    model = make_pipeline(SimpleImputer(strategy="mean"), GradientBoostingClassifier(random_state=0))
    model.fit(train[features], train[TARGET])
    return float(roc_auc_score(test[TARGET], model.predict_proba(test[features])[:, 1]))
