from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupShuffleSplit

from data_utils import load_visits

TARGET = "diagnosis"
FEATURES = [f"x{i}" for i in range(8)]


def evaluate() -> float:
    df = load_visits()
    train_idx, test_idx = next(GroupShuffleSplit(test_size=0.3, random_state=0).split(df, groups=df["patient_id"]))
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(train[FEATURES], train[TARGET])
    return float((model.predict(test[FEATURES]) == test[TARGET]).mean())
