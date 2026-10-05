from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from features import build_table

TARGET = "churned"


def evaluate() -> float:
    df = build_table()
    train, test = train_test_split(df, test_size=0.3, random_state=0)
    features = [c for c in df.columns if c not in (TARGET, "customer_id")]
    model = RandomForestClassifier(n_estimators=200, random_state=0)
    model.fit(train[features], train[TARGET])
    return float((model.predict(test[features]) == test[TARGET]).mean())
