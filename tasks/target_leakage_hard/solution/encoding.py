import pandas as pd


def target_encode(train: pd.DataFrame, test: pd.DataFrame, column: str, target: str):
    """Replace a high-cardinality category with the mean target of its group, learned on train only."""
    means = train.groupby(column)[target].mean()
    prior = train[target].mean()
    train, test = train.copy(), test.copy()
    for df in (train, test):
        df[column + "_te"] = df[column].map(means).fillna(prior)
    return train.drop(columns=column), test.drop(columns=column)
