import pandas as pd


def target_encode(df: pd.DataFrame, column: str, target: str) -> pd.DataFrame:
    """Replace a high-cardinality category with the mean target of its group."""
    df = df.copy()
    df[column + "_te"] = df.groupby(column)[target].transform("mean")
    return df.drop(columns=column)
