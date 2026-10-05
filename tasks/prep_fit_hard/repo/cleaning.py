import pandas as pd

LAB_COLUMNS = ["bmi", "systolic_bp", "glucose", "cholesterol"]


def impute_labs(df: pd.DataFrame, by: str) -> pd.DataFrame:
    """Fill missing lab values with the mean of comparable patients."""
    df = df.copy()
    for col in LAB_COLUMNS:
        df[col] = df[col].fillna(df.groupby(by)[col].transform("mean"))
    return df
