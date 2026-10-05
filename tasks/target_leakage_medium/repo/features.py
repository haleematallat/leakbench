import pandas as pd

FEATURES = ["income", "loan_amount", "credit_score", "years_employed", "debt_to_income", "repayment_ratio"]


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["debt_to_income"] = df.loan_amount / df.income
    df["repayment_ratio"] = df.total_repaid / df.loan_amount
    return df
