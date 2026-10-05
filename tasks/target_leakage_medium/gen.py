import numpy as np, pandas as pd

rng = np.random.default_rng(51)
n = 3000
df = pd.DataFrame({
    "income": rng.lognormal(10.8, 0.4, n).round(0),
    "loan_amount": rng.uniform(2000, 40000, n).round(0),
    "credit_score": rng.normal(680, 60, n).round(0),
    "years_employed": rng.integers(0, 30, n),
})
logit = -0.012 * (df.credit_score - 680) + 2.5 * df.loan_amount / df.income - 0.04 * df.years_employed - 0.8
df["defaulted"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
paid_share = np.where(df.defaulted == 1, rng.uniform(0.05, 0.7, n), rng.uniform(0.95, 1.05, n))
df["total_repaid"] = (df.loan_amount * paid_share).round(0)
df.to_csv("repo/data/loans.csv", index=False)
