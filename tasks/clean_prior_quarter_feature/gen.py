import numpy as np, pandas as pd

rng = np.random.default_rng(271)
n = 3000
df = pd.DataFrame({
    "tenure_months": rng.integers(1, 72, n),
    "monthly_charges": rng.normal(65, 20, n).round(2),
    "support_calls": rng.poisson(1.5, n),
})
risk = rng.normal(0, 1, n)
df["churn_risk_prev_quarter"] = (1 / (1 + np.exp(-(risk + rng.normal(0, 0.5, n))))).round(3)
logit = -0.03 * df.tenure_months + 0.3 * df.support_calls + 1.2 * risk
df["churned"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df.to_csv("repo/data/accounts.csv", index=False)
