import numpy as np, pandas as pd

rng = np.random.default_rng(81)
n, k = 4000, 8
X = rng.normal(size=(n, k)).round(3)
logit = X @ rng.normal(size=k) * 0.7 - 3.0
df = pd.DataFrame(X, columns=[f"txn_{i}" for i in range(k)])
df["fraud"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df.to_csv("repo/data/transactions.csv", index=False)
