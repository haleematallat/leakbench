import numpy as np, pandas as pd

rng = np.random.default_rng(131)
n, k = 3000, 8
X = rng.normal(size=(n, k)).round(3)
p = 1 / (1 + np.exp(-(X @ rng.normal(size=k) * 0.6)))
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(k)])
df["churned"] = (rng.random(n) < p).astype(int)
df.to_csv("repo/data/customers.csv", index=False)
