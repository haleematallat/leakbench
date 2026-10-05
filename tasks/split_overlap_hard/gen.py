import numpy as np, pandas as pd

rng = np.random.default_rng(21)
n, k = 2500, 10
X = rng.normal(size=(n, k)).round(3)
p = 1 / (1 + np.exp(-(X @ rng.normal(size=k) * 0.5)))
df = pd.DataFrame(X, columns=[f"sensor_{i}" for i in range(k)])
df["fault"] = (rng.random(n) < p).astype(int)
df.to_csv("repo/data/readings.csv", index=False)
