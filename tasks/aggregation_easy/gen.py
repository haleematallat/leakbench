import numpy as np, pandas as pd

rng = np.random.default_rng(161)
n, k = 3000, 6
X = rng.normal(size=(n, k)).round(3)
p = 1 / (1 + np.exp(-(X @ rng.normal(size=k) * 0.5)))
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(k)])
df["label"] = (rng.random(n) < p).astype(int)
for c in ["f1", "f4"]:
    df.loc[rng.random(n) < 0.2, c] = np.nan
df.to_csv("repo/data/claims.csv", index=False)
