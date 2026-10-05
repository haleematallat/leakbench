import numpy as np, pandas as pd

rng = np.random.default_rng(71)
n, p = 240, 3000
X = rng.normal(size=(n, p)).round(3)
logit = 0.8 * X[:, 0] - 0.6 * X[:, 1]
y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df = pd.DataFrame(X, columns=[f"gene_{i:04d}" for i in range(p)])
df["responder"] = y
df.to_csv("repo/data/expression.csv", index=False)
