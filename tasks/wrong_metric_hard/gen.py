import numpy as np, pandas as pd

rng = np.random.default_rng(151)
n, k = 3000, 8
X = rng.normal(size=(n, k)).round(3)
p = 1 / (1 + np.exp(-(X @ rng.normal(size=k) * 0.6)))
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(k)])
df["will_upgrade"] = np.where(rng.random(n) < p, "yes", "no")
df.to_csv("repo/data/users.csv", index=False)
