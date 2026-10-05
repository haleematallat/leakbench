import numpy as np, pandas as pd

rng = np.random.default_rng(11)
n, k = 4000, 8
X = rng.normal(size=(n, k)).round(3)
p = 1 / (1 + np.exp(-(X @ rng.normal(size=k) * 0.6)))
df = pd.DataFrame(X, columns=[f"f{i}" for i in range(k)])
df.insert(0, "customer_id", np.arange(100000, 100000 + n))
df["churned"] = (rng.random(n) < p).astype(int)
# each yearly export is a rolling 3000-customer window, so the two overlap
df.iloc[:3000].sample(frac=1, random_state=1).to_csv("repo/data/export_2023.csv", index=False)
df.iloc[1000:].sample(frac=1, random_state=2).to_csv("repo/data/export_2024.csv", index=False)
