import numpy as np, pandas as pd

rng = np.random.default_rng(61)
n, zips = 3000, 600
zip_effect = rng.normal(0, 0.4, zips)
z = rng.integers(0, zips, n)
X = rng.normal(size=(n, 5)).round(3)
logit = X @ np.array([0.6, -0.4, 0.3, 0.0, 0.2]) + zip_effect[z]
df = pd.DataFrame(X, columns=["age_z", "income_z", "visits_z", "tenure_z", "spend_z"])
df.insert(0, "zip_code", [f"{10000 + 37 * i:05d}" for i in z])
df["responded"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
df.to_csv("repo/data/campaign.csv", index=False)
