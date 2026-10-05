import numpy as np, pandas as pd

rng = np.random.default_rng(181)
n, k = 4000, 30  # the candidate markers turn out to carry no signal at all
X = rng.normal(size=(n, k)).round(3)
df = pd.DataFrame(X, columns=[f"m{i:02d}" for i in range(k)])
df["label"] = rng.integers(0, 2, n)
df.to_csv("repo/data/markers.csv", index=False)
