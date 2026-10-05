import numpy as np, pandas as pd

rng = np.random.default_rng(171)
parts = []
for site, (n, signal) in enumerate([(3000, 0.3)] + [(40, 3.0)] * 9):
    X = rng.normal(size=(n, 5)).round(3)
    p = 1 / (1 + np.exp(-(X[:, 0] * signal + X[:, 1] * signal / 2)))
    d = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    d.insert(0, "site", f"site_{site:02d}")
    d["positive"] = (rng.random(n) < p).astype(int)
    parts.append(d)
pd.concat(parts).sample(frac=1, random_state=0).to_csv("repo/data/scans.csv", index=False)
