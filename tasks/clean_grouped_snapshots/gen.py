import numpy as np, pandas as pd

rng = np.random.default_rng(251)
customers, k = 800, 6
base = rng.normal(size=(customers, k))
churn = (rng.random(customers) < 1 / (1 + np.exp(-(base[:, 0] - 0.7 * base[:, 1])))).astype(int)
rows = []
for c in range(customers):
    for m in range(rng.integers(3, 7)):
        rows.append([f"C{c:05d}", m, *(base[c] + rng.normal(0, 0.4, k)).round(3), churn[c]])
pd.DataFrame(rows, columns=["customer_id", "month", *[f"f{i}" for i in range(k)], "churned"]).sample(frac=1, random_state=0).to_csv("repo/data/snapshots.csv", index=False)
