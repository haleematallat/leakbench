import numpy as np, pandas as pd

rng = np.random.default_rng(231)
patients, k = 500, 8
effect = rng.normal(size=(patients, k))  # stable per-patient physiology
diagnosis = (rng.random(patients) < 1 / (1 + np.exp(-(effect[:, 0] * 0.9 - effect[:, 1] * 0.6)))).astype(int)
rows = []
for p in range(patients):
    for v in range(rng.integers(2, 9)):
        rows.append([f"P{p:04d}", f"V{p:04d}-{v}", *(effect[p] + rng.normal(0, 0.3, k)).round(3), diagnosis[p]])
df = pd.DataFrame(rows, columns=["patient_id", "visit_id", *[f"x{i}" for i in range(k)], "diagnosis"]).sample(frac=1, random_state=0)
df.to_csv("repo/data/visits.csv", index=False)
