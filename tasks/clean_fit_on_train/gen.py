import numpy as np, pandas as pd

rng = np.random.default_rng(291)
n = 3000
df = pd.DataFrame({
    "age": rng.integers(20, 85, n),
    "bmi": rng.normal(27, 5, n).round(1),
    "systolic_bp": rng.normal(130, 18, n).round(0),
    "glucose": rng.normal(105, 25, n).round(0),
    "cholesterol": rng.normal(200, 35, n).round(0),
})
logit = 0.04 * (df.age - 50) + 0.05 * (df.bmi - 27) + 0.015 * (df.glucose - 105) - 0.8
df["readmitted"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
for col in ["bmi", "systolic_bp", "glucose", "cholesterol"]:  # lab results often missing
    df.loc[rng.random(n) < 0.45, col] = np.nan
df.to_csv("repo/data/patients.csv", index=False)
