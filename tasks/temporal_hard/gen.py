import numpy as np, pandas as pd

rng = np.random.default_rng(121)
n = 730
dates = pd.date_range("2023-01-01", periods=n, freq="D")
level = np.empty(n); level[0] = 200
for t in range(1, n):  # demand drifts but reverts to its long-run mean
    level[t] = 200 + 0.95 * (level[t - 1] - 200) + rng.normal(0, 6)
weekly = 25 * np.isin(dates.dayofweek, [4, 5])
promo = rng.integers(0, 2, n)
sales = (level + weekly + 15 * promo + rng.normal(0, 8, n)).round(1)
pd.DataFrame({"date": dates.strftime("%d/%m/%Y"), "promo": promo, "sales": sales}).to_csv("repo/data/daily_sales.csv", index=False)
