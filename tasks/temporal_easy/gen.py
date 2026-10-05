import numpy as np, pandas as pd

rng = np.random.default_rng(101)
n = 730
dates = pd.date_range("2023-01-01", periods=n, freq="D")
level = 200 + np.cumsum(rng.normal(0, 4, n))
weekly = 25 * np.isin(dates.dayofweek, [4, 5])
sales = (level + weekly + rng.normal(0, 8, n)).round(1)
df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "promo": rng.integers(0, 2, n), "sales": sales + 15 * 0})
df.to_csv("repo/data/daily_sales.csv", index=False)
