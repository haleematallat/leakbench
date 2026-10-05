from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
EXPORTS = ["export_2023.csv", "export_2024.csv"]


def load_customers() -> pd.DataFrame:
    """All customers from the yearly CRM exports."""
    frames = [pd.read_csv(DATA / name) for name in EXPORTS]
    return pd.concat(frames, ignore_index=True)
