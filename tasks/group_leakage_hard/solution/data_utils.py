from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"
EXPORTS = ["visits_legacy.csv", "visits_new.csv", "visits_mobile.csv"]


def load_visits() -> pd.DataFrame:
    """Visits from every EHR export."""
    df = pd.concat([pd.read_csv(DATA / name) for name in EXPORTS], ignore_index=True)
    df["patient_id"] = df["patient_id"].str.strip().str.upper().str.replace("-", "", regex=False)
    return df
