from pathlib import Path

import pandas as pd

DATA = Path(__file__).parent / "data"


def build_table() -> pd.DataFrame:
    """Customer table enriched with their support-ticket history."""
    customers = pd.read_csv(DATA / "customers.csv")
    tickets = pd.read_csv(DATA / "tickets.csv")
    return customers.merge(tickets, on="customer_id", how="left")
