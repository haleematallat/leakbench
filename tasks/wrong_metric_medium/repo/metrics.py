import numpy as np
import pandas as pd


def accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    y_true = y_true.sort_index()  # stable order for reproducible reports
    return float((y_true.to_numpy() == np.asarray(y_pred)).mean())
