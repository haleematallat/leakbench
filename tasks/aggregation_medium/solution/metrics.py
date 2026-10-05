import pandas as pd


def overall_accuracy(test: pd.DataFrame, preds, target: str) -> float:
    """Accuracy across all test patients."""
    correct = pd.Series(preds == test[target].to_numpy(), index=test.index)
    return float(correct.mean())
