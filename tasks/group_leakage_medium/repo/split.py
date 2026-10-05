import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from config import GROUP_KEY, TEST_SIZE


def grouped_split(df: pd.DataFrame, seed: int = 0):
    train_idx, test_idx = next(GroupShuffleSplit(test_size=TEST_SIZE, random_state=seed).split(df, groups=df[GROUP_KEY]))
    return df.iloc[train_idx], df.iloc[test_idx]
