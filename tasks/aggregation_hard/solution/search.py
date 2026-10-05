import numpy as np
from sklearn.linear_model import LogisticRegression


def candidate_subsets(columns, n_candidates=300, size=5, seed=0):
    rng = np.random.default_rng(seed)
    return [list(rng.choice(columns, size=size, replace=False)) for _ in range(n_candidates)]


def best_subset(X_fit, y_fit, X_select, y_select, columns):
    """Pick the marker panel whose model scores best on the selection data."""
    best, best_score = None, -1.0
    for subset in candidate_subsets(columns):
        score = LogisticRegression(max_iter=1000).fit(X_fit[subset], y_fit).score(X_select[subset], y_select)
        if score > best_score:
            best, best_score = subset, score
    return best, best_score
