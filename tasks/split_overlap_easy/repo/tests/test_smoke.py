import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))
from pipeline import evaluate


def test_evaluate_returns_a_probability():
    assert 0.0 <= evaluate() <= 1.0
