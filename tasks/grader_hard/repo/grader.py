import re


def canonical(text: str) -> str:
    """Strip formatting so '1,000', ' 1000 ' and '1000.' compare equal."""
    return re.sub(r"[^0-9a-z]", "", text.lower())


def is_correct(answer: str, reference: str) -> bool:
    """Numeric answers are compared by value."""
    return canonical(answer) == canonical(reference)
