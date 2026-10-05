import json, random

rng = random.Random(191)
cities = ["Paris", "Lima", "Oslo", "Cairo", "Hanoi", "Quito", "Accra", "Dhaka", "Seoul", "Rome", "Bern", "Doha"]
rows = []
for i in range(400):
    ref = rng.choice(cities)
    other = rng.choice([c for c in cities if c != ref])
    r = rng.random()
    if r < 0.5:
        ans = rng.choice([ref, ref.lower(), f" {ref}.", ref.upper()])
    elif r < 0.75:  # hedged: names the right city alongside a wrong one
        ans = rng.choice([f"{other} or {ref}", f"Either {ref} or {other}", f"{ref}/{other}"])
    else:
        ans = other
    rows.append({"id": i, "question": f"Q{i}: Which city ...?", "reference": ref, "model_answer": ans})
with open("repo/data/predictions.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
