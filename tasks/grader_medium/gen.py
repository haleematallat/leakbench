import json, random

rng = random.Random(201)
rows = []
for i in range(500):
    a, b = rng.randint(10, 99), rng.randint(10, 99)
    ref = a * b
    r = rng.random()
    if r < 0.45:
        ans = str(ref)
    elif r < 0.7:
        ans = str(ref + rng.choice([-10, -1, 1, 10, 100]))
    else:  # not a bare number
        ans = rng.choice([f"about {ref}", f"{ref} (approx)", "I cannot compute this.", f"The answer is {ref + 1}", ""])
    rows.append({"id": i, "question": f"What is {a} * {b}?", "reference": ref, "model_answer": ans})
with open("repo/data/predictions.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
