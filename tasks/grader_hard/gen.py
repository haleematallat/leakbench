import json, random

rng = random.Random(211)
rows = []
for i in range(500):
    ref = round(rng.uniform(-50, 50), 1)
    r = rng.random()
    if r < 0.4:
        ans = rng.choice([f"{ref}", f"{ref:,.1f}", f" {ref} "])
    elif r < 0.65:
        ans = f"{-ref}"  # sign error
    elif r < 0.85:
        ans = f"{ref * 10:.0f}" if ref * 10 == round(ref * 10) else f"{ref}"  # dropped the decimal point
    else:
        ans = f"{ref + rng.choice([-1, 1, 2]):.1f}"
    rows.append({"id": i, "question": f"Q{i}", "reference": f"{ref}", "model_answer": ans})
with open("repo/data/predictions.jsonl", "w") as f:
    f.writelines(json.dumps(r) + "\n" for r in rows)
