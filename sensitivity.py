import costing as C

df = C.load_scores()
base = dict(C_FN=C.C_FN, C_FP=C.C_FP, BASE_RATE=C.BASE_RATE)

def run(label, **kw):
    for k, v in kw.items():
        setattr(C, k, v)
    t, c1 = C.best_single(df)
    lo, hi, c2 = C.best_pair(df)
    share, per_day, people = C.review_load(df, lo, hi)
    nothing = 1000 * C.BASE_RATE * C.C_FN
    print(f"{label:<22} solo={c1:7.1f} (t={t:.2f}) | con revision={c2:7.1f} "
          f"({lo:.2f}-{hi:.2f}, {100*share:4.1f}%, {people:.1f} pers.) | nada={nothing:7.1f}")
    for k, v in base.items():
        setattr(C, k, v)

for v in (30, 75, 150, 300, 600):
    run(f"C_FN = {v}", C_FN=v)
for v in (4, 12, 30, 60):
    run(f"C_FP = {v}", C_FP=v)
for v in (0.005, 0.01, 0.03, 0.10):
    run(f"BASE_RATE = {v}", BASE_RATE=v)