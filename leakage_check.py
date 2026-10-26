from pathlib import Path

root = Path("_explore/whwcy_v5")

def stems(split):
    return {p.name.split(".rf.")[0] for p in (root / split / "images").glob("*")}

tr, va, te = stems("train"), stems("valid"), stems("test")
print("originales train:", len(tr), "| valid:", len(va), "| test:", len(te))
print("en comun train-valid:", len(tr & va))
print("en comun train-test:", len(tr & te))
print("en comun valid-test:", len(va & te))