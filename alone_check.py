import json
import numpy as np

R = json.load(open("outputs/costing_results.json"))
Fp, Fn = np.array(R["F_pos"]), np.array(R["F_neg"])
c_fn, d = R["params"]["c_fn"], R["costs"]["false_dispatch_eur"]
print("max proxy score (s_neg):", R["thresholds"][int(np.argmax(Fn == 0))])
for b in [1e-5, 1e-4, 1e-3, 1e-2]:
    alone = b * (1 - Fp) * c_fn + (1 - b) * Fn * d
    k = int(alone.argmin())
    print(f"b={b:g}: model alone {alone[k]*1000:.2f} EUR per 1000 frames at threshold {R['thresholds'][k]}")