POSITIVE_CLASSES = ["Box_broken", "Open_package"]   # "compromised": set aside at intake
NEGATIVE_CLASSES = ["Box", "Package"]
SCORE_RULE = "max confidence over detections of any POSITIVE_CLASSES in the image; 0.0 if none"

# Costs in DOP, per unit. Source: owner interview + July 2025 inventory snapshot.
COST_FN_DOP = 7000           # damaged unit passes: replacement at cost 4400 + two-way local freight 2500; claim window closed
COST_FP_DOP = 40             # intact unit sent to full unpack: 10 min x RD$3/min loaded wage
COST_REVIEW_DOP = 40         # human review = full 10-min unpack, catches everything
COST_FAIL_CONFIRM_DOP = 10   # model FAIL: photo is the claim evidence, 3-min confirmation glance
MIN_PER_REVIEW = 10
MIN_PER_FAIL_CONFIRM = 3
WAGE_DOP_PER_MIN = 3         # ~RD$32,000/month loaded / ~190 h

BASE_RATE = 0.01             # ~1 damaged per 100 received, per owner
BASE_RATE_SWEEP = [0.005, 0.01, 0.03, 0.05, 0.10, 0.15]
CONTAINER_SIZES = {"20ft": 200, "40ft_low": 400, "40ft_high": 600}   # per owner; no daily flow, receiving is per container
STAFF = 2
HOURS_PER_DAY = 7            # 8 h minus 1 h lunch
CLAIM_WINDOW_DAYS = 7
DOP_PER_EUR = 68.67          # poundsterlinglive.com, 2026-10-23

# A model FAIL files a claim on the photo without unpacking. On an intact unit
# that is a false claim.
COST_FALSE_CLAIM_DOP = 500   # admin to file and reverse a claim on an intact unit (~1 h at ~RD$250/h) plus credibility with the supplier/forwarder; owner to confirm
COST_FALSE_CLAIM_SWEEP = [100, 500, 2000]

# Chosen operating point.
# t_high: just above the highest score of any intact test image (0.934), so no
# intact unit is failed on the photo. t_low: the symmetric rule (just below the
# lowest detected damaged score, 0.088) gives 0.08, but that sends 10.5% of
# boxes to review to catch one extra damaged box in 62, which the test set
# cannot resolve; 0.20 is the low end of the exact-minimum plateau and halves
# the extra load. Both edges rest on one test image each; re-derive from dock
# data.
CHOSEN_T_LOW = 0.20
CHOSEN_T_HIGH = 0.94
