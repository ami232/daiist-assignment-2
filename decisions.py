ROBOFLOW_WORKSPACE = "a-yvlja"
ROBOFLOW_PROJECT = "helmet-and-non-helmet-abb47"
ROBOFLOW_VERSION = 1

POSITIVE_CLASSES: list[str] | None = ["Without Helmet"]

# Assumed: ~3% of riders leave the depot without a helmet.
BASE_RATE: float | None = 0.03

def _total(*components: float | None) -> float | None:
    return None if any(c is None for c in components) else float(sum(components))

# False negative: rider without a helmet let through.
FN_SAFETY_FINE: float | None = 2.00            # 1% chance of roadside inspection x ~EUR 200 fine
FN_EXPECTED_ACCIDENT_COST: float | None = 3.00 # 0.02% chance of accident per trip x ~EUR 15,000
                                               # (sick pay, insurance, liability)

# False positive: rider with a helmet stopped anyway.
FP_RIDER_TIME: float | None = 0.40             # 1 min lost x EUR 0.40/min (~EUR 24/h loaded cost)
FP_DELIVERY_DELAY: float | None = 0.25         # 5% chance the delay makes a delivery late x EUR 5

# True positive: rider without a helmet correctly stopped.
TP_RIDER_TIME: float | None = 2.00             # 5 min to fetch/put on a helmet x EUR 0.40/min
TP_DELIVERY_DELAY: float | None = 1.00         # 20% chance of a late delivery x EUR 5

# True negative: rider with a helmet correctly let through.
TN_COST: float | None = 0.00                   # no intervention, no delay

COST_FALSE_NEGATIVE = _total(FN_SAFETY_FINE, FN_EXPECTED_ACCIDENT_COST)    # 5.00
COST_FALSE_POSITIVE = _total(FP_RIDER_TIME, FP_DELIVERY_DELAY)             # 0.65
COST_TRUE_POSITIVE = _total(TP_RIDER_TIME, TP_DELIVERY_DELAY)              # 3.00
COST_TRUE_NEGATIVE = _total(TN_COST)                                       # 0.00

# Human review
SECONDS_PER_REVIEW: float | None = 10.0         # glance at one flagged image
REVIEWER_COST_PER_HOUR: float | None = 20.0     # loaded hourly cost (salary + overhead)
ITEMS_PER_DAY: int | None = 1500                # riders through the checkpoint per day
REVIEW_HOURS_PER_PERSON_PER_DAY: float | None = 6.0  # productive hours after breaks

REVIEW_BAND: tuple[float, float] | None = (0.27, 0.79)  # chosen after inspecting the sweep
