# Standalone calculation bundle

An independent re-derivation of the costing from the 24-row threshold count
table, using only the Python standard library. It does not run YOLO or touch the
pipeline; it exists to check `costing.py` (which sweeps 104 thresholds over every
object's score) by a second route. Both give t = 0.03, EUR 4.64 per 1,000 and an
empty review band. Because this table has 24 thresholds, break-evens that land on a
threshold missing from it (e.g. 0.12) show the nearest listed one instead.

See `REPORT.md` at the repo root for the scenario, sourced cost anchors and
limitations.

## Recompute

With Python 3, run `python3 recompute.py` from this directory. The program uses only
the standard library and reads `threshold_counts.csv` next to itself. It has no
network or model dependencies. Edit the `BASE` dictionary in the program to change
inputs; `assumptions.json` is an output record, not a configuration file.

The program checks the provided placeholder example and nonincreasing counts,
then recreates the result files. It evaluates exactly 24 automatic thresholds and
300 pairs with L <= H. An empty band is permitted and is the base-case winner.

## Files and units

- `threshold_counts.csv`: all 24 supplied count rows, transcribed unchanged.
- `recompute.py`: standalone formulas, complete grid search and sensitivity logic.
- `assumptions.json`: selected inputs; money is EUR; prevalence/error rates are fractions.
- `model_sweep.csv`: every automatic threshold and component costs.
- `band_sweep.csv`: every band, including infeasible bands flagged `feasible=false`.
- `sensitivity.csv`: exact affine cost-crossing and capacity-crossing calculations
  for the connected stability interval around the base optimum. `below` and `above`
  are the immediately adjacent optimum, not the optimum everywhere beyond the
  first crossing. Equal endpoints such as `[0.03,0.03)` mean automatic sorting.
- `stress_cases.csv`: selected one-at-a-time sweeps; seconds and utilisation also
  update allocated review cost. Other inputs are varied independently.
- `results.json`: best policies, labour alternatives, components, placeholder checks
  and nominal Wilson intervals.

`cost` and every component ending in `_cost` are EUR per 1,000 incoming objects.
`cost_day` is EUR per assumed operating day. `review_share` is review events per
incoming real object, including stray events under the supplied convention.
`people` is concurrent fractional staffing demand, not a promise that fractional
employees can be hired. Denominators are 31, 93 and 2,922: the 3,049 labelled
objects become 3,046 units because the 3 cut pieces are also boxed as potatoes and
are counted once.

The primary cost per review includes paid nonproductive time: hourly cost times
seconds divided by 3,600 and utilisation. The active-time comparison omits the
utilisation division. The dedicated-worker comparison removes per-review labour
and adds one full shift instead, avoiding double counting.

Positive correct-reject and correct-pass costs are optional sensitivity extensions,
zero in the base case. Correct good passes exclude stray events. No capital,
severe-incident or downstream handling costs are imputed.

Reported external values are dated anchors, not live inputs. No source refresh,
model retraining, unit-level threshold sweep or independent performance validation
is performed by this program.
