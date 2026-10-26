# Parcel inspection scenario

Jimena selected damaged parcels on 25 October 2026 and requested a feedback loop. The decision is whether a warehouse inspector should pass an image, hold the parcel for inspection, or review an uncertain prediction. The positive class is visible box damage. An opened box is a separate detection class; the model does not infer its contents or product safety.

The report evaluates one inspection image as one decision. Images containing several boxes remain in the supplied dataset and are counted as positive if any labelled box is damaged. This scene-level evaluation must not be described as validated per-parcel production performance.

## Cost inputs and what supports them

- **Labour benchmark:** INE, Spain, Q2 2026, reports total labour cost of **€26.18 per effective hour** across surveyed activities. This is a national benchmark, not a measured warehouse wage. At an assumed 30 seconds per image, review labour is €26.18 × 30 / 3600 = **€0.2182** per image. Source: https://www.ine.es/dyngs/Prensa/ETCL2T26.htm (published 17 September 2026, accessed 25 October 2026).
- **Missed visible damage: €40.** Illustrative decomposition: €30 expected avoidable merchandise loss plus €5 additional logistics and €5 handling/customer service. These amounts are assumptions, not estimates extracted from a carrier study or guaranteed recoveries. Sensitivity analysis varies the total.
- **Unnecessary hold: €2.** Assumed inspection/handling/delay cost. A correct damage hold also costs €2 for preventive handling; passing an intact image costs €0. These costs do not include model development or fixed hardware.
- **10% positive rate:** scenario, not a warehouse observation. A DHL Philippines packaging article quotes 10% damaged packaging during shipping, but provides no method sufficient to treat it as this warehouse's prevalence. Use it only as context; analyse 1%, 5%, 10% and 20%. Source: https://www.dhl.com/discover/en-ph/logistics-advice/essential-guides/five-mistakes-to-avoid-during-packing (19 March 2023).
- **Volume and capacity:** 2,000 inspection images/day, 30 seconds/review, six productive hours/person/day, and two review hours/day. All are planning assumptions. Two review hours accommodate 240 images/day, or 12% of this volume.
- **Human accuracy:** 99% sensitivity and 99% specificity are unvalidated planning inputs. The dashboard exposes them; a real time-and-motion study and independently adjudicated labels are required before operational decisions.

UPS asks claimants to retain the item and original packaging and provide value evidence. This supports treating exterior damage as a reason for inspection, not automatic proof of destroyed contents or a fixed financial loss. Source: https://es-us.ups.com/us/en/support/file-a-claim (accessed 25 October 2026).

## Feedback loop boundary

The assignment requires the public dataset to be used as supplied, so new user uploads and review labels are stored separately for audit and diagnosis. They are not silently added to the coursework training data. The feedback agent can identify error patterns, propose changes and evaluate a candidate. It must not overwrite verified labels, claim its own guesses as truth, or promote a new model without a recorded approval and a fresh independent evaluation plan.

These notes provide sourced inputs and explicit assumptions for Jimena's own report decisions; they are not a claim that a model has already trained or that the system is ready to ship.
