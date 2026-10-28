# Research notes and starting assumptions

Retrieved 27 October 2026. These are working references, not the student's final
analysis or evidence of a production deployment.

## Published benchmarks

- [Europcar Spain damage policy](https://www.europcar.es/es-es/p/informacion-legal/politica-de-gestion-de-danos)
  links to its damage-assessment matrix. The matrix linked on the retrieval date
  is [valid from 10 December 2025](https://www.europcar.es/es-es/p/assets/1bB3O5WAkmeBkwmD2bNDRs).
  Examples for a front-right door, page 24, vary by vehicle group: a 2–5 cm
  scratch is EUR 146.76–159.03; a 2–5 cm dent is EUR 257.36–269.63. These are
  example tariff entries, not national average repair prices or a measured
  rental company's incremental loss from a model error. There are many other
  entries, sizes and vehicle groups; these two examples do not describe the
  whole damage-price distribution.
- [Eurostat annual labour-cost dataset](https://ec.europa.eu/eurostat/databrowser/view/tps00173/default/table?lang=en),
  source series `lc_lci_lev`, gives Spain's 2025 hourly total employer labour
  cost as EUR 26.4. Verified through its
  [official API](https://ec.europa.eu/eurostat/api/dissemination/statistics/1.0/data/lc_lci_lev?lang=en&geo=ES&time=2025&unit=EUR&nace_r2=B-S_X_O&lcstruct=D1_D4_MD5).
  The sector scope is industry, construction and services excluding public
  administration. This is a broad employer-cost benchmark, not the wage or
  actual hourly cost of a rental-return inspector.

## Scenario accepted by Javier

Javier chose rental-car inspections, said missing damage costs more, and
accepted these starting assumptions in chat on 27 October 2026:

| Input | Starting value | Status |
|---|---:|---|
| Unnecessary inspection after a false alarm | EUR 5 | Business assumption |
| Additional expected loss when damage is missed | EUR 200 | Business assumption informed by tariff scale |
| Vehicle returns per day | 100 | Hypothetical branch volume |
| Human photo-review time | 5 minutes | Workflow assumption to time in a pilot |
| Cases with relevant visible damage | 10% | Production assumption, not dataset prevalence |

The error-cost ratio is 200 / 5 = 40. A simple optional costing convention is
EUR 30 per staff-hour, rounded above the broad EUR 26.4 benchmark, making a
5-minute photo review cost EUR 2.50 and a 10-minute closer inspection EUR 5.
That convention is a proposed assumption, not an observed market service price.

Before costing the trained system, define whether a case is one photo or one
complete vehicle return. A public single-image dataset cannot automatically
validate a multi-photo return workflow. Count people by review minutes divided
by available productive minutes; do not silently treat photo counts as returns.

The EUR 200 missed-damage cost means *additional expected unrecovered loss*, not
that a repair costs nothing if the model detects it. It is not measured from
Europcar. Treat repair expenditure common to both actions consistently, price
true-positive inspection actions too, and vary the assumed missed-loss amount,
damage prevalence, labour rate and reviewer accuracy in sensitivity analysis.

## Dataset choice and audit

The business framing was recorded in REPORT.md before searching datasets.
The selected original public dataset is
[celebal-wlpui / car-damage-zxcdc / 1](https://universe.roboflow.com/celebal-wlpui/car-damage-zxcdc/dataset/1).
Its classes are Damage and Undamage. The original 250 / 84 / 84 split and all
labels were retained, and no dataset fork was created. The API listed the
licence as Public Domain. Exported labels contain 232 Damage boxes, 173
Undamage boxes and 13 images without boxes. Computed counts, box sizes and
exact-duplicate checks are in artifacts/dataset_audit.json. These were
computed before the first training run. The visual audit showed major crash
damage, stock-style clean-car photos and some questionable Damage labels.

Carpro's car-scratch-and-dent dataset was considered but not selected: the
metadata showed only 20 validation and 20 test images, six dirt instances,
zero Car instances, and no established clean-image negative population.

Other candidates with no published clean-image information have not been
accepted merely because their names match the topic. Estimating false alarms
requires relevant negative examples in validation and test data.

## Account setup

The assignment requires a working Roboflow key before training code is written.
The official [key instructions](https://docs.roboflow.com/developer/authentication/find-your-roboflow-api-key)
link to [API settings](https://app.roboflow.com/settings/api). Use the private
key in the ignored local `.env` file. Never include it in Git, report, screenshots,
logs or the Blackboard backup ZIP. Javier authorized retrieval of the existing
workspace key through the Roboflow page. It was saved locally and accepted by
the official API; its value is not recorded here.

## Additional modelling assumptions

The initial cost model uses EUR 30 per staff-hour, 95% reviewer sensitivity,
95% reviewer specificity, 360 productive staff minutes per day and 10 minutes
for a closer physical inspection. These are illustrative assumptions, not
measured human performance or contractual staffing estimates. A correct model
referral also costs EUR 5; detecting damage does not make the inspection free.
Human-review cost includes the review itself plus subsequent mistakes or
inspection referrals. The workload calculation includes both photo-review
time and closer-inspection time, and prices common repair expenditure outside
the comparison rather than pretending that it is avoidable.

The first real experiment used 320-pixel images. Its test results were
inspected. Javier then asked to try changes and use evidence to improve the
model; a second experiment changes only the input size to 480 pixels. Model
and threshold selection use validation costs. The report must disclose that
the test split was inspected during development; it is not a completely blind
final estimate of performance.
