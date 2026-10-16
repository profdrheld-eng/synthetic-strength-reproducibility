# Method and result map

| Scientific component | Implementation | Output / scope |
|---|---|---|
| Artificial cohorts and fixed reference | `source/strength_privacy/population.py`, `norms.py` | Artificial only; corrected female reduced-model residual scales |
| Two fixed CART configurations | `source/strength_privacy/synthesis.py` | Depth 8, minimum leaf 5; strength-first and interleaved orders |
| z-score and PC1 preservation | `source/joint_strength/utility.py`, `pc1.py` | Unchanged 13 technical limits and 29 checks, not validated acceptance standards |
| Privacy attacks and knowledge errors | `source/strength_privacy/attacks.py`, `adaptive.py`, `source/joint_strength/runner.py` | Original and transferred adapted collections; singling out, linkability, inference |
| Full artificial experiments / controls | `source/joint_strength/study.py` | `study/results.json`, independent arithmetic check |
| Exploratory tail and relationship diagnosis | `diagnostics/diagnose.py`, `report.py` | `diagnostics/results.json`; selection after joint results, pointwise intervals |
| Fixed artificial manuscript targets | `reference/artificial_reported_values.json`, `scripts/check_reported_values.py` | 145 entries; no empirical target register in this repository |
| Original-data reference reconstruction | `empirical_reference/reproduce.py`, `reference_utils.py` | Local `AGGREGATE_RESULTS.json`, input manifest; no synthetic rows |
| Empirical CART, attacks and controls | `empirical/empirical.py`, `run.py` | Local summary, 1,000 joint releases, 200 full-cohort utility releases, 110 controls |
| Empirical arithmetic / tie checks | `empirical/verify.py`, `verify_additional.py`, `verify_copy_ties.py` | Aggregate checks; tie diagnostic explicitly opens authorized original data |

Artificial uncertainty is evaluated across ten independent cohorts **within**
each population model. Shared seeds pair models; do not pool these as 30
independent replications. Repeated releases are not independent cohorts.
The empirical analysis uses five overlapping training folds and conditional,
sex-standardized person-cluster bootstrap intervals with 2,000 draws. It is not
an independent-cohort validation of a population risk bound.

Low combined preservation pass rates must be interpreted against natural
controls, which also rarely pass the strict complete filter. Passing an
individual PC1 check is not passing all z-score checks. A positive member-minus-
control attack difference is descriptive evidence under the specified attack,
not a fully calibrated causal leakage estimate. A nonsignificant control test
does not establish equivalence. There is no anonymity certification or release
permission implicit in these results.

The artificial reference uses transcribed published rounded reduced equations
and corrected associated residual scales. Equation rounding and the unverified
original residual-scale denominator remain limitations.
The empirical reference uses historical unrounded coefficients and sample
residual SD (`n-1`), supplied separately. These references are deliberately not
silently substituted for each other.

The inherited artificial status token `PROVISIONAL_ROUNDED_UNVERIFIED_RSD` is
retained for frozen-code identity. The selected fixtures contain the corrected
published reduced-model residual scales; the provisional scientific limitations
concern equation rounding and the unresolved original scale-denominator definition.
Do not interpret that historical
token as a new verification result or as a switch to empirical parameters.

Published norm source: Isenmann E, Geisler S, Havers T, Gavanda S, Diel P, Held S,
Flenker U. *Normalization of muscle strength measures in barbell exercises by
sex, age, and anthropometry: notes on allometry.* Scientific Reports.
2026;16:20797. https://doi.org/10.1038/s41598-026-60646-z.
Citation identifies the reference; it does not grant this project rights in it.
