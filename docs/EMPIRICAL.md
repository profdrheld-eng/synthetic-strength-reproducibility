# Protected empirical inputs and local execution

These commands are optional and are never part of the artificial test default.
Do not run them without authorized access. No original data, historical
regression coefficients, residual parameter CSVs or original-paper source
archive are distributed here. Missing inputs are a real reproducibility limit,
not a successful artificial reconstruction of empirical results.

## Required inputs

`STRENGTH_ORIGINAL_PROJECT` points to the authorized original-paper directory,
containing:

* `df_updated.xlsx`: sheets `z_scores` and `TE` (TE header row index 4).
  z_scores columns: `id`, `code`, `sex`, `age`, `bodymass`, `bodyheight`,
  `sq1rm`, `bp1rm`, `dl1rm`, `Z_squat`, `Z_benchpress`, `Z_deadlift`.
  TE requires `Probandencode`. Sex labels are `weiblich` and `männlich`.
* `analysis_pipeline.py`, `tools/run_pc1_comparison.py`: trusted, audited original
  scientific source files. Selected function ASTs are executed locally. These
  must not be replaced with untrusted downloaded code.
* `results_pc1_final/00_pc1_diagnostics.csv`,
  `00_pc1_loadings_and_correlations.csv`, `01_sample_descriptives.csv`,
  `02_zscore_validation.csv`: saved original aggregate reference tables.

The published row flow is 393 source persons (197 female, 196 male), 384 exact
joins, 374 complete linked z-score profiles (186 female, 188 male). No fuzzy
join, imputation or alternate-case substitution is performed.

`STRENGTH_PARAMETERS` points to authorized historical aggregate exports:

* `coefficients.csv`: `variant,stratum,lift,term,coefficient`; reduced-model
  rows for female/male and SQ/BP/DL. Each combination has exactly the terms
  `intercept,height_cm,age_years,age_squared,log_mass_kg,log_mass_squared`.
* `residual_scales.csv`: `variant,stratum,lift,sd_residuals,sigma_residual`.
  The adapter uses reduced-model `sd_residuals`, the sample residual SD with
  denominator `n-1`. The reference audit also compares `sigma_residual`.

Invented parameters in `empirical/artificial_parameters.py` are solely test
fixtures. They must never replace the historical empirical reference.

## Commands

Use the complete optional environment from the README. Set paths to real,
authorized local inputs. The placeholders below are examples, not shipped data.
Output directories must be new and outside the repository.

```sh
export STRENGTH_ORIGINAL_PROJECT=/authorized/original-project
export STRENGTH_PARAMETERS=/authorized/historical-parameters
export STRENGTH_REFERENCE_OUTPUT=/local/new-reference-output
export STRENGTH_EMPIRICAL_OUTPUT=/local/new-cart-output
export STRENGTH_DATA="$STRENGTH_ORIGINAL_PROJECT/df_updated.xlsx"
../strength-env/bin/python empirical_reference/reproduce.py
../strength-env/bin/python empirical/run.py
../strength-env/bin/python empirical/verify.py
../strength-env/bin/python empirical/verify_additional.py
../strength-env/bin/python empirical/verify_copy_ties.py
```

The reference audit records input hashes; the CART run checks them before and
after execution. Do not generate or modify a reference manifest to make a
mismatched input pass. Assertion failures require investigation. Do not use
`python -O`. Local provenance outputs can contain file paths and input hashes.

The empirical runner blocks `socket.connect` events while data are in memory.
This is a limited in-process guard, not an operating-system security boundary.
Person contributions exist in memory for the coupled bootstrap. Only aggregates
are written by these scripts; aggregate output is not an automatic publication
or anonymity clearance. The tie check reads the original dataset again.

No empirical run is claimed from running the artificial tests. Earlier local
empirical reproduction evidence remains outside this minimal repository. A
future externally authorized empirical reproduction must separately compare the
full manuscript's 166 empirical target entries and tables; this repository does
not distribute that empirical reference register.
