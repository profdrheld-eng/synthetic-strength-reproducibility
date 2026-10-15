# Synthetic strength: privacy and scientific preservation

Minimal code companion for a methodological study combining artificial experiments
and a local audit of strength z-scores and a principal-component strength index.
This is a private working repository, not a data release or a submitted paper.
Manuscript authorship and final human approval remain open.

The question is whether two fixed CART synthesis configurations preserve the
specified scientific results while limiting attack success. Preservation and
privacy are separate endpoints. There is no numerical privacy acceptance rule,
joint safe-and-useful pass rate, or proof of anonymity.

## Contents

* `source/`: frozen artificial generators, corrected published-reference
  implementation, depth-8 CART configurations, attacks, PC1, controls and utility.
* `diagnostics/`: the existing, explicitly exploratory tail and nonlinear-relation
  replay. These diagnostics are outside the unchanged preservation filter.
* `empirical/`, `empirical_reference/`: local-only adapters and original-reference
  checks. Protected inputs and historical regression parameters are not included.
* `reference/artificial_reported_values.json`: 145 fixed artificial manuscript
  entries with logical result paths; no empirical results are distributed.
* `tests/` and the tests alongside each component: artificial checks only.

No original or empirically generated synthetic person rows, empirically fitted models,
participant identifiers, historical regression parameter exports, manuscript,
figures, reviewer reports, environments or dependency wheels are included.
Three aggregate PC1 weights and its variance fraction remain in the empirical
loader as historical validation targets, not participant records.

## Install

Use Python 3.12 and a virtual environment **outside** the repository:

```sh
python3.12 -m venv ../strength-env
../strength-env/bin/python -m pip install --require-hashes -r requirements-macos-arm64.lock
../strength-env/bin/python -m pip check
```

The hash locks target macOS arm64. The test suite also needs pandas, so install
the complete optional environment before running **all** tests:

```sh
../strength-env/bin/python -m pip install --require-hashes -r requirements-empirical-macos-arm64.lock
../strength-env/bin/python run_tests.py
```

`requirements.txt` and `requirements-empirical.txt` pin the same versions without
artifact hashes for other platforms, which have not been validated. Do not use
Python `-O`, which disables scientific validation assertions. No CI service,
container, hosted environment or dependency binary is bundled.

## Artificial reproduction

```sh
../strength-env/bin/python run_artificial.py --output ../strength-artificial-smoke --smoke
../strength-env/bin/python run_artificial.py --output ../strength-artificial-full
```

Output directories must be new and outside the repository. The smoke run checks
plumbing only. The full command runs all fixed artificial cohorts, independently
checks aggregate arithmetic, replays the exploratory diagnostics and compares all
145 artificial reported entries at the pre-existing absolute tolerance `1e-10`.
It can take about half an hour and produces substantial local intermediate
aggregates. Nothing is uploaded by these commands.

The artificial study retains three population models, ten independent reference
cohorts per model, 197 women and 196 men, two fixed CART variants and 100 releases
per cohort. It evaluates 6,000 CART releases, 6,000 natural comparisons, 660
control-configuration rows and 19,980 paired privacy conditions. The 13 limits,
29 checks, seeds, attack collections and uncertainty calculations are unchanged.

## Empirical reproduction

Read [the protected-input contract](docs/EMPIRICAL.md) first. It is deliberately
separate and requires authorized local inputs; the artificial default never
opens these inputs. Empirical synthetic releases are **not established anonymous**.
The empirical commands write local aggregates, including potentially sensitive
provenance; do not upload their outputs without a separate review.

## Provenance, licences and limits

See [provenance](docs/PROVENANCE.md), [licence scope](LICENSE_SCOPE.md),
[scientific interpretation](docs/METHOD_MAP.md), and
[verification status](docs/VERIFICATION.md). MIT applies only to our own code
contributions, CC BY 4.0 only to our own documentation and artificial aggregates.
Published formula constants, third-party software and cited publications are
excluded from those ownership claims. Dependency installation uses upstream
licences; this repository contains no copies of their source distributions.

AI tools assisted coding, checking and wording. Automated checks do not replace
author responsibility, independent scientific validation or final human review.
No paper author list or completed human approval is asserted here.

The corrected code at commit `7a8615a2bac34a9a6066be0b56089e4249e685d5` has
completed an authorized local empirical reproduction from a fresh GitHub clone
and new environment. All 166 registered empirical entries and 545 numerical
table cells were confirmed within the fixed tolerance. See the bounded
[verification summary](docs/VERIFICATION.md); protected inputs, empirical outputs
and detailed audit records remain local. Later documentation-only changes do
not change the tested analysis code. The repository remains private.
