# Verification status, 15 October 2026

This is a technical packaging check, not final human approval of the paper or a
privacy certification. All checks in this round use artificial inputs only.

| Check | Status | Evidence and scope |
|---|---|---|
| Frozen scientific kernels and four fixtures | PASS | Byte identity against the corrected local study; seeds, margins and configurations unchanged |
| Numerical function preservation in auxiliary scripts | PASS | 107 pre-existing non-main function ASTs checked; main/path/import plumbing reviewed separately |
| Official dependency availability | PASS | All 14 selected wheel hashes verified against primary PyPI metadata; six optional local-build hashes corrected without version changes |
| Fresh optional environment | PASS | Hash-checked offline installation from verified official wheels; `pip check` passed |
| Artificial tests | PASS | 45 unit/integration tests, including ten fold/variant mirror cases with invented parameters and no existing Excel input |
| Coupled-bootstrap arithmetic | PASS | Independent explicit scalar-sum calculation on artificial arrays, tolerance `1e-12` |
| Fixed result checker | PASS | All 145 entries checked against pre-existing artificial aggregates, maximum difference zero; this alone is not a fresh reproduction |
| Official-environment smoke run | PASS | Artificial CLI plumbing only, not the full paper calculation |
| Full artificial reproduction | PASS | All 30 cohorts, 6,000 CART releases and 6,000 natural comparisons completed; all 145 reported artificial values reproduced with zero difference |
| Empirical data reproduction in this round | NOT RUN | No person data or historical parameter exports read; prior local empirical evidence remains separate |
| Public empirical reference targets | NOT INCLUDED | The 166-entry empirical register and empirical aggregates are outside this minimal package |
| Final human paper review / authorship | OPEN | No completed approval or author list asserted |
| GitHub visibility | PRIVATE | Own account `profdrheld-eng`, repository `synthetic-strength-reproducibility`; no public release |

The complete artificial calculation uses the same seven hash-locked numerical
dependencies as the fixed original experiment. The separate fresh optional
environment validates the publicly available Excel-support artefacts and the
artificial empirical-adapter tests; it does not establish a new original-data
reproduction. No other platform has been tested.

Independent aggregate verification checked 12,660 utility rows, 367,140 utility
flags, 19,980 privacy conditions, 82,200 means, 6,978 interval nodes and 252
control tests. Its maximum numerical difference was `1.1368683772161603e-13`,
below the fixed `1e-10` tolerance. The exploratory replay verified 6,000 CART
fingerprints; its separate report checker verified 216,000 tail rows and 348,000
utility flags with maximum aggregate error zero. All 145 fixed artificial
manuscript entries matched exactly. This is not an independent reimplementation
of the attacks and does not establish empirical-data reproducibility.

Local structural-preflight findings were reviewed: its unsupported-extension
warnings concern `.gitignore`, `.gitattributes` and the two text lockfiles, which
were inspected separately. The helper does not inspect `.git`; tracked files and
the reachable Git tree are checked separately. An initial literal-pattern false
positive was resolved without exempting the scanner file. A regression test
first demonstrated that exemption and then passed after its removal.
The helper itself does not assess scientific
validity, secrets or rights; exact tracked-file review and selected secret/path
checks are separate. Those checks cannot prove absence of every possible flaw.

The artificial original-reference comparison and full manuscript's empirical
claims remain scientifically distinct. No test here establishes anonymity,
validates the numerical preservation limits, or completes ethics, authorship,
data-release or final human approval decisions.
