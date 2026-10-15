# Technical provenance

This repository is a new curated code-only derivative of the corrected local
joint study and empirical audit. The manuscript and all original working
directories are preserved separately. No study variant, seed, preservation
margin, attack selection or inference target was changed for packaging.

The Python files in `source/strength_privacy/` and `source/joint_strength/` are
byte-identical to the corrected frozen study kernels. The four source fixtures
are also byte-identical. `source/docs/INPUT_PROVENANCE.json` is newly generated
for the selected relative filenames and hashes, without workstation paths.
`source/docs/KERNEL_PROVENANCE.json` describes historical port lineage, not a
claim that every historical prototype is shipped or identical to current code.

Changes to the auxiliary scripts are restricted to source discovery, explicit
input/output configuration, parameter-loading validation and avoiding writes
into the repository. The original reference script is now guarded by `main()`.
The empirical test suite uses explicitly invented parameters. Existing numerical
function ASTs outside `main()` were checked against their local origins; the
reference audit's network guard moved into `main()` without changing its body.
`main()` plumbing was inspected separately. AST identity does not by itself
validate the scientific implementation, so tests and arithmetic checks remain
necessary.

All seven base dependency hashes and the pandas hash matched official PyPI
artefacts. Six optional Excel-support wheels in the earlier local environment
were local builds. Their hashes were replaced with official PyPI wheel hashes
of the **same versions**. The optional environment was freshly installed and
tested against the corrected lock. No wheel files are distributed.

`FILES.json` inventories selected repository files by SHA-256, excluding itself
to avoid a circular hash. `scripts/verify_repository.py` compares that allowlist
against the exact tracked Git files. Git history and the remote tree are checked
separately before handoff. Generated outputs, raw data and internal audit logs
remain outside the repository. Neither a hash nor private visibility proves
scientific correctness, ownership or anonymity.

AI assistance covered coding, technical checks and prose. Final human review and
paper authorship remain open. No final human approval is attributed to any person.
