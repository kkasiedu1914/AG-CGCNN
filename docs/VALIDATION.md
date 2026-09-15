# Release validation

Validated locally on **Windows, Python 3.12.14, CPU**, on 15 September 2026.

## Results

| Check | Result |
|---|---|
| Functional and safety test suite | **21 passed** |
| Two curated notebooks | **All 10 code cells executed in order** |
| Regression | One-target and two-target training, prediction and SHAP passed |
| Classification | Training, probability output and SHAP passed |
| Prediction-head equivalence | Passed for 0/5 geometric descriptors and 1/3 hidden dense layers |
| Data checks | Invalid identifiers, CSV formula identifiers, non-finite values, wrong columns, duplicates and fractional class labels rejected |
| Input safety | Directory traversal, oversized CIF ZIP members, unsupported pickle checkpoints and tampered checkpoint dimensions/scalers rejected |
| Reproducibility | Seeded disjoint splits and training-only normalization checked |
| Notebook hygiene | Valid notebook format; saved outputs and execution counts cleared |
| Distribution | Python wheel built successfully |
| Dependency advisory check | **96 installed package versions checked against PyPI advisory metadata; no advisories returned** |

The notebook check executed each Python code cell in order in a local Python process, including the IPython image-display cell. It was not a browser-driven Jupyter UI test. The test suite emitted three SHAP/Matplotlib pending-deprecation warnings; they did not affect results.

The environment's Windows temporary-directory ACL behavior prevented ordinary pip setup, so dependencies were staged into an isolated workspace directory from official PyPI wheels with SHA-256 verification. The PyPI advisory check queried each exact installed version; it is a point-in-time advisory check, not a guarantee of complete vulnerability coverage. The included GitHub workflow also runs the test suite and `pip-audit` in ordinary Linux environments when enabled by the host repository.

## Tested package versions

The validation used PyTorch 2.14.0 (CPU), pymatgen 2026.5.4, pymatgen-core 2026.8.30, NumPy 2.5.3, SHAP 0.52.0, Matplotlib 3.11.2, pandas 3.0.5, scikit-learn 1.9.1 and pytest 9.1.1. The optional [validation environment snapshot](validation-environment.json) records all installed versions. These versions describe the tested environment; they are not a universal platform lockfile.

## What these checks establish

They establish that the released graph model, data pipeline, training, safe serialization, prediction and explanation workflow work together on small synthetic inputs, and that selected invalid/unsafe inputs fail as expected.

They do **not** establish numerical reproduction of the paper, full-dataset performance, GPU compatibility, the correctness of every historical experiment, or the absence of every possible vulnerability. Historical reference snapshots are not executable release code and were not functionally retested. Full research reproduction requires the original experiment data and settings.

## Run checks yourself

```bash
python -m pip install -e ".[test]" pip-audit
python -m pytest -q
python scripts/check_release.py
python -m pip_audit
```

The CI configuration checks Python 3.11 and 3.12 on Linux. The [initial hosted run](https://github.com/kkasiedu1914/AG-CGCNN/actions/runs/35004468958) passed all 21 tests and source-hygiene checks on both versions. Its dependency audit passed on Python 3.12 and flagged the runner's preinstalled setuptools 79.0.1 on Python 3.11 (PYSEC-2026-3447; reported fixed version 83.0.0). The build requirement and CI setup now require setuptools 83 or newer. Consult [GitHub Actions](https://github.com/kkasiedu1914/AG-CGCNN/actions) for the latest audit result. The unpublished local AG-CGCNN package itself is not in PyPI's advisory database; its code is covered by the source review and tests, not that database check.
