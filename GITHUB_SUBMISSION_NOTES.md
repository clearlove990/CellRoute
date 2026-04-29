# GitHub Submission Notes

This folder is the clean public-facing GitHub and release package for `CellRoute`.

Canonical framing:

- Repository name: `CellRoute`
- Short description: `Sample-aware evaluation framework for differential spatial motif analysis with LOSO alignment, sample-level testing, and simulation calibration.`
- Repository description: `Release package for sample-aware differential spatial motif analysis, centered on donor-aware evaluation, leave-one-sample-out stability, permutation calibration, and manuscript-ready reproducibility assets.`

Frozen manuscript scope:

- breast instability and pseudoreplication panel
- AD preservation panel
- calibration simulation
- SCZ downsampling plus 63-sample inference validation
- 08C supplementary stress test only

Included in the public package:

- `README.md`
- `REPOSITORY_DESCRIPTION.txt`
- `requirements.txt`
- `environment.yml`
- `pyproject.toml`
- `src/spatial_context/`
- `scripts/`
- `experiments/08B_statistical_validity_paper_pack/`
- `data_catalog/`
- `legacy/`

Release cleanups applied:

- public framing updated from HVG benchmarking to spatial motif inference validity
- manuscript-facing freeze, claim-audit, cover-letter, and reviewer-response documents added
- nested duplicate staging repository removed from the distributable tree
- dependency declarations aligned to the actual public imports such as `h5py`, `statsmodels`, and `seaborn`

Claim guardrails:

- Do not present the release as a new spatial representation-learning method.
- Do not rewrite AD or SCZ panels as disease-mechanism discovery.
- Do not rewrite real-data support tiers as BH-controlled discoveries.
- Keep HVG content in `legacy/` only.
