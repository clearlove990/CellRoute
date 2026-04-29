# CellRoute

`CellRoute` is a manuscript-facing release for a sample-aware evaluation framework for differential spatial motif analysis. The package is intentionally positioned as an inference-validity / protocol-style contribution rather than as a new spatial representation-learning method or a disease-mechanism discovery report.

The frozen submission scope is:

- breast cancer Visium as the instability and pseudoreplication case
- Alzheimer disease MTG as the preservation-under-control case
- calibration simulation for sample-count and multiplicity limits
- SCZ downsampling plus 63-sample high-N inference validation
- 08C aging-brain stress test as supplement only

## Release Surface

The public package centers on four reproducibility pillars:

- sample-aware motif fitting and differential testing
- leave-one-sample-out (LOSO) motif alignment and stability analysis
- sample-level null control and permutation-based calibration
- manuscript figure, table, and submission-material regeneration

Included:

- `src/spatial_context/`: reusable spatial motif, LOSO, and sample-level testing code
- `scripts/`: manuscript-facing analysis runners and calibration entry points
- `experiments/08B_statistical_validity_paper_pack/`: figures, tables, manuscript text, and reviewer-facing materials
- `data_catalog/`: dataset manifests and download notes
- `legacy/`: archived HVG-era material retained outside the active release surface

Excluded from the main package surface:

- raw public datasets
- local caches, environments, and temporary files
- nested git metadata and duplicate staging trees

## Submission Materials

The release ships the manuscript materials under `experiments/08B_statistical_validity_paper_pack/text/`, including:

- `submission_manuscript.md`
- `experiment_scope_freeze.md`
- `final_claim_audit.md`
- `supplement_map.md`
- `data_code_availability.md`
- `cover_letter_draft.md`
- `reviewer_response_scaffold.md`

## Install

### Conda

```powershell
conda env create -f environment.yml
conda activate cellroute-sample-aware
pip install -e .
```

### Pip

```powershell
python -m pip install -r requirements.txt
python -m pip install -e .
```

## Key Entry Points

Sample-aware differential validation:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_sample_aware_cross_sample_differential.py
```

Sample-aware baseline regeneration:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_sample_aware_tissue_motif_baseline.py
```

Sample-level permutation calibration:

```powershell
$env:PYTHONPATH = "src"
python scripts/run_sample_permutation_simulation.py
```

Main-figure regeneration:

```powershell
Set-Location experiments\08B_statistical_validity_paper_pack\figures\scripts
python render_main_figures.py
```

## GPU Behavior

PyTorch-backed components check `torch.cuda.is_available()` at runtime and use `cuda` automatically when available. CUDA-enabled PCA and sparse aggregation are used where implemented, and all workflows fall back to CPU without code changes.

## Layout

- `src/spatial_context/neighborhood.py`: spatial `.h5ad` loading, neighborhood construction, and runtime/device reporting
- `src/spatial_context/motif_embedding.py`: sample-aware motif fitting, LOSO assignment, and centroid alignment helpers
- `src/spatial_context/cross_sample_differential.py`: sample-level motif summary construction and full-vs-LOSO consistency statistics
- `src/spatial_context/differential_motif.py`: mixed-effects testing and evidence-tier assignment
- `src/spatial_context/sample_level_testing.py`: sample-level permutation and null-control utilities
- `src/spatial_context/simulation.py`: calibration simulation scenarios and aggregation helpers
- `scripts/run_sample_aware_cross_sample_differential.py`: primary sample-aware differential workflow
- `scripts/run_sample_aware_tissue_motif_baseline.py`: sample-aware motif baseline regeneration
- `scripts/run_sample_permutation_simulation.py`: calibration simulation runner

## Data Availability

Raw spatial inputs remain hosted by their original public repositories. This release is for code plus processed reproducibility materials, not for re-hosting raw source data.
