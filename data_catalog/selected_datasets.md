# Selected Spatial Datasets

## Active Picks

### Primary: Wu 2021 Breast Cancer Visium
- Source: [Zenodo 4739739](https://zenodo.org/records/4739739)
- Why now: Tumor-stroma-immune story is strong; paired metadata and pathology labels help interpretation; six sections are manageable.
- Main trade-off: No normal control arm; subtype heterogeneity is useful but adds confounding.
- Planned role: `primary`

### Validation: GSE220442 AD MTG
- Source: [GEO GSE220442](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220442)
- Why now: Direct disease/control contrast; six Visium sections; compact counts+images archive is easy to reproduce.
- Main trade-off: Cell type labels may need deconvolution or clustering if only spot-level metadata are present.
- Planned role: `validation`

## Reserve Candidate

### GSE212903 Aging Mouse Brain
- Source: [GEO GSE212903](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE212903)
- Why reserve: Clean spatial layers and condition labels; useful reserve validation for region-aware motifs.
- Why not active this week: Older/aging contrast is weaker than disease-control; raw archive is larger than the two primary picks.
