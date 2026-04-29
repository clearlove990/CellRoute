# Candidate Spatial Datasets

Screening target: prioritize multi-sample spatial cohorts with condition labels, coordinates, interpretable biology, and download paths that are reproducible inside this repository.

## Shortlist Table

| Dataset | Direction | Samples | Spots/Cells | Labels | Histology | Cell type | Selected |
| --- | --- | ---: | ---: | --- | --- | --- | --- |
| Wu 2021 Breast Cancer Visium | Tumor microenvironment | 6 | 24584 | Clinical subtype: HER2+, TNBC, ER+/PR+ | yes | yes | yes |
| GSE220442 AD MTG | Brain disease/control | 6 | 22499 | Alzheimer disease vs control | yes | yes | yes |
| GSE212903 Aging Mouse Brain | Reserve | 6 | 17368 | 2 months, 18 months, 24 months | yes | yes | no |
| GSE189184 Colitis Panel | Reserve | 16 | 50857 | Ulcerative colitis, healthy, ICI colitis | yes | yes | no |
| GSE152506 AD Mouse Plaque Niches | Reserve | 20 | 10079 | AppNL-G-F KI vs WT; 3, 6, 12, 18 months | partial | partial | no |

## Notes By Dataset

### Wu 2021 Breast Cancer Visium
- Source: [Zenodo 4739739](https://zenodo.org/records/4739739)
- Tissue/technology: Breast tumor; 10x Visium
- Conditions: Clinical subtype: HER2+, TNBC, ER+/PR+
- Estimated size: ~0.90 GB
- Pros: Tumor-stroma-immune story is strong; paired metadata and pathology labels help interpretation; six sections are manageable.
- Cons: No normal control arm; subtype heterogeneity is useful but adds confounding.
- Download artifacts: filtered_count_matrices.tar.gz, spatial.tar.gz, metadata.tar.gz

### GSE220442 AD MTG
- Source: [GEO GSE220442](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220442)
- Tissue/technology: Middle temporal gyrus; 10x Visium
- Conditions: Alzheimer disease vs control
- Estimated size: ~0.05 GB for counts+images package
- Pros: Direct disease/control contrast; six Visium sections; compact counts+images archive is easy to reproduce.
- Cons: Cell type labels may need deconvolution or clustering if only spot-level metadata are present.
- Download artifacts: GSE220442_counts_and_images.tar.gz, GSE220442_metadata.csv.gz

### GSE212903 Aging Mouse Brain
- Source: [GEO GSE212903](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE212903)
- Tissue/technology: Brain; 10x Visium
- Conditions: 2 months, 18 months, 24 months
- Estimated size: ~1.42 GB
- Pros: Clean spatial layers and condition labels; useful reserve validation for region-aware motifs.
- Cons: Older/aging contrast is weaker than disease-control; raw archive is larger than the two primary picks.
- Download artifacts: GSE212903_RAW.tar, GSE212903_spatial.tar.gz

### GSE189184 Colitis Panel
- Source: [GEO GSE189184](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189184)
- Tissue/technology: Colon; 10x Visium
- Conditions: Ulcerative colitis, healthy, ICI colitis
- Estimated size: ~8.73 GB
- Pros: Excellent inflammation story with healthy and disease states; immune and stromal neighborhoods are directly relevant.
- Cons: Archive is large for a first-week screen; preprocessing cost is meaningfully higher than the selected pair.
- Download artifacts: GSE189184_RAW.tar

### GSE152506 AD Mouse Plaque Niches
- Source: [GEO GSE152506](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152506)
- Tissue/technology: Brain; Spatial Transcriptomics (legacy ST)
- Conditions: AppNL-G-F KI vs WT; 3, 6, 12, 18 months
- Estimated size: ~0.26 GB
- Pros: Large number of sections and explicit genotype/time conditions make it a strong disease-model candidate.
- Cons: Legacy ST packaging is less turnkey than Visium; coordinate and image handling likely needs custom work.
- Download artifacts: GSE152506_raw_counts.txt.gz
