from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for candidate in (str(SRC), str(SCRIPTS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hvg_research import build_default_method_registry

import build_topconf_selector_round2_package as pkg
import run_real_inputs_round1 as rr1


OUTPUT_DIR = ROOT / "artifacts_restart_broad_bfs_smoke"
SAFE_ANCHOR_METHOD = "adaptive_hybrid_hvg"
TOP5_METHODS = (
    "adaptive_eb_shrinkage_hvg",
    "adaptive_invariant_residual_hvg",
    "adaptive_spectral_locality_hvg",
    "adaptive_stability_jackknife_hvg",
    "adaptive_risk_parity_hvg",
)
SMOKE_DATASETS = (
    "GBM_sd",
    "cellxgene_human_kidney_nonpt",
    "paul15",
    "cellxgene_immune_five_donors",
    "mus_tissue",
    "homo_tissue",
)
POSITIVE_SMOKE_DATASETS = (
    "GBM_sd",
    "cellxgene_human_kidney_nonpt",
    "paul15",
    "cellxgene_immune_five_donors",
)
ATLAS_CONTROL_DATASETS = (
    "mus_tissue",
    "homo_tissue",
)
REFERENCE_METHODS = (
    SAFE_ANCHOR_METHOD,
    "adaptive_stat_hvg",
    "variance",
    "mv_residual",
    "fano",
    "multinomial_deviance_hvg",
    "analytic_pearson_residual_hvg",
    "triku_hvg",
    "scanpy_seurat_v3_hvg",
    "scanpy_cell_ranger_hvg",
    "scran_model_gene_var_hvg",
)
ANALYSIS_CONDITION_NAMES = (
    "targeted_shift_gap",
    "winner_overlap_pull_positive",
    "winner_overlap_pull_gap",
    "winner_corr_pull_gap",
    "control_guard",
)
BFS_WEIGHTS = {
    "benchmark_fit_score": 0.27,
    "mechanism_plausibility_score": 0.22,
    "novelty_score": 0.15,
    "smoke_testability": 0.14,
    "implementation_ease_score": 0.10,
    "noise_robustness_score": 0.07,
    "atlas_safety_score": 0.05,
}


LITERATURE_ENTRIES = [
    {
        "name": "Empirical Bayes variance shrinkage (limma)",
        "source_field": "robust statistics / empirical Bayes",
        "core_idea": "Borrow strength across features to shrink unstable gene-level variance estimates toward a global prior.",
        "relation": "Strongly transferable because the benchmark gap looks like selective rank instability, not missing biology constraints.",
        "reference": "[Smyth 2004](https://doi.org/10.2202/1544-6115.1027)",
    },
    {
        "name": "James-Stein style shrinkage",
        "source_field": "shrinkage estimation",
        "core_idea": "Shrink noisy per-feature estimates toward a pooled target when disagreement or variance is large.",
        "relation": "A natural template for gene-level score fusion that stays inside one deterministic scorer.",
        "reference": "[James and Stein 1961](https://projecteuclid.org/euclid.bsmsp/1200512173)",
    },
    {
        "name": "sctransform regularized NB regression",
        "source_field": "GLM / count modeling",
        "core_idea": "Use regularized negative-binomial regression to stabilize variance under sequencing-depth and count noise.",
        "relation": "Highly relevant to the count-model-friendly and batch-heavy pockets where anchor headroom remains.",
        "reference": "[Hafemeister and Satija 2019](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-019-1874-1)",
    },
    {
        "name": "GLM-PCA / deviance residual modeling",
        "source_field": "GLM / count modeling",
        "core_idea": "Model counts with Poisson or multinomial structure and keep genes with large deviance from the null count model.",
        "relation": "Directly relevant because multinomial deviance repeatedly appears near the remaining headroom frontier.",
        "reference": "[Townes et al. 2019](https://doi.org/10.1186/s13059-019-1861-6)",
    },
    {
        "name": "Analytic Pearson residuals",
        "source_field": "count residual modeling",
        "core_idea": "Use analytically stabilized Pearson residual variance as a fast count-aware signal.",
        "relation": "Transferable as a lightweight count bridge inside a single scorer without re-opening selector logic.",
        "reference": "[Lause et al. 2021](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-021-02451-7)",
    },
    {
        "name": "M3Drop / dropout-aware feature selection",
        "source_field": "dropout modeling",
        "core_idea": "Use the mean-dropout relationship to find genes whose zeros carry informative structure.",
        "relation": "Relevant mainly to the high-dropout trajectory-like subset, but too one-dimensional to be a full mainline by itself.",
        "reference": "[Andrews and Hemberg 2018](https://academic.oup.com/bioinformatics/article/35/16/2865/5257060)",
    },
    {
        "name": "triku nearest-neighbor enrichment",
        "source_field": "single-cell graph locality",
        "core_idea": "Favor genes whose expression is locally enriched in transcriptionally similar neighborhoods rather than globally dispersed.",
        "relation": "Directly relevant because GBM-like headroom looks triku-like rather than purely mean-variance-like.",
        "reference": "[Ascension et al. 2022](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-022-02629-7)",
    },
    {
        "name": "Robust Rank Aggregation",
        "source_field": "rank aggregation / meta-analysis",
        "core_idea": "Estimate whether an item is consistently near the top across ranked lists more often than chance.",
        "relation": "Conceptually relevant for fusing complementary gene-level signals, but the previous cheap rank probe already warns that naive rank fusion is not enough.",
        "reference": "[Kolde et al. 2012](https://academic.oup.com/bioinformatics/article/28/4/573/213339)",
    },
    {
        "name": "Borda / voting aggregation",
        "source_field": "social choice / meta-ranking",
        "core_idea": "Fuse rankings by position without trusting score scale comparability.",
        "relation": "Relevant as a baseline family, but mostly surface-level unless paired with regime-aware risk control.",
        "reference": "[Borda count overview](https://en.wikipedia.org/wiki/Borda_count)",
    },
    {
        "name": "Stability selection",
        "source_field": "stability selection",
        "core_idea": "Prefer variables that remain selected under subsampling or perturbation rather than single-pass peaks.",
        "relation": "Very relevant because the benchmark headroom is mostly in stability and clustering rather than marker recovery.",
        "reference": "[Meinshausen and Buhlmann 2010](https://academic.oup.com/jrsssb/article/72/4/417/7076513)",
    },
    {
        "name": "Cepo differential stability",
        "source_field": "single-cell marker stability",
        "core_idea": "Rank genes by stability of differential signal rather than only mean shifts.",
        "relation": "Transferable as a reminder that stability-weighted signal can be useful, though Cepo itself is marker-oriented rather than HVG-oriented.",
        "reference": "[Kim et al. 2021](https://www.nature.com/articles/s41467-021-23841-5)",
    },
    {
        "name": "Laplacian Score",
        "source_field": "graph spectral methods",
        "core_idea": "Reward features that preserve local manifold geometry on a neighborhood graph.",
        "relation": "Relevant because some remaining headroom looks like local structure and cluster separation without global biology shifts.",
        "reference": "[He et al. 2005](https://papers.nips.cc/paper_files/paper/2005/hash/b5b03f06271f8917685d14cea7c6c50a-Abstract.html)",
    },
    {
        "name": "MCFS multi-cluster feature selection",
        "source_field": "spectral feature selection",
        "core_idea": "Use multi-cluster spectral structure to identify features aligned with latent partitions.",
        "relation": "Relevant as a graph/spectral family, though heavier than needed for a smoke-stage anchor upgrade.",
        "reference": "[Cai et al. 2010](https://dl.acm.org/doi/10.1145/1835804.1835848)",
    },
    {
        "name": "mRMR minimum redundancy maximum relevance",
        "source_field": "information theory",
        "core_idea": "Prefer features that are informative yet non-redundant with already selected features.",
        "relation": "Useful conceptually for avoiding duplicated HVGs, but selection-order dependence makes it awkward as a single fast scorer.",
        "reference": "[Peng et al. 2005](https://ieeexplore.ieee.org/document/1453511)",
    },
    {
        "name": "Information bottleneck feature scoring",
        "source_field": "information theory",
        "core_idea": "Keep features that preserve mutual information with a latent representation while compressing noise.",
        "relation": "Interesting for theory, but too indirect for the current benchmark where we need cheap targeted smoke tests first.",
        "reference": "[Tishby et al. 1999](https://arxiv.org/abs/physics/0004057)",
    },
    {
        "name": "Graph signal denoising / Moran-like autocorrelation",
        "source_field": "graph signal processing",
        "core_idea": "Score features by graph-local autocorrelation or residual energy after graph smoothing.",
        "relation": "Strongly transferable to triku-like headroom without importing an entire new selector bank.",
        "reference": "[Shuman et al. 2013](https://ieeexplore.ieee.org/document/6515121)",
    },
    {
        "name": "Equal risk contribution / risk parity",
        "source_field": "portfolio theory",
        "core_idea": "Combine signals so no one component dominates the risk budget, encouraging balanced contribution.",
        "relation": "Transferable to multi-signal gene scoring because current winners are dispersed and any one signal can collapse on controls.",
        "reference": "[Maillard et al. 2010](https://ssrn.com/abstract=2009778)",
    },
    {
        "name": "Invariant Risk Minimization",
        "source_field": "causal invariance",
        "core_idea": "Favor predictors whose signal persists across environments rather than one environment-specific shortcut.",
        "relation": "Highly relevant to batch-heavy regimes if environments are interpreted as batches or pseudo-batches.",
        "reference": "[Arjovsky et al. 2019](https://arxiv.org/abs/1907.02893)",
    },
    {
        "name": "Pareto multi-objective optimization",
        "source_field": "multi-objective optimization",
        "core_idea": "Keep candidates that are jointly good across several objectives without reducing them too early to a scalar.",
        "relation": "Transferable because the benchmark evidence says no single scorer dominates all positive-headroom datasets.",
        "reference": "[Deb 2001](https://doi.org/10.1002/9780470052821)",
    },
    {
        "name": "Insurance hypothesis / diversity-stability",
        "source_field": "ecology",
        "core_idea": "Systems are more stable when multiple partially redundant components cover different shocks.",
        "relation": "Useful as a design metaphor for score ensembles, but only meaningful if translated into explicit gene-level robustness formulas.",
        "reference": "[Yachi and Loreau 1999](https://www.pnas.org/doi/10.1073/pnas.96.4.1463)",
    },
    {
        "name": "BYOL agreement regularization",
        "source_field": "self-supervised representation learning",
        "core_idea": "Promote representations that agree across views without relying on negative pairs.",
        "relation": "Transferable only at the principle level: agreement should be rewarded, but full self-supervised training is too heavy for this benchmark round.",
        "reference": "[Grill et al. 2020](https://papers.nips.cc/paper_files/paper/2020/hash/f3ada80d5c4ee70142b17b8192b2958e-Abstract.html)",
    },
    {
        "name": "VICReg",
        "source_field": "agreement / regularization",
        "core_idea": "Balance invariance, variance, and covariance terms so agreement does not collapse diversity.",
        "relation": "Conceptually useful because any consensus HVG scorer must avoid collapsing to everywhere-flat agreement.",
        "reference": "[Bardes et al. 2021](https://arxiv.org/abs/2105.04906)",
    },
    {
        "name": "GeneBasis",
        "source_field": "stable single-cell gene selection",
        "core_idea": "Select genes that preserve cell-state neighborhoods and remain generalizable across tissues.",
        "relation": "Relevant as a stability-and-coverage family, but closer to marker panel design than fast HVG ranking.",
        "reference": "[Marinov et al. 2023](https://www.cell.com/cell-systems/fulltext/S2405-4712(23)00124-0)",
    },
    {
        "name": "scGeneFit",
        "source_field": "compressed sensing / marker selection",
        "core_idea": "Use optimization to find compact marker panels that preserve cell-type separability.",
        "relation": "Interesting but mostly surface-similar; marker compression is not the remaining bottleneck in this benchmark.",
        "reference": "[Dumitrascu et al. 2021](https://www.nature.com/articles/s41592-021-01265-6)",
    },
]


LONG_LIST = [
    {
        "idea_name": "adaptive_eb_shrinkage_hvg",
        "family": "shrinkage / empirical Bayes",
        "source_field": "empirical Bayes",
        "one_liner": "Shrink noisy multi-signal gene scores toward a conservative prior when scorers disagree.",
        "mechanism": "Posterior-like score fusion over variance, MV residual, Pearson residual, Fano, and deviance ranks with disagreement-dependent shrinkage.",
        "expected_target_regime": "batch-heavy plus count-model-friendly non-atlas regimes",
        "why_effective": "Headroom is structured and winner methods are dispersed; shrinkage can use the extra signals without inducing an everywhere shift.",
        "not_selector_reason": "No dataset-level routing, no abstain stage, and no new expert bank policy layer.",
        "minimal_implementation_cost": "low",
        "failure_reason": "Could over-shrink and collapse back to the current anchor on the same 10/12 near-tied datasets.",
        "mathematical_form": "S_i = p_i + lambda_i (t_i - p_i), lambda_i = tau^2 / (tau^2 + Var_j z_ij).",
        "reference_support": "Smyth 2004; James-Stein 1961; Hafemeister and Satija 2019",
        "benchmark_fit_score": 5,
        "mechanism_plausibility_score": 5,
        "novelty_score": 4,
        "implementation_ease_score": 4,
        "smoke_testability": 5,
        "noise_robustness_score": 4,
        "atlas_safety_score": 4,
    },
    {
        "idea_name": "adaptive_invariant_residual_hvg",
        "family": "invariance / count residual",
        "source_field": "causal invariance + GLM",
        "one_liner": "Reward genes whose count-aware residual signal remains strong across batches or pseudo-environments.",
        "mechanism": "Compute per-environment deviance-style ranks and score genes by worst-group plus mean-group strength minus environment variance.",
        "expected_target_regime": "batch-heavy, donor-heterogeneous, count-model-friendly datasets",
        "why_effective": "The best remaining gains are in batch-heavy kidney and donor-mixed panels where one-environment shortcuts are risky.",
        "not_selector_reason": "The environments act inside a single gene scorer rather than deciding which method to release.",
        "minimal_implementation_cost": "medium",
        "failure_reason": "Pseudo-environments may be too noisy and suppress useful but environment-specific rare programs.",
        "mathematical_form": "S_i = q25_e(d_ie) + alpha mean_e(d_ie) - beta std_e(d_ie) + gamma mean_e(r_ie).",
        "reference_support": "Townes et al. 2019; Lause et al. 2021; Arjovsky et al. 2019",
        "benchmark_fit_score": 5,
        "mechanism_plausibility_score": 4,
        "novelty_score": 5,
        "implementation_ease_score": 3,
        "smoke_testability": 4,
        "noise_robustness_score": 4,
        "atlas_safety_score": 4,
    },
    {
        "idea_name": "adaptive_spectral_locality_hvg",
        "family": "graph / spectral",
        "source_field": "graph spectral methods",
        "one_liner": "Add a graph-locality term that rewards genes with neighborhood-consistent structure rather than only global dispersion.",
        "mechanism": "Build a fast PCA kNN graph and score graph-local autocorrelation blended with count-aware residual signals.",
        "expected_target_regime": "high-dropout trajectory-like and local-neighborhood-structured datasets",
        "why_effective": "GBM-like headroom looks more triku-like than classical anchor-like, so graph locality could unlock the missing cluster separation.",
        "not_selector_reason": "A single graph-aware scorer, not a graph-based router over existing experts.",
        "minimal_implementation_cost": "medium",
        "failure_reason": "Could overfit local smoothness and drift too far on atlas-like controls that already prefer variance-like behavior.",
        "mathematical_form": "S_i = z(corr(x_i, P x_i) - eta ||x_i - P x_i||^2) + bridge_i.",
        "reference_support": "He et al. 2005; Shuman et al. 2013; Ascension et al. 2022",
        "benchmark_fit_score": 4,
        "mechanism_plausibility_score": 4,
        "novelty_score": 5,
        "implementation_ease_score": 3,
        "smoke_testability": 3,
        "noise_robustness_score": 3,
        "atlas_safety_score": 3,
    },
    {
        "idea_name": "adaptive_stability_jackknife_hvg",
        "family": "stability / resampling",
        "source_field": "stability selection",
        "one_liner": "Reward genes whose high ranks survive deterministic split perturbations of the dataset.",
        "mechanism": "Fuse full-data scores with leave-one-split-out rank consistency across pseudo-replicates.",
        "expected_target_regime": "datasets where residual headroom is mostly stability and clustering",
        "why_effective": "Current evidence says the remaining gains are more about stability than biology proxies, making a direct stability prior attractive.",
        "not_selector_reason": "Uses resampling to regularize gene scores, not to choose between methods.",
        "minimal_implementation_cost": "medium",
        "failure_reason": "May simply re-encode the anchor's current conservatism and fail to create enough regime-specific movement.",
        "mathematical_form": "S_i = (1-rho) s_i^full + rho (mean_r s_ir - lambda std_r s_ir).",
        "reference_support": "Meinshausen and Buhlmann 2010; GeneBasis 2023; Cepo 2021",
        "benchmark_fit_score": 4,
        "mechanism_plausibility_score": 4,
        "novelty_score": 4,
        "implementation_ease_score": 3,
        "smoke_testability": 4,
        "noise_robustness_score": 5,
        "atlas_safety_score": 4,
    },
    {
        "idea_name": "adaptive_risk_parity_hvg",
        "family": "portfolio / multi-objective",
        "source_field": "risk parity",
        "one_liner": "Fuse multiple gene signals with balanced risk contributions instead of trusting whichever score dominates.",
        "mechanism": "Compute risk-parity weights from scorer covariance and penalize one-scorer concentration.",
        "expected_target_regime": "mixed-regime datasets where different scorers each capture part of the missing signal",
        "why_effective": "Winner dispersion suggests complementary signals, and balanced weighting may keep atlas drift under control.",
        "not_selector_reason": "It never chooses among experts; it only risk-adjusts a single joint score.",
        "minimal_implementation_cost": "low",
        "failure_reason": "Could be too generic and behave like a nicer-looking version of the failed consensus probes.",
        "mathematical_form": "S_i = w^T z_i + alpha tail(z_i) - beta sd(z_i), w propto Sigma^{-1} 1.",
        "reference_support": "Maillard et al. 2010; Deb 2001; VICReg 2021",
        "benchmark_fit_score": 4,
        "mechanism_plausibility_score": 3,
        "novelty_score": 4,
        "implementation_ease_score": 5,
        "smoke_testability": 5,
        "noise_robustness_score": 3,
        "atlas_safety_score": 4,
    },
    {
        "idea_name": "adaptive_count_bridge_glm_hvg",
        "family": "count-model residual",
        "source_field": "GLM",
        "one_liner": "Inject a continuous GLM count-bridge term into the anchor core without any extra routing layer.",
        "mechanism": "Blend deviance and Pearson residual structure into the adaptive-stat core with profile-conditioned bridge weights.",
        "expected_target_regime": "count-model-friendly and donor-heterogeneous datasets",
        "why_effective": "Multinomial deviance keeps showing up near the frontier, so a more disciplined count bridge remains plausible.",
        "not_selector_reason": "Pure score fusion, no policy logic.",
        "minimal_implementation_cost": "low",
        "failure_reason": "Very close to the previous cheap probes and may again fail the targeted perturbation test.",
        "mathematical_form": "S_i = (1-b) c_i + b (a d_i + (1-a) p_i).",
        "reference_support": "Townes et al. 2019; Lause et al. 2021",
        "benchmark_fit_score": 5,
        "mechanism_plausibility_score": 4,
        "novelty_score": 3,
        "implementation_ease_score": 5,
        "smoke_testability": 5,
        "noise_robustness_score": 3,
        "atlas_safety_score": 3,
    },
    {
        "idea_name": "adaptive_pareto_front_hvg",
        "family": "multi-objective / Pareto",
        "source_field": "Pareto optimization",
        "one_liner": "Prefer genes that are jointly strong across several scorers rather than elite in only one.",
        "mechanism": "Approximate Pareto-front depth from several rank vectors and use it as the final score.",
        "expected_target_regime": "heterogeneous positive-headroom datasets with dispersed winner signals",
        "why_effective": "A multi-objective view matches the no-single-winner evidence better than another scalar threshold sweep.",
        "not_selector_reason": "Operates per gene, not per dataset.",
        "minimal_implementation_cost": "medium",
        "failure_reason": "Can flatten useful score magnitudes and converge toward the anchor or generic rank fusion.",
        "mathematical_form": "S_i = depth(z_i) + alpha min_j z_ij - beta dispersion(z_i).",
        "reference_support": "Deb 2001; Pareto feature selection literature",
        "benchmark_fit_score": 4,
        "mechanism_plausibility_score": 3,
        "novelty_score": 4,
        "implementation_ease_score": 3,
        "smoke_testability": 4,
        "noise_robustness_score": 3,
        "atlas_safety_score": 4,
    },
    {
        "idea_name": "adaptive_robust_rank_meta_hvg",
        "family": "rank aggregation",
        "source_field": "meta-analysis",
        "one_liner": "Use a robust-rank-aggregation style p-value over multiple scorer rankings.",
        "mechanism": "Convert rank positions into order-statistic significance and keep genes that repeatedly land near the top.",
        "expected_target_regime": "mixed regimes where score scales are mismatched",
        "why_effective": "Still plausible in theory, but the last round already warned that generic rank aggregation is too weak unless made more targeted.",
        "not_selector_reason": "No routing and no dataset-level decision stage.",
        "minimal_implementation_cost": "low",
        "failure_reason": "Too close to the already failed cheap rank-aggregate probe.",
        "mathematical_form": "S_i = -log p_RRA(rank_i1, ..., rank_iJ).",
        "reference_support": "Kolde et al. 2012",
        "benchmark_fit_score": 3,
        "mechanism_plausibility_score": 2,
        "novelty_score": 2,
        "implementation_ease_score": 5,
        "smoke_testability": 5,
        "noise_robustness_score": 2,
        "atlas_safety_score": 3,
    },
    {
        "idea_name": "adaptive_differential_stability_hvg",
        "family": "stability / marker-style",
        "source_field": "single-cell stability",
        "one_liner": "Translate differential-stability ideas into unsupervised HVG ranking using pseudo-clusters.",
        "mechanism": "Estimate pseudo-cluster-specific effect stability and reward genes with reproducible pseudo-cluster contrast.",
        "expected_target_regime": "datasets where cluster silhouette is the main residual headroom",
        "why_effective": "Potentially aligned with the benchmark headroom, but it depends heavily on pseudo-cluster quality.",
        "not_selector_reason": "All operations are inside the scorer and use no expert release policy.",
        "minimal_implementation_cost": "medium",
        "failure_reason": "Pseudo-cluster leakage can make it circular and unstable.",
        "mathematical_form": "S_i = mean_c Delta_ic - lambda sd_c Delta_ic over pseudo-clusters.",
        "reference_support": "Cepo 2021",
        "benchmark_fit_score": 3,
        "mechanism_plausibility_score": 3,
        "novelty_score": 4,
        "implementation_ease_score": 3,
        "smoke_testability": 3,
        "noise_robustness_score": 3,
        "atlas_safety_score": 3,
    },
    {
        "idea_name": "adaptive_laplacian_residual_hvg",
        "family": "graph / spectral",
        "source_field": "Laplacian regularization",
        "one_liner": "Blend mean-variance residuals with graph-Laplacian preservation.",
        "mechanism": "Use Laplacian score as a regularizer on top of residual-based ranks.",
        "expected_target_regime": "local-manifold regimes with moderate heterogeneity",
        "why_effective": "A lighter cousin of the spectral-locality idea that may still move toward triku-like behavior.",
        "not_selector_reason": "One graph-aware score, no router.",
        "minimal_implementation_cost": "medium",
        "failure_reason": "May duplicate spectral-locality ideas without enough extra signal.",
        "mathematical_form": "S_i = r_i - lambda (x_i^T L x_i / x_i^T D x_i).",
        "reference_support": "He et al. 2005",
        "benchmark_fit_score": 3,
        "mechanism_plausibility_score": 3,
        "novelty_score": 3,
        "implementation_ease_score": 3,
        "smoke_testability": 3,
        "noise_robustness_score": 3,
        "atlas_safety_score": 3,
    },
    {
        "idea_name": "adaptive_mrmr_hvg",
        "family": "information theory",
        "source_field": "mRMR",
        "one_liner": "Re-rank top genes by relevance minus redundancy to avoid over-filling the set with one program.",
        "mechanism": "Use a fast redundancy penalty among high-ranked genes from the base scorer bank.",
        "expected_target_regime": "redundant atlas-like or donor-heavy panels",
        "why_effective": "Could reduce repeated genes, but current evidence says the main gap is not redundant atlas-like biology coverage.",
        "not_selector_reason": "Selection order changes, but there is still no dataset-level expert choice.",
        "minimal_implementation_cost": "high",
        "failure_reason": "Selection-order dependence and high compute cost make it ill-suited for this restart round.",
        "mathematical_form": "S_i = rel_i - beta mean_j< i MI(i, j).",
        "reference_support": "Peng et al. 2005",
        "benchmark_fit_score": 2,
        "mechanism_plausibility_score": 2,
        "novelty_score": 4,
        "implementation_ease_score": 1,
        "smoke_testability": 2,
        "noise_robustness_score": 2,
        "atlas_safety_score": 3,
    },
    {
        "idea_name": "adaptive_information_bottleneck_hvg",
        "family": "information theory",
        "source_field": "information bottleneck",
        "one_liner": "Score genes by how much latent cluster information they preserve after aggressive compression.",
        "mechanism": "Use an unsupervised latent representation and keep features with the highest marginal information contribution.",
        "expected_target_regime": "representation-heavy datasets",
        "why_effective": "Interesting for theory, but poorly matched to a smoke-stage question about a single fast scorer upgrade.",
        "not_selector_reason": "No routing layer.",
        "minimal_implementation_cost": "high",
        "failure_reason": "Too indirect, too expensive, and likely to create representation-learning project bloat.",
        "mathematical_form": "S_i approx I(gene_i; Z) - beta I(gene_i; X).",
        "reference_support": "Tishby et al. 1999",
        "benchmark_fit_score": 1,
        "mechanism_plausibility_score": 2,
        "novelty_score": 5,
        "implementation_ease_score": 1,
        "smoke_testability": 1,
        "noise_robustness_score": 2,
        "atlas_safety_score": 2,
    },
    {
        "idea_name": "adaptive_diversity_insurance_hvg",
        "family": "ecology / redundancy",
        "source_field": "ecology",
        "one_liner": "Reward genes that contribute complementary signal coverage across pseudo-environments.",
        "mechanism": "Use a diversity-insurance term so one signal family cannot dominate the full selected set.",
        "expected_target_regime": "heterogeneous donor or tissue mixtures",
        "why_effective": "Appealing as a design metaphor, but the mapping to a fast gene scorer is still loose.",
        "not_selector_reason": "No routing stage.",
        "minimal_implementation_cost": "high",
        "failure_reason": "Could become vague redundancy engineering rather than a crisp new anchor-core principle.",
        "mathematical_form": "S_i = mean_env s_ie + alpha diversity_gain_i.",
        "reference_support": "Yachi and Loreau 1999",
        "benchmark_fit_score": 2,
        "mechanism_plausibility_score": 2,
        "novelty_score": 4,
        "implementation_ease_score": 1,
        "smoke_testability": 2,
        "noise_robustness_score": 2,
        "atlas_safety_score": 3,
    },
    {
        "idea_name": "adaptive_self_supervised_agreement_hvg",
        "family": "agreement / self-supervised",
        "source_field": "representation learning",
        "one_liner": "Use view-agreement regularization over multiple gene-score augmentations.",
        "mechanism": "Construct view pairs from different scorers or subsamples and keep genes with high agreement but non-collapsed variance.",
        "expected_target_regime": "mixed non-atlas regimes",
        "why_effective": "Agreement principles are relevant, but full SSL machinery is probably too much for this benchmark stage.",
        "not_selector_reason": "Agreement happens within one scorer.",
        "minimal_implementation_cost": "high",
        "failure_reason": "Too easy to turn into a representation-learning side project unrelated to the benchmark bottleneck.",
        "mathematical_form": "S_i = inv_i + alpha var_i - beta cov_i across views.",
        "reference_support": "BYOL 2020; VICReg 2021",
        "benchmark_fit_score": 2,
        "mechanism_plausibility_score": 2,
        "novelty_score": 5,
        "implementation_ease_score": 1,
        "smoke_testability": 1,
        "noise_robustness_score": 2,
        "atlas_safety_score": 3,
    },
    {
        "idea_name": "adaptive_wavelet_denoise_hvg",
        "family": "signal processing / denoising",
        "source_field": "signal processing",
        "one_liner": "Denoise gene profiles on a cell graph before scoring dispersion.",
        "mechanism": "Apply graph- or trajectory-aware denoising and score residual energy after smoothing.",
        "expected_target_regime": "trajectory-like and dropout-heavy datasets",
        "why_effective": "Could help with noisy trajectories, but is less direct than explicit graph-locality scoring.",
        "not_selector_reason": "Single scorer transformation.",
        "minimal_implementation_cost": "high",
        "failure_reason": "Denoising can wash out exactly the rare, sharp programs that create cluster separation gains.",
        "mathematical_form": "S_i = ||x_i - Denoise(x_i)||^2 + alpha disp_i.",
        "reference_support": "graph signal denoising literature",
        "benchmark_fit_score": 2,
        "mechanism_plausibility_score": 2,
        "novelty_score": 4,
        "implementation_ease_score": 1,
        "smoke_testability": 2,
        "noise_robustness_score": 2,
        "atlas_safety_score": 2,
    },
    {
        "idea_name": "adaptive_cvar_worstgroup_hvg",
        "family": "robust optimization",
        "source_field": "risk-sensitive optimization",
        "one_liner": "Rank genes by a CVaR-style score over pseudo-environment-specific strengths.",
        "mechanism": "Replace mean aggregation with downside-sensitive worst-group aggregation.",
        "expected_target_regime": "batch-heavy heterogeneous datasets",
        "why_effective": "Potentially strong for donor/batch robustness, but may be too pessimistic when environments are only approximate.",
        "not_selector_reason": "Still a single score, not a release policy.",
        "minimal_implementation_cost": "medium",
        "failure_reason": "Worst-group pessimism may kill useful rare-program genes.",
        "mathematical_form": "S_i = CVaR_q({s_ie}_e) + alpha global_i.",
        "reference_support": "distributionally robust optimization literature",
        "benchmark_fit_score": 3,
        "mechanism_plausibility_score": 3,
        "novelty_score": 4,
        "implementation_ease_score": 3,
        "smoke_testability": 3,
        "noise_robustness_score": 4,
        "atlas_safety_score": 4,
    },
]


TOP5_INFO = {
    "adaptive_eb_shrinkage_hvg": {
        "family": "shrinkage / empirical Bayes",
        "why_top5": "Best benchmark-fit to novelty balance and the cleanest way to exploit dispersed winner signals without opening another routing layer.",
        "headroom_target": "batch-heavy and count-model-friendly headroom while shrinking back on atlas-like controls.",
        "math_note": "Posterior-like blend over base score ranks with disagreement-dependent shrinkage toward an atlas-safe prior.",
        "references": [
            "Smyth 2004",
            "James and Stein 1961",
            "Hafemeister and Satija 2019",
        ],
        "minimal_change_path": "Add one new anchor-core scorer that reuses existing variance, MV residual, Fano, deviance, and Pearson residual signals.",
        "smoke_plan": "Check whether it pulls positive-headroom datasets toward the right winners more than controls, then benchmark only if the pull is targeted.",
        "failure_mode": "Too much shrinkage turns it into a prettier version of the anchor with no real perturbation.",
    },
    "adaptive_invariant_residual_hvg": {
        "family": "invariance / count residual",
        "why_top5": "Most directly aimed at donor and batch heterogeneity, which is exactly where the remaining anchor headroom still lives.",
        "headroom_target": "cellxgene_human_kidney_nonpt and similar donor-heavy datasets; secondarily paul15-like count-model regimes.",
        "math_note": "Worst-group and mean-group deviance ranks across pseudo-environments, penalized by environment variance.",
        "references": [
            "Townes et al. 2019",
            "Lause et al. 2021",
            "Arjovsky et al. 2019",
        ],
        "minimal_change_path": "Reuse existing count scorers and only add pseudo-environment construction plus worst-group aggregation.",
        "smoke_plan": "Require a larger pull toward winner references on positive-headroom donor datasets than on atlas controls.",
        "failure_mode": "Pseudo-environment noise may suppress real biology if the environment split is not meaningful.",
    },
    "adaptive_spectral_locality_hvg": {
        "family": "graph / spectral",
        "why_top5": "Most distinct mechanism family and the best shot at the GBM/triku-style structural gap that the current anchor clearly misses.",
        "headroom_target": "high-dropout trajectory-like and neighborhood-structured clustering gains.",
        "math_note": "Graph-local autocorrelation minus residual energy on a PCA kNN graph, then blended with count-aware ranks.",
        "references": [
            "He et al. 2005",
            "Shuman et al. 2013",
            "Ascension et al. 2022",
        ],
        "minimal_change_path": "Build one small kNN graph on sampled cells and use it as an additional gene-level prior.",
        "smoke_plan": "Look for GBM-specific movement toward triku-like signals without causing a similar shift on atlas controls.",
        "failure_mode": "Can overshoot into generic locality and damage atlas-like behavior.",
    },
    "adaptive_stability_jackknife_hvg": {
        "family": "stability / resampling",
        "why_top5": "Directly matches the current evidence that most remaining headroom is in stability and cluster silhouette rather than biology guardrails.",
        "headroom_target": "positive-headroom clustering/stability space without forcing a strong new signal family.",
        "math_note": "Full-data score plus leave-one-split-out stability mean minus rank variance penalty.",
        "references": [
            "Meinshausen and Buhlmann 2010",
            "GeneBasis 2023",
            "Cepo 2021",
        ],
        "minimal_change_path": "No new external method family; only deterministic splits over existing score components.",
        "smoke_plan": "Require stability-oriented movement on the positive subset and near-anchor behavior on controls.",
        "failure_mode": "Can become too conservative and fail to move enough.",
    },
    "adaptive_risk_parity_hvg": {
        "family": "portfolio / multi-objective",
        "why_top5": "Covers the ensemble-weighting family with a more disciplined mechanism than generic consensus or trimmed ranks.",
        "headroom_target": "mixed non-atlas regimes where no one scorer is dominant but over-reliance on any one scorer is risky.",
        "math_note": "Covariance-adjusted risk-parity weighting plus support-tail and concentration terms.",
        "references": [
            "Maillard et al. 2010",
            "Deb 2001",
            "Bardes et al. 2021",
        ],
        "minimal_change_path": "Purely algebraic fusion over already available score vectors.",
        "smoke_plan": "Needs to show better positive-vs-control targeting than the previous consensus probe; otherwise it is dead on arrival.",
        "failure_mode": "May still collapse into a generic consensus and repeat the old failure mode.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Broad-research -> BFS -> top5 -> smoke -> final decision restart.")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--real-data-root", type=str, default="data/real_inputs")
    parser.add_argument("--gate-model-path", type=str, default=rr1.DEFAULT_GATE_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=2)
    parser.add_argument("--refine-epochs", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    evidence = load_evidence_tables()
    write_broad_research_outputs(output_dir=output_dir, evidence=evidence)

    longlist_df = build_longlist_df()
    longlist_df.to_csv(output_dir / "idea_bfs_longlist.csv", index=False)
    write_bfs_ranking(output_dir=output_dir, longlist_df=longlist_df, evidence=evidence)

    top5_df = build_top5_df(longlist_df)
    top5_df.to_csv(output_dir / "top5_candidates.csv", index=False)
    write_top5_candidates(output_dir=output_dir, top5_df=top5_df)

    resources = load_dataset_resources(real_data_root=(ROOT / args.real_data_root).resolve())
    dataset_cache = pkg.DatasetCache(resources)

    write_smoke_protocol(output_dir=output_dir)
    analysis_df = run_analysis_level_smoke(
        dataset_cache=dataset_cache,
        dataset_names=SMOKE_DATASETS,
        candidate_methods=tuple(top5_df["idea_name"].astype(str).tolist()),
        gate_model_path=str(Path(args.gate_model_path).resolve()),
        top_k=int(args.top_k),
        seed=int(args.seed),
        refine_epochs=int(args.refine_epochs),
        evidence=evidence,
    )
    analysis_summary = summarize_analysis_smoke(analysis_df)

    benchmark_raw = pd.DataFrame()
    benchmark_summary = pd.DataFrame()
    passing_methods = analysis_summary[analysis_summary["analysis_pass"] == True]["method"].astype(str).tolist()  # noqa: E712
    if passing_methods:
        benchmark_methods = tuple([SAFE_ANCHOR_METHOD, *passing_methods])
        benchmark_raw = run_dataset_benchmark(
            dataset_cache=dataset_cache,
            dataset_names=SMOKE_DATASETS,
            method_names=benchmark_methods,
            gate_model_path=str(Path(args.gate_model_path).resolve()),
            top_k=int(args.top_k),
            seed=int(args.seed),
            bootstrap_samples=int(args.bootstrap_samples),
            refine_epochs=int(args.refine_epochs),
        )
        biology_df = run_biology_proxy(
            dataset_cache=dataset_cache,
            dataset_names=SMOKE_DATASETS,
            method_names=benchmark_methods,
            gate_model_path=str(Path(args.gate_model_path).resolve()),
            top_k=int(args.top_k),
            seed=int(args.seed),
            refine_epochs=int(args.refine_epochs),
        )
        benchmark_raw = benchmark_raw.merge(
            biology_df[["dataset", "method", "weighted_marker_recall_at_50"]],
            on=["dataset", "method"],
            how="left",
        )
        benchmark_summary = summarize_benchmark_smoke(benchmark_raw)

    write_smoke_results(
        output_dir=output_dir,
        analysis_df=analysis_df,
        analysis_summary=analysis_summary,
        benchmark_raw=benchmark_raw,
        benchmark_summary=benchmark_summary,
        evidence=evidence,
    )
    write_final_recommendation(
        output_dir=output_dir,
        top5_df=top5_df,
        analysis_summary=analysis_summary,
        benchmark_summary=benchmark_summary,
    )


def load_evidence_tables() -> dict[str, pd.DataFrame]:
    failure_df = pd.read_csv(ROOT / "artifacts_topconf_selector_round2" / "failure_taxonomy.csv")
    manifest_df = pd.read_csv(ROOT / "artifacts_topconf_selector_round2" / "dataset_manifest.csv")
    headroom_df = pd.read_csv(ROOT / "artifacts_next_direction" / "anchor_headroom_tables.csv")
    headroom_df = headroom_df[headroom_df["row_type"] == "dataset"].copy()
    comparison_df = pd.read_csv(ROOT / "artifacts_adaptive_hybrid_litaware_eval" / "benchmark_method_comparison.csv")
    hybrid_global_df = pd.read_csv(ROOT / "artifacts_adaptive_hybrid_litaware_eval" / "benchmark_global_summary.csv")
    return {
        "failure_df": failure_df,
        "manifest_df": manifest_df,
        "headroom_df": headroom_df,
        "comparison_df": comparison_df,
        "hybrid_global_df": hybrid_global_df,
    }


def write_broad_research_outputs(*, output_dir: Path, evidence: dict[str, pd.DataFrame]) -> None:
    headroom_df = evidence["headroom_df"].copy()
    positive_df = headroom_df[headroom_df["headroom_vs_best_single"] > 0].copy()
    atlas_df = headroom_df[headroom_df["regime"] == "atlas-like / large homogeneous panel"].copy()
    comparison_df = evidence["comparison_df"].copy()
    near_tie_count = int(np.sum(np.abs(pd.to_numeric(comparison_df["score_delta"], errors="coerce").fillna(0.0)) <= 0.01))
    total_pair_rows = int(len(comparison_df))

    literature_lines = [
        "# Literature Map",
        "",
        "Internet access was available for this restart, so the map below combines repo artifacts with primary method papers or primary method pages.",
        "",
    ]
    for idx, entry in enumerate(LITERATURE_ENTRIES, start=1):
        literature_lines.extend(
            [
                f"## {idx}. {entry['name']}",
                f"- Source field: {entry['source_field']}",
                f"- Core idea: {entry['core_idea']}",
                f"- Relation to this benchmark: {entry['relation']}",
                f"- Key reference: {entry['reference']}",
                "",
            ]
        )
    (output_dir / "literature_map.md").write_text("\n".join(literature_lines) + "\n", encoding="utf-8")

    research_lines = [
        "# Research Notes",
        "",
        "## Repo-Constrained Starting Point",
        f"- Positive-headroom datasets in the current evidence table: {len(positive_df)} / {len(headroom_df)}.",
        f"- Mean headroom on the positive subset: {float(positive_df['headroom_vs_best_single'].mean()):.4f}.",
        f"- Mean headroom on atlas-like controls: {float(atlas_df['headroom_vs_best_single'].mean()):.4f}.",
        f"- `adaptive_hybrid_hvg` vs `adaptive_stat_hvg` near-ties in the existing comparison table: {near_tie_count} / {total_pair_rows} rows within |delta| <= 0.01.",
        "",
        "## Most Transferable Families",
        "- Empirical Bayes / shrinkage ideas are genuinely transferable because they directly address noisy gene-rank disagreement inside one scorer.",
        "- Count-aware residual ideas remain transferable because multinomial-deviance-like behavior keeps appearing near the frontier on positive-headroom datasets.",
        "- Stability-selection ideas are transferable because the remaining score gap is concentrated in clustering and stability rather than marker recall.",
        "- Graph locality and spectral ideas are transferable specifically because GBM-like headroom looks neighborhood-structured rather than globally variance-driven.",
        "- Invariance ideas are transferable for donor- or batch-heavy datasets because they regularize against one-environment shortcuts without re-opening a selector.",
        "",
        "## Surface-Similar But Weakly Transferable",
        "- Generic rank aggregation is only weakly transferable: the previous cheap rank probe already failed the targeted perturbation test.",
        "- Marker compression methods such as scGeneFit or GeneBasis are useful inspiration, but their main objective is not the current benchmark bottleneck.",
        "- Full self-supervised feature learning is too indirect for this round and risks turning the project into representation-learning sprawl.",
        "- Official reproduction and bank cleanup are still scientifically useful, but the current evidence says they are no longer the main rate-limiting step.",
        "",
        "## Search-Space Compression After Broad Research",
        "- The broad search did not reopen selector/routing as a credible mainline because the benchmark headroom is now mostly anchor-core space.",
        "- The search also did not support another purely generic consensus scorer; any surviving candidate must produce regime-aware movement toward the right winner signals.",
        "- The credible restart space compresses to five broad families: shrinkage, invariance-aware count residuals, graph-locality priors, split-stability priors, and disciplined multi-objective weighting.",
        "",
        "## Practical Implication",
        "- The top-5 shortlist below is intentionally diverse across those five families rather than five tiny variants of the last consensus probe.",
        "",
    ]
    (output_dir / "research_notes.md").write_text("\n".join(research_lines) + "\n", encoding="utf-8")


def build_longlist_df() -> pd.DataFrame:
    df = pd.DataFrame(LONG_LIST).copy()
    df["implementation_cost"] = df["minimal_implementation_cost"]
    df["risk_of_collapsing_into_noise"] = 6 - df["noise_robustness_score"]
    df["risk_of_harming_atlas_like_controls"] = 6 - df["atlas_safety_score"]
    df["bfs_total_score"] = sum(df[column] * weight for column, weight in BFS_WEIGHTS.items())
    df = df.sort_values(["bfs_total_score", "benchmark_fit_score", "smoke_testability"], ascending=[False, False, False]).reset_index(drop=True)
    df["bfs_rank"] = np.arange(1, len(df) + 1)
    return df[
        [
            "idea_name",
            "source_field",
            "mechanism",
            "expected_target_regime",
            "mathematical_form",
            "reference_support",
            "benchmark_fit_score",
            "novelty_score",
            "implementation_cost",
            "smoke_testability",
            "failure_reason",
            "bfs_rank",
            "family",
            "one_liner",
            "why_effective",
            "not_selector_reason",
            "mechanism_plausibility_score",
            "implementation_ease_score",
            "noise_robustness_score",
            "atlas_safety_score",
            "bfs_total_score",
        ]
    ].rename(columns={"failure_reason": "main_risk"})


def write_bfs_ranking(*, output_dir: Path, longlist_df: pd.DataFrame, evidence: dict[str, pd.DataFrame]) -> None:
    headroom_df = evidence["headroom_df"]
    regime_summary = (
        headroom_df.groupby("regime", as_index=False)
        .agg(
            dataset_count=("headroom_vs_best_single", "count"),
            mean_headroom=("headroom_vs_best_single", "mean"),
            median_headroom=("headroom_vs_best_single", "median"),
            max_headroom=("headroom_vs_best_single", "max"),
        )
    )
    top_rows = longlist_df.head(10)
    lines = [
        "# BFS Ranking",
        "",
        "## Transparent Scoring Rule",
        "- benchmark_fit_score (0.27): does the mechanism match the actual positive-headroom regimes in the repo evidence?",
        "- mechanism_plausibility_score (0.22): is there a concrete mechanism, not just a slogan?",
        "- novelty_score (0.15): is it meaningfully different from the existing anchor and the two failed cheap probes?",
        "- smoke_testability (0.14): can we cheaply tell whether it creates the right targeted movement?",
        "- implementation_ease_score (0.10): can it be prototyped without reopening a large engineering branch?",
        "- noise_robustness_score (0.07): how likely is it to avoid collapsing into generic noise?",
        "- atlas_safety_score (0.05): how likely is it to avoid damaging atlas-like controls?",
        "",
        "## Regime Reminder",
    ]
    for row in regime_summary.itertuples(index=False):
        lines.append(
            f"- {row.regime}: count={int(row.dataset_count)} mean_headroom={float(row.mean_headroom):.4f} "
            f"median={float(row.median_headroom):.4f} max={float(row.max_headroom):.4f}."
        )
    lines.extend(
        [
            "",
            "## Top-10 By BFS Score",
        ]
    )
    for row in top_rows.itertuples(index=False):
        lines.append(
            f"- rank {int(row.bfs_rank)} `{row.idea_name}` ({row.family}): total={float(row.bfs_total_score):.3f}, "
            f"fit={int(row.benchmark_fit_score)}, novelty={int(row.novelty_score)}, smoke={int(row.smoke_testability)}, risk={row.main_risk}"
        )
    lines.extend(
        [
            "",
            "## Why The Search Stayed Broad",
            "- The longlist intentionally kept graph, invariance, stability, shrinkage, portfolio, Pareto, and information-theoretic families alive until after scoring.",
            "- Generic consensus/rank ideas were not excluded a priori; they simply scored lower after the benchmark-fit and novelty penalties were applied.",
            "",
        ]
    )
    (output_dir / "bfs_ranking.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_top5_df(longlist_df: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    for method_name in TOP5_METHODS:
        row = longlist_df[longlist_df["idea_name"] == method_name].iloc[0]
        info = TOP5_INFO[method_name]
        selected_rows.append(
            {
                "idea_name": method_name,
                "family": info["family"],
                "bfs_rank": int(row["bfs_rank"]),
                "why_top5": info["why_top5"],
                "structural_headroom_target": info["headroom_target"],
                "mathematical_form": row["mathematical_form"],
                "reference_support": "; ".join(info["references"]),
                "minimal_change_path": info["minimal_change_path"],
                "smoke_plan": info["smoke_plan"],
                "failure_mode": info["failure_mode"],
            }
        )
    return pd.DataFrame(selected_rows)


def write_top5_candidates(*, output_dir: Path, top5_df: pd.DataFrame) -> None:
    lines = [
        "# Top-5 Candidate Summary",
        "",
        "The shortlist was not chosen as five tiny variants of one consensus trick. It intentionally covers five mechanism families that remain credible under the current benchmark evidence.",
        "",
    ]
    for row in top5_df.itertuples(index=False):
        lines.extend(
            [
                f"## {row.idea_name}",
                f"- Family: {row.family}",
                f"- Why it made top-5: {row.why_top5}",
                f"- Structural headroom target: {row.structural_headroom_target}",
                f"- Mathematical form: {row.mathematical_form}",
                f"- Reference support: {row.reference_support}",
                f"- Minimal change path from the current anchor: {row.minimal_change_path}",
                f"- Minimal smoke test: {row.smoke_plan}",
                f"- Most likely failure mode: {row.failure_mode}",
                "",
            ]
        )
    (output_dir / "top5_candidates.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_smoke_protocol(*, output_dir: Path) -> None:
    lines = [
        "# Smoke Protocol",
        "",
        "## Ordered Flow",
        "- analysis-level smoke on all top-5 candidates first",
        "- benchmark smoke only for analysis-passing candidates",
        "- final recommendation only after both filters",
        "",
        "## Analysis-Level Metrics",
        "- rank_corr_to_anchor",
        "- topk_overlap_to_anchor and topk_shift_vs_anchor",
        "- topk_overlap_to_best_single and delta_overlap_to_best_single_vs_anchor",
        "- rank_corr_to_best_single and delta_rank_corr_to_best_single_vs_anchor",
        "- score_dispersion_ratio_vs_anchor",
        "- delta correlations to variance, deviance, triku, scanpy_seurat_v3, scanpy_cell_ranger, and scran signals when available",
        "",
        "## Analysis Pass Rule",
        "- Condition 1: positive_minus_control_shift >= 0.03",
        "- Condition 2: positive_overlap_pull_vs_anchor > 0",
        "- Condition 3: overlap_pull_gap >= 0.015",
        "- Condition 4: corr_pull_gap >= 0.01",
        "- Condition 5: control_overlap_pull_vs_anchor >= -0.02",
        "- Pass if at least 4 of the 5 conditions hold.",
        "",
        "## Benchmark Smoke Datasets",
        "- GBM_sd",
        "- cellxgene_human_kidney_nonpt",
        "- paul15",
        "- cellxgene_immune_five_donors",
        "- mus_tissue",
        "- homo_tissue",
        "",
        "## Benchmark Pass Rule",
        "- positive-headroom subset mean delta vs adaptive_hybrid_hvg > 0",
        "- atlas-like controls mean delta >= -0.05",
        "- mean biology proxy delta >= -0.02",
        "- mean runtime ratio vs anchor <= 1.5",
        "- mean cluster delta + mean stability delta > 0",
        "",
    ]
    (output_dir / "smoke_protocol.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_dataset_resources(*, real_data_root: Path) -> pkg.DatasetResources:
    manifest_df = pd.read_csv(ROOT / "artifacts_topconf_selector_round2" / "dataset_manifest.csv")
    return pkg.load_dataset_resources(real_data_root=real_data_root, manifest_df=manifest_df)


def run_analysis_level_smoke(
    *,
    dataset_cache: pkg.DatasetCache,
    dataset_names: tuple[str, ...],
    candidate_methods: tuple[str, ...],
    gate_model_path: str,
    top_k: int,
    seed: int,
    refine_epochs: int,
    evidence: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    registry = build_default_method_registry(
        top_k=top_k,
        refine_epochs=refine_epochs,
        random_state=seed,
        gate_model_path=gate_model_path,
    )
    headroom_lookup = evidence["headroom_df"].set_index("dataset")
    rows: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        dataset = dataset_cache.get(dataset_name, seed)
        current_top_k = min(top_k, dataset.counts.shape[1])
        best_single_method = str(headroom_lookup.loc[dataset_name, "best_single_expert"])
        methods_to_compute = list(dict.fromkeys([*REFERENCE_METHODS, *candidate_methods, best_single_method]))
        score_cache: dict[str, np.ndarray] = {}
        topk_cache: dict[str, np.ndarray] = {}
        for method_name in methods_to_compute:
            if method_name not in registry:
                continue
            score_cache[method_name] = np.asarray(registry[method_name](dataset.counts, dataset.batches, current_top_k), dtype=np.float64)
            topk_cache[method_name] = topk_indices(score_cache[method_name], current_top_k)

        anchor_scores = score_cache[SAFE_ANCHOR_METHOD]
        anchor_topk = topk_cache[SAFE_ANCHOR_METHOD]
        best_scores = score_cache.get(best_single_method, anchor_scores)
        best_topk = topk_cache.get(best_single_method, anchor_topk)

        anchor_overlap_to_best = jaccard(anchor_topk, best_topk)
        anchor_corr_to_best = spearman_correlation(anchor_scores, best_scores)
        anchor_corr_map = {
            method_name: spearman_correlation(anchor_scores, score_cache[method_name])
            for method_name in score_cache
        }
        group_name = "positive_headroom" if dataset_name in POSITIVE_SMOKE_DATASETS else "atlas_control"

        for method_name in candidate_methods:
            candidate_scores = score_cache[method_name]
            candidate_topk = topk_cache[method_name]
            row = {
                "row_type": "analysis_dataset",
                "dataset": dataset_name,
                "group_name": group_name,
                "method": method_name,
                "best_single_method": best_single_method,
                "rank_corr_to_anchor": spearman_correlation(anchor_scores, candidate_scores),
                "topk_overlap_to_anchor": jaccard(candidate_topk, anchor_topk),
                "topk_shift_vs_anchor": 1.0 - jaccard(candidate_topk, anchor_topk),
                "topk_overlap_to_best_single": jaccard(candidate_topk, best_topk),
                "delta_overlap_to_best_single_vs_anchor": jaccard(candidate_topk, best_topk) - anchor_overlap_to_best,
                "rank_corr_to_best_single": spearman_correlation(candidate_scores, best_scores),
                "delta_rank_corr_to_best_single_vs_anchor": spearman_correlation(candidate_scores, best_scores) - anchor_corr_to_best,
                "score_dispersion_ratio_vs_anchor": safe_ratio(np.std(candidate_scores), np.std(anchor_scores)),
            }
            for ref_name in (
                "variance",
                "multinomial_deviance_hvg",
                "triku_hvg",
                "scanpy_seurat_v3_hvg",
                "scanpy_cell_ranger_hvg",
                "scran_model_gene_var_hvg",
            ):
                if ref_name in score_cache:
                    row[f"delta_corr_to_{ref_name}_vs_anchor"] = (
                        spearman_correlation(candidate_scores, score_cache[ref_name]) - anchor_corr_map[ref_name]
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def summarize_analysis_smoke(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for method_name, group in df.groupby("method", sort=False):
        positive_group = group[group["group_name"] == "positive_headroom"]
        control_group = group[group["group_name"] == "atlas_control"]
        positive_shift = float(positive_group["topk_shift_vs_anchor"].mean())
        control_shift = float(control_group["topk_shift_vs_anchor"].mean())
        positive_overlap_pull = float(positive_group["delta_overlap_to_best_single_vs_anchor"].mean())
        control_overlap_pull = float(control_group["delta_overlap_to_best_single_vs_anchor"].mean())
        positive_corr_pull = float(positive_group["delta_rank_corr_to_best_single_vs_anchor"].mean())
        control_corr_pull = float(control_group["delta_rank_corr_to_best_single_vs_anchor"].mean())
        conditions = {
            "targeted_shift_gap": (positive_shift - control_shift) >= 0.03,
            "winner_overlap_pull_positive": positive_overlap_pull > 0.0,
            "winner_overlap_pull_gap": (positive_overlap_pull - control_overlap_pull) >= 0.015,
            "winner_corr_pull_gap": (positive_corr_pull - control_corr_pull) >= 0.01,
            "control_guard": control_overlap_pull >= -0.02,
        }
        condition_count = int(sum(bool(value) for value in conditions.values()))
        summary_row = {
            "row_type": "analysis_summary",
            "method": method_name,
            "positive_shift_vs_anchor": positive_shift,
            "control_shift_vs_anchor": control_shift,
            "positive_minus_control_shift": positive_shift - control_shift,
            "positive_overlap_pull_vs_anchor": positive_overlap_pull,
            "control_overlap_pull_vs_anchor": control_overlap_pull,
            "overlap_pull_gap": positive_overlap_pull - control_overlap_pull,
            "positive_corr_pull_vs_anchor": positive_corr_pull,
            "control_corr_pull_vs_anchor": control_corr_pull,
            "corr_pull_gap": positive_corr_pull - control_corr_pull,
            "mean_rank_corr_to_anchor": float(group["rank_corr_to_anchor"].mean()),
            "mean_score_dispersion_ratio_vs_anchor": float(group["score_dispersion_ratio_vs_anchor"].mean()),
            "analysis_condition_count": condition_count,
            "analysis_pass": bool(condition_count >= 4),
        }
        for condition_name, condition_value in conditions.items():
            summary_row[f"condition_{condition_name}"] = bool(condition_value)
        for ref_name in (
            "variance",
            "multinomial_deviance_hvg",
            "triku_hvg",
            "scanpy_seurat_v3_hvg",
            "scanpy_cell_ranger_hvg",
            "scran_model_gene_var_hvg",
        ):
            column = f"delta_corr_to_{ref_name}_vs_anchor"
            if column in group.columns:
                summary_row[f"positive_mean_{column}"] = float(positive_group[column].mean())
                summary_row[f"control_mean_{column}"] = float(control_group[column].mean())
        rows.append(summary_row)
    return pd.DataFrame(rows).sort_values(
        ["analysis_pass", "analysis_condition_count", "overlap_pull_gap", "positive_minus_control_shift"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)


def run_dataset_benchmark(
    *,
    dataset_cache: pkg.DatasetCache,
    dataset_names: tuple[str, ...],
    method_names: tuple[str, ...],
    gate_model_path: str,
    top_k: int,
    seed: int,
    bootstrap_samples: int,
    refine_epochs: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        dataset = dataset_cache.get(dataset_name, seed)
        spec = dataset_cache.resources.spec_map[dataset_name]
        rows.extend(
            rr1.run_round1_dataset_benchmark(
                dataset=dataset,
                dataset_id=spec.dataset_id,
                spec=spec,
                method_names=method_names,
                gate_model_path=gate_model_path,
                refine_epochs=refine_epochs,
                top_k=top_k,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
            )
        )
    return rr1.add_run_level_scores(pd.DataFrame(rows))


def run_biology_proxy(
    *,
    dataset_cache: pkg.DatasetCache,
    dataset_names: tuple[str, ...],
    method_names: tuple[str, ...],
    gate_model_path: str,
    top_k: int,
    seed: int,
    refine_epochs: int,
) -> pd.DataFrame:
    registry = build_default_method_registry(
        top_k=top_k,
        refine_epochs=refine_epochs,
        random_state=seed,
        gate_model_path=gate_model_path,
    )
    rows: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        dataset = dataset_cache.get(dataset_name, seed)
        if dataset.labels is None:
            continue
        labels = np.asarray(dataset.labels, dtype=object)
        if np.unique(labels).size < 2:
            continue
        markers, class_weights = pkg.compute_one_vs_rest_markers(
            counts=dataset.counts,
            labels=labels,
            top_n=50,
        )
        current_top_k = min(top_k, dataset.counts.shape[1])
        for method_name in method_names:
            scores = np.asarray(registry[method_name](dataset.counts, dataset.batches, current_top_k), dtype=np.float64)
            selected = topk_indices(scores, current_top_k)
            _, weighted_marker, rare_marker = pkg.marker_recovery(
                selected=selected,
                marker_sets=markers,
                class_weights=class_weights,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method_name,
                    "weighted_marker_recall_at_50": float(weighted_marker),
                    "rare_marker_recall_at_50": float(rare_marker),
                }
            )
    return pd.DataFrame(rows)


def summarize_benchmark_smoke(raw_df: pd.DataFrame) -> pd.DataFrame:
    failure_df = pd.read_csv(ROOT / "artifacts_topconf_selector_round2" / "failure_taxonomy.csv").set_index("dataset")
    headroom_df = pd.read_csv(ROOT / "artifacts_next_direction" / "anchor_headroom_tables.csv")
    headroom_df = headroom_df[headroom_df["row_type"] == "dataset"].set_index("dataset")
    rows: list[dict[str, object]] = []
    for method_name in sorted(set(raw_df["method"].astype(str).tolist()) - {SAFE_ANCHOR_METHOD}):
        delta_df = build_delta_rows(
            raw_df=raw_df,
            method=method_name,
            anchor_method=SAFE_ANCHOR_METHOD,
            failure_df=failure_df,
            headroom_df=headroom_df,
        )
        positive_mean_delta = float(delta_df[delta_df["dataset"].isin(POSITIVE_SMOKE_DATASETS)]["overall_delta_vs_anchor"].mean())
        atlas_mean_delta = float(delta_df[delta_df["dataset"].isin(ATLAS_CONTROL_DATASETS)]["overall_delta_vs_anchor"].mean())
        biology_mean_delta = float(delta_df["biology_delta_vs_anchor"].mean())
        runtime_ratio = float(delta_df["runtime_ratio_vs_anchor"].mean())
        cluster_delta = float(delta_df["cluster_silhouette_delta_vs_anchor"].mean())
        stability_delta = float(delta_df["stability_delta_vs_anchor"].mean())
        rows.append(
            {
                "row_type": "benchmark_summary",
                **summarize_delta_table(
                    delta_df=delta_df,
                    method=method_name,
                    positive_headroom_datasets=list(POSITIVE_SMOKE_DATASETS),
                    atlas_datasets=list(ATLAS_CONTROL_DATASETS),
                ),
                "smoke_pass": bool(
                    positive_mean_delta > 0.0
                    and atlas_mean_delta >= -0.05
                    and biology_mean_delta >= -0.02
                    and runtime_ratio <= 1.50
                    and (cluster_delta + stability_delta) > 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["smoke_pass", "positive_headroom_mean_delta", "overall_mean_delta"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_delta_rows(
    *,
    raw_df: pd.DataFrame,
    method: str,
    anchor_method: str,
    failure_df: pd.DataFrame,
    headroom_df: pd.DataFrame,
) -> pd.DataFrame:
    method_df = raw_df[raw_df["method"] == method].copy().set_index("dataset")
    anchor_df = raw_df[raw_df["method"] == anchor_method].copy().set_index("dataset")
    join_cols = [
        "overall_score",
        "cluster_silhouette",
        "stability",
        "neighbor_preservation",
        "ari",
        "nmi",
        "label_silhouette",
        "runtime_sec",
        "weighted_marker_recall_at_50",
    ]
    merged = method_df.join(anchor_df[join_cols].add_prefix("anchor_"), how="inner")
    merged = merged.join(failure_df[["regime"]], how="left")
    merged = merged.join(headroom_df[["headroom_vs_best_single"]], how="left")
    merged = merged.reset_index()
    merged["overall_delta_vs_anchor"] = merged["overall_score"] - merged["anchor_overall_score"]
    merged["cluster_silhouette_delta_vs_anchor"] = merged["cluster_silhouette"] - merged["anchor_cluster_silhouette"]
    merged["stability_delta_vs_anchor"] = merged["stability"] - merged["anchor_stability"]
    merged["neighbor_preservation_delta_vs_anchor"] = merged["neighbor_preservation"] - merged["anchor_neighbor_preservation"]
    merged["ari_delta_vs_anchor"] = merged["ari"] - merged["anchor_ari"]
    merged["nmi_delta_vs_anchor"] = merged["nmi"] - merged["anchor_nmi"]
    merged["label_silhouette_delta_vs_anchor"] = merged["label_silhouette"] - merged["anchor_label_silhouette"]
    merged["biology_delta_vs_anchor"] = merged["weighted_marker_recall_at_50"] - merged["anchor_weighted_marker_recall_at_50"]
    merged["runtime_ratio_vs_anchor"] = merged["runtime_sec"] / np.maximum(merged["anchor_runtime_sec"], 1e-8)
    return merged


def summarize_delta_table(
    *,
    delta_df: pd.DataFrame,
    method: str,
    positive_headroom_datasets: list[str],
    atlas_datasets: list[str],
) -> dict[str, object]:
    return {
        "method": method,
        "dataset_count": int(len(delta_df)),
        "overall_mean_delta": float(delta_df["overall_delta_vs_anchor"].mean()),
        "positive_headroom_mean_delta": float(delta_df[delta_df["dataset"].isin(positive_headroom_datasets)]["overall_delta_vs_anchor"].mean()),
        "atlas_like_mean_delta": float(delta_df[delta_df["dataset"].isin(atlas_datasets)]["overall_delta_vs_anchor"].mean()),
        "mean_cluster_delta": float(delta_df["cluster_silhouette_delta_vs_anchor"].mean()),
        "mean_stability_delta": float(delta_df["stability_delta_vs_anchor"].mean()),
        "mean_neighbor_delta": float(delta_df["neighbor_preservation_delta_vs_anchor"].mean()),
        "mean_biology_delta": float(delta_df["biology_delta_vs_anchor"].mean()),
        "mean_runtime_ratio_vs_anchor": float(delta_df["runtime_ratio_vs_anchor"].mean()),
    }


def write_smoke_results(
    *,
    output_dir: Path,
    analysis_df: pd.DataFrame,
    analysis_summary: pd.DataFrame,
    benchmark_raw: pd.DataFrame,
    benchmark_summary: pd.DataFrame,
    evidence: dict[str, pd.DataFrame],
) -> None:
    result_frames = [analysis_df, analysis_summary]
    if not benchmark_raw.empty:
        result_frames.append(benchmark_raw.assign(row_type="benchmark_dataset"))
    if not benchmark_summary.empty:
        result_frames.append(benchmark_summary)
    result_df = pd.concat(result_frames, ignore_index=True, sort=False)
    result_df.to_csv(output_dir / "smoke_results.csv", index=False)

    lines = [
        "# Smoke Screening",
        "",
        "## Analysis-Level Smoke",
    ]
    for row in analysis_summary.itertuples(index=False):
        lines.append(
            f"- `{row.method}`: pass={bool(row.analysis_pass)} condition_count={int(row.analysis_condition_count)} "
            f"shift_gap={float(row.positive_minus_control_shift):.4f} overlap_pull_gap={float(row.overlap_pull_gap):.4f} "
            f"corr_pull_gap={float(row.corr_pull_gap):.4f} control_pull={float(row.control_overlap_pull_vs_anchor):.4f}."
        )
    lines.extend(["", "## Benchmark Smoke"])
    if benchmark_summary.empty:
        lines.append("- No candidate cleared the analysis-level screen, so no benchmark smoke was run.")
    else:
        for row in benchmark_summary.itertuples(index=False):
            lines.append(
                f"- `{row.method}`: pass={bool(row.smoke_pass)} positive_mean_delta={float(row.positive_headroom_mean_delta):.4f} "
                f"atlas_mean_delta={float(row.atlas_like_mean_delta):.4f} biology_delta={float(row.mean_biology_delta):.4f} "
                f"cluster_delta={float(row.mean_cluster_delta):.4f} stability_delta={float(row.mean_stability_delta):.4f} "
                f"runtime_ratio={float(row.mean_runtime_ratio_vs_anchor):.4f}."
            )
    lines.extend(
        [
            "",
            "## Reference Pull Patterns",
        ]
    )
    headroom_lookup = evidence["headroom_df"].set_index("dataset")
    for method_name, group in analysis_df.groupby("method", sort=False):
        positive_group = group[group["group_name"] == "positive_headroom"]
        if positive_group.empty:
            continue
        strongest_row = positive_group.sort_values("delta_overlap_to_best_single_vs_anchor", ascending=False).iloc[0]
        weakest_row = positive_group.sort_values("delta_overlap_to_best_single_vs_anchor").iloc[0]
        lines.append(
            f"- `{method_name}` strongest positive pull: {strongest_row['dataset']} toward `{headroom_lookup.loc[strongest_row['dataset'], 'best_single_expert']}` "
            f"(delta_overlap={float(strongest_row['delta_overlap_to_best_single_vs_anchor']):.4f}); weakest positive pull: "
            f"{weakest_row['dataset']} (delta_overlap={float(weakest_row['delta_overlap_to_best_single_vs_anchor']):.4f})."
        )
    (output_dir / "smoke_screening.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_final_recommendation(
    *,
    output_dir: Path,
    top5_df: pd.DataFrame,
    analysis_summary: pd.DataFrame,
    benchmark_summary: pd.DataFrame,
) -> None:
    top5_names = top5_df["idea_name"].astype(str).tolist()
    lines = [
        "# Final Recommendation",
        "",
        "## Direct Answers",
        "1. Most promising top-5 after broad search:",
    ]
    for row in top5_df.itertuples(index=False):
        lines.append(f"- `{row.idea_name}` ({row.family}) because {row.why_top5}")
    lines.extend(
        [
            "",
            "2. Why each had a credible path:",
        ]
    )
    for row in top5_df.itertuples(index=False):
        lines.append(f"- `{row.idea_name}` targets {row.structural_headroom_target}.")

    lines.extend(
        [
            "",
            "3. Which candidates failed in smoke and why:",
        ]
    )
    if benchmark_summary.empty:
        for row in analysis_summary.itertuples(index=False):
            if bool(row.analysis_pass):
                lines.append(
                    f"- `{row.method}` cleared analysis but no benchmark summary was produced; treat as unresolved and do not unlock DFS."
                )
            else:
                failed_conditions = [
                    condition_name
                    for condition_name in ANALYSIS_CONDITION_NAMES
                    if not bool(getattr(row, f'condition_{condition_name}'))
                ]
                lines.append(f"- `{row.method}` failed analysis because it missed: {', '.join(failed_conditions)}.")
    else:
        benchmark_lookup = benchmark_summary.set_index("method")
        for method_name in top5_names:
            if method_name in benchmark_lookup.index:
                row = benchmark_lookup.loc[method_name]
                if bool(row["smoke_pass"]):
                    lines.append(
                        f"- `{method_name}` passed benchmark smoke with positive_headroom_mean_delta={float(row['positive_headroom_mean_delta']):.4f}."
                    )
                else:
                    lines.append(
                        f"- `{method_name}` failed benchmark smoke: positive_mean_delta={float(row['positive_headroom_mean_delta']):.4f}, "
                        f"atlas_mean_delta={float(row['atlas_like_mean_delta']):.4f}, biology_delta={float(row['mean_biology_delta']):.4f}, "
                        f"runtime_ratio={float(row['mean_runtime_ratio_vs_anchor']):.4f}."
                    )
            else:
                row = analysis_summary[analysis_summary["method"] == method_name].iloc[0]
                failed_conditions = [
                    condition_name
                    for condition_name in ANALYSIS_CONDITION_NAMES
                    if not bool(row[f"condition_{condition_name}"])
                ]
                lines.append(f"- `{method_name}` failed analysis because it missed: {', '.join(failed_conditions)}.")

    passing_benchmark = benchmark_summary[benchmark_summary["smoke_pass"] == True].copy() if not benchmark_summary.empty else pd.DataFrame()  # noqa: E712
    lines.extend(["", "4. Is there a unique direction worth continuing?"])
    if len(passing_benchmark) == 1:
        best_row = passing_benchmark.iloc[0]
        lines.append(
            f"- Yes. `{best_row['method']}` is the only benchmark-passing candidate and is the only justified continuation target."
        )
        lines.append("- Minimal DFS plan: ablate the new term against the anchor on the full 12-dataset benchmark and verify that gains remain concentrated in clustering/stability.")
    elif len(passing_benchmark) > 1:
        best_row = passing_benchmark.sort_values(
            ["positive_headroom_mean_delta", "overall_mean_delta", "mean_runtime_ratio_vs_anchor"],
            ascending=[False, False, True],
        ).iloc[0]
        lines.append(
            f"- Not cleanly unique. `{best_row['method']}` is currently best, but more than one candidate passed and the benchmark does not isolate one uncontested mainline."
        )
    else:
        strongest_analysis = analysis_summary.sort_values(
            ["analysis_pass", "analysis_condition_count", "overlap_pull_gap", "positive_minus_control_shift"],
            ascending=[False, False, False, False],
        ).iloc[0]
        lines.append(
            f"- No. The strongest analysis-stage candidate was `{strongest_analysis['method']}`, but it did not justify a unique continuation path."
        )

    lines.extend(["", "5. If not, should the benchmark-level search pause?"])
    if len(passing_benchmark) == 0:
        lines.append(
            "- Yes. The current benchmark does not show a sufficiently credible new mainline beyond `adaptive_hybrid_hvg`, so the right move is to pause rather than create more project bulk."
        )
    else:
        lines.append("- Not yet. At least one candidate passed the smoke screen, so a narrow DFS branch is justified.")
    lines.append("")
    (output_dir / "final_recommendation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    return np.argsort(np.asarray(scores, dtype=np.float64))[-int(top_k):]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_set = set(np.asarray(a).tolist())
    b_set = set(np.asarray(b).tolist())
    return float(len(a_set & b_set) / max(len(a_set | b_set), 1))


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_rank = pd.Series(np.asarray(a, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)
    b_rank = pd.Series(np.asarray(b, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(a_rank) < 1e-8 or np.std(b_rank) < 1e-8:
        return 0.0
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-8:
        return 0.0
    return float(numerator / denominator)


if __name__ == "__main__":
    main()
