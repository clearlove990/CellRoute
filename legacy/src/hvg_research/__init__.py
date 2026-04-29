from __future__ import annotations

from .data import SCRNADataset, SCRNAInputSpec, discover_scrna_input_specs, load_scrna_dataset, sanitize_dataset, subsample_dataset

__all__ = [
    "SCRNADataset",
    "SCRNAInputSpec",
    "discover_scrna_input_specs",
    "load_scrna_dataset",
    "sanitize_dataset",
    "subsample_dataset",
]


try:
    from .baselines import (
        score_analytic_pearson_residual_hvg,
        score_fano,
        score_mean_var_residual,
        score_multinomial_deviance_hvg,
        score_seurat_v3_like_hvg,
        score_variance,
    )
    from .eval import evaluate_real_selection, evaluate_selection
    from .gate_learning import GateLearningConfig, predict_gate_weights, train_gate_model
    from .holdout_selector import (
        RiskControlledSelectorConfig,
        RiskControlledSelectorPolicy,
        SelectorDecision,
        choose_torch_device,
    )
    from .methods import build_default_method_registry
    from .official_baselines import (
        score_scanpy_cell_ranger_hvg,
        score_scanpy_seurat_v3_hvg,
        score_scran_model_gene_var_hvg,
        score_seurat_r_vst_hvg,
        score_triku_hvg,
    )
    from .refine_moe_hvg import RefineMoEHVGSelector
    from .synthetic import SyntheticSCRNAData, generate_synthetic_scrna
except ModuleNotFoundError as exc:
    if exc.name != "torch":
        raise
else:
    __all__.extend(
        [
            "GateLearningConfig",
            "RefineMoEHVGSelector",
            "RiskControlledSelectorConfig",
            "RiskControlledSelectorPolicy",
            "SyntheticSCRNAData",
            "SelectorDecision",
            "build_default_method_registry",
            "choose_torch_device",
            "evaluate_real_selection",
            "evaluate_selection",
            "generate_synthetic_scrna",
            "predict_gate_weights",
            "score_fano",
            "score_analytic_pearson_residual_hvg",
            "score_mean_var_residual",
            "score_multinomial_deviance_hvg",
            "score_scanpy_cell_ranger_hvg",
            "score_scanpy_seurat_v3_hvg",
            "score_scran_model_gene_var_hvg",
            "score_seurat_r_vst_hvg",
            "score_seurat_v3_like_hvg",
            "score_triku_hvg",
            "score_variance",
            "train_gate_model",
        ]
    )
