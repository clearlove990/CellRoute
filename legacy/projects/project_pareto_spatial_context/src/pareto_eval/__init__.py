from .clean_win import (
    CONTINUOUS_PROFILE,
    DISCRETE_PROFILE,
    ParetoProfile,
    classify_candidate_against_baseline,
    classify_config,
    classify_frame,
    compute_metric_deltas,
    get_runtime_device,
    profile_for_evaluation_mode,
)
from .frontier import (
    extract_pareto_frontier,
    pairwise_dominance_matrix,
    pareto_dominates,
    rank_candidates_by_dominance_depth,
)
from .metrics import (
    CONTINUOUS_REGISTRY,
    DISCRETE_REGISTRY,
    DEFAULT_METRIC_REGISTRIES,
    MetricRegistry,
    MetricSpec,
    build_metric_registry,
    registry_for_evaluation_mode,
)
from .plotting import plot_frontier, plot_metric_heatmap, plot_win_loss_barplot
from .rank_blend import build_rank_blend_sweep, generate_alpha_grid, interpolate_rank_columns, interpolate_values
from .replacement_frontier import (
    extract_replacement_frontier,
    simulate_anchor_replacement_budgets,
    summarize_replacement_frontier,
)
from .reporting import (
    build_candidate_summary,
    build_group_label_summary,
    generate_csv_summary,
    generate_markdown_summary,
    write_markdown_summary,
)

__all__ = [
    "CONTINUOUS_PROFILE",
    "CONTINUOUS_REGISTRY",
    "DEFAULT_METRIC_REGISTRIES",
    "DISCRETE_PROFILE",
    "DISCRETE_REGISTRY",
    "MetricRegistry",
    "MetricSpec",
    "ParetoProfile",
    "build_metric_registry",
    "build_candidate_summary",
    "build_group_label_summary",
    "build_rank_blend_sweep",
    "classify_candidate_against_baseline",
    "classify_config",
    "classify_frame",
    "compute_metric_deltas",
    "extract_pareto_frontier",
    "extract_replacement_frontier",
    "generate_alpha_grid",
    "generate_csv_summary",
    "generate_markdown_summary",
    "get_runtime_device",
    "interpolate_rank_columns",
    "interpolate_values",
    "pairwise_dominance_matrix",
    "pareto_dominates",
    "plot_frontier",
    "plot_metric_heatmap",
    "plot_win_loss_barplot",
    "profile_for_evaluation_mode",
    "rank_candidates_by_dominance_depth",
    "registry_for_evaluation_mode",
    "simulate_anchor_replacement_budgets",
    "summarize_replacement_frontier",
    "write_markdown_summary",
]
