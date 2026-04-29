from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for candidate in (str(SRC), str(SCRIPTS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hvg_research import build_default_method_registry

import build_topconf_selector_round2_package as pkg
import run_real_inputs_round1 as rr1


DEFAULT_OUTPUT_DIR = ROOT / "artifacts_sigma_analysis_gate"
DEFAULT_ANCHOR_METHOD = "adaptive_hybrid_hvg"
DEFAULT_CANDIDATE_METHOD = "sigma_hvg"

POSITIVE_HEADROOM_DATASETS = (
    "GBM_sd",
    "cellxgene_human_kidney_nonpt",
    "paul15",
    "cellxgene_immune_five_donors",
)
ATLAS_CONTROL_DATASETS = (
    "FBM_cite",
    "homo_tissue",
    "mus_tissue",
    "E-MTAB-4388",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the SIGMA analysis gate and optional minimal benchmark.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--real-data-root", type=str, default=str(ROOT / "data" / "real_inputs"))
    parser.add_argument("--anchor-method", type=str, default=DEFAULT_ANCHOR_METHOD)
    parser.add_argument("--candidate-method", type=str, default=DEFAULT_CANDIDATE_METHOD)
    parser.add_argument("--gate-model-path", type=str, default=rr1.DEFAULT_GATE_MODEL_PATH)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--refine-epochs", type=int, default=6)
    parser.add_argument("--bootstrap-samples", type=int, default=1)
    return parser.parse_args()


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float) and np.isnan(value):
        return "NA"
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def topk_indices(scores: np.ndarray, k: int) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float64).ravel()
    if scores.size == 0:
        return np.asarray([], dtype=np.int64)
    k = max(1, min(int(k), int(scores.size)))
    idx = np.argpartition(scores, -k)[-k:]
    return np.asarray(idx, dtype=np.int64)


def jaccard(left: np.ndarray, right: np.ndarray) -> float:
    left_set = set(np.asarray(left, dtype=np.int64).tolist())
    right_set = set(np.asarray(right, dtype=np.int64).tolist())
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    if not union:
        return 1.0
    return float(len(left_set & right_set) / len(union))


def spearman_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_arr = np.asarray(left, dtype=np.float64).ravel()
    right_arr = np.asarray(right, dtype=np.float64).ravel()
    if left_arr.size != right_arr.size or left_arr.size < 2:
        return 0.0
    left_rank = pd.Series(left_arr).rank(method="average").to_numpy(dtype=np.float64)
    right_rank = pd.Series(right_arr).rank(method="average").to_numpy(dtype=np.float64)
    corr = np.corrcoef(left_rank, right_rank)[0, 1]
    if np.isnan(corr):
        return 0.0
    return float(corr)


def mean_or_nan(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.mean())


def load_headroom_table() -> pd.DataFrame:
    path = ROOT / "artifacts_next_direction" / "anchor_headroom_tables.csv"
    df = pd.read_csv(path)
    return df[df["row_type"] == "dataset"].copy().reset_index(drop=True)


def load_dataset_resources(real_data_root: Path) -> pkg.DatasetResources:
    manifest_df = pd.read_csv(ROOT / "artifacts_topconf_selector_round2" / "dataset_manifest.csv")
    return pkg.load_dataset_resources(real_data_root=real_data_root, manifest_df=manifest_df)


def build_dataset_split() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name in POSITIVE_HEADROOM_DATASETS:
        rows.append({"dataset": dataset_name, "role": "positive_headroom"})
    for dataset_name in ATLAS_CONTROL_DATASETS:
        rows.append({"dataset": dataset_name, "role": "atlas_control"})
    return pd.DataFrame(rows)


def build_analysis_rows(
    *,
    dataset_cache: pkg.DatasetCache,
    headroom_df: pd.DataFrame,
    top_k: int,
    seed: int,
    refine_epochs: int,
    gate_model_path: str | None,
    anchor_method: str,
    candidate_method: str,
) -> pd.DataFrame:
    registry = build_default_method_registry(
        top_k=top_k,
        refine_epochs=refine_epochs,
        random_state=seed,
        gate_model_path=gate_model_path,
    )
    missing_methods = [name for name in (anchor_method, candidate_method) if name not in registry]
    if missing_methods:
        raise KeyError(f"Method(s) not available in registry: {missing_methods}")

    headroom_lookup = headroom_df.set_index("dataset")
    rows: list[dict[str, object]] = []
    analysis_datasets = (*POSITIVE_HEADROOM_DATASETS, *ATLAS_CONTROL_DATASETS)
    for dataset_name in analysis_datasets:
        dataset = dataset_cache.get(dataset_name, seed)
        current_top_k = min(int(top_k), int(dataset.counts.shape[1]))
        best_single_method = str(headroom_lookup.loc[dataset_name, "best_single_expert"])
        methods_to_compute = tuple(dict.fromkeys([anchor_method, candidate_method, best_single_method]))
        score_cache: dict[str, np.ndarray] = {}
        topk_cache: dict[str, np.ndarray] = {}
        for method_name in methods_to_compute:
            score_cache[method_name] = np.asarray(
                registry[method_name](dataset.counts, dataset.batches, current_top_k),
                dtype=np.float64,
            )
            topk_cache[method_name] = topk_indices(score_cache[method_name], current_top_k)

        anchor_scores = score_cache[anchor_method]
        anchor_topk = topk_cache[anchor_method]
        best_scores = score_cache[best_single_method]
        best_topk = topk_cache[best_single_method]
        group_name = "positive_headroom" if dataset_name in POSITIVE_HEADROOM_DATASETS else "atlas_control"
        anchor_overlap_to_best = jaccard(anchor_topk, best_topk)
        anchor_corr_to_best = spearman_correlation(anchor_scores, best_scores)

        candidate_scores = score_cache[candidate_method]
        candidate_topk = topk_cache[candidate_method]
        rows.append(
            {
                "dataset": dataset_name,
                "group_name": group_name,
                "method": candidate_method,
                "anchor_method": anchor_method,
                "best_single_method": best_single_method,
                "rank_corr_to_anchor": spearman_correlation(candidate_scores, anchor_scores),
                "topk_overlap_to_anchor": jaccard(candidate_topk, anchor_topk),
                "topk_shift_vs_anchor": 1.0 - jaccard(candidate_topk, anchor_topk),
                "topk_overlap_to_best_single": jaccard(candidate_topk, best_topk),
                "delta_overlap_to_best_single_vs_anchor": jaccard(candidate_topk, best_topk) - anchor_overlap_to_best,
                "rank_corr_to_best_single": spearman_correlation(candidate_scores, best_scores),
                "delta_rank_corr_to_best_single_vs_anchor": spearman_correlation(candidate_scores, best_scores)
                - anchor_corr_to_best,
                "score_dispersion_ratio_vs_anchor": float(
                    np.std(candidate_scores) / max(np.std(anchor_scores), 1e-8)
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_analysis(df: pd.DataFrame, *, candidate_method: str) -> pd.DataFrame:
    positive_group = df[df["group_name"] == "positive_headroom"].copy()
    control_group = df[df["group_name"] == "atlas_control"].copy()
    positive_shift = mean_or_nan(positive_group["topk_shift_vs_anchor"])
    control_shift = mean_or_nan(control_group["topk_shift_vs_anchor"])
    positive_overlap_pull = mean_or_nan(positive_group["delta_overlap_to_best_single_vs_anchor"])
    control_overlap_pull = mean_or_nan(control_group["delta_overlap_to_best_single_vs_anchor"])
    positive_corr_pull = mean_or_nan(positive_group["delta_rank_corr_to_best_single_vs_anchor"])
    control_corr_pull = mean_or_nan(control_group["delta_rank_corr_to_best_single_vs_anchor"])
    conditions = {
        "targeted_shift_gap": (positive_shift - control_shift) >= 0.03,
        "winner_overlap_pull_positive": positive_overlap_pull > 0.0,
        "winner_overlap_pull_gap": (positive_overlap_pull - control_overlap_pull) >= 0.015,
        "winner_corr_pull_gap": (positive_corr_pull - control_corr_pull) >= 0.01,
        "control_guard": control_overlap_pull >= -0.02,
    }
    summary_row = {
        "method": candidate_method,
        "positive_shift_vs_anchor": positive_shift,
        "control_shift_vs_anchor": control_shift,
        "positive_minus_control_shift": positive_shift - control_shift,
        "positive_overlap_pull_vs_anchor": positive_overlap_pull,
        "control_overlap_pull_vs_anchor": control_overlap_pull,
        "overlap_pull_gap": positive_overlap_pull - control_overlap_pull,
        "positive_corr_pull_vs_anchor": positive_corr_pull,
        "control_corr_pull_vs_anchor": control_corr_pull,
        "corr_pull_gap": positive_corr_pull - control_corr_pull,
        "mean_rank_corr_to_anchor": float(df["rank_corr_to_anchor"].mean()),
        "mean_score_dispersion_ratio_vs_anchor": float(df["score_dispersion_ratio_vs_anchor"].mean()),
        "analysis_condition_count": int(sum(bool(value) for value in conditions.values())),
    }
    summary_row["analysis_pass"] = bool(summary_row["analysis_condition_count"] >= 4)
    for condition_name, value in conditions.items():
        summary_row[f"condition_{condition_name}"] = bool(value)
    return pd.DataFrame([summary_row])


def run_minimal_benchmark(
    *,
    dataset_cache: pkg.DatasetCache,
    resources: pkg.DatasetResources,
    top_k: int,
    seed: int,
    refine_epochs: int,
    bootstrap_samples: int,
    gate_model_path: str | None,
    anchor_method: str,
    candidate_method: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    method_names = (anchor_method, candidate_method)
    dataset_names = (*POSITIVE_HEADROOM_DATASETS, *ATLAS_CONTROL_DATASETS)
    rows: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        dataset = dataset_cache.get(dataset_name, seed)
        spec = resources.spec_map[dataset_name]
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
    raw_df = rr1.add_run_level_scores(pd.DataFrame(rows))
    summary_df = rr1.summarize_by_keys(raw_df, keys=["dataset", "dataset_id", "method"])
    summary_df = rr1.rank_within_group(summary_df, group_cols=["dataset", "dataset_id"], rank_col="dataset_rank")

    candidate_df = summary_df[summary_df["method"] == candidate_method].copy().set_index("dataset")
    anchor_df = summary_df[summary_df["method"] == anchor_method].copy().set_index("dataset")
    join_cols = ["overall_score", "cluster_silhouette", "stability", "neighbor_preservation", "runtime_sec"]
    optional_biology_col = "weighted_marker_recall_at_50"
    if optional_biology_col in anchor_df.columns and optional_biology_col in candidate_df.columns:
        join_cols.append(optional_biology_col)
    joined = candidate_df.join(anchor_df[join_cols].add_prefix("anchor_"), how="inner").reset_index()
    joined["overall_delta_vs_anchor"] = joined["overall_score"] - joined["anchor_overall_score"]
    joined["cluster_delta_vs_anchor"] = joined["cluster_silhouette"] - joined["anchor_cluster_silhouette"]
    joined["stability_delta_vs_anchor"] = joined["stability"] - joined["anchor_stability"]
    joined["neighbor_delta_vs_anchor"] = joined["neighbor_preservation"] - joined["anchor_neighbor_preservation"]
    joined["runtime_ratio_vs_anchor"] = joined["runtime_sec"] / np.maximum(joined["anchor_runtime_sec"], 1e-8)
    if optional_biology_col in joined.columns and f"anchor_{optional_biology_col}" in joined.columns:
        joined["biology_delta_vs_anchor"] = joined[optional_biology_col] - joined[f"anchor_{optional_biology_col}"]
        mean_biology_delta = float(joined["biology_delta_vs_anchor"].mean())
        biology_pass = mean_biology_delta >= -0.02
    else:
        joined["biology_delta_vs_anchor"] = np.nan
        mean_biology_delta = float("nan")
        biology_pass = True

    benchmark_row = {
        "method": candidate_method,
        "positive_headroom_mean_delta": float(
            joined[joined["dataset"].isin(POSITIVE_HEADROOM_DATASETS)]["overall_delta_vs_anchor"].mean()
        ),
        "atlas_control_mean_delta": float(
            joined[joined["dataset"].isin(ATLAS_CONTROL_DATASETS)]["overall_delta_vs_anchor"].mean()
        ),
        "mean_cluster_delta": float(joined["cluster_delta_vs_anchor"].mean()),
        "mean_stability_delta": float(joined["stability_delta_vs_anchor"].mean()),
        "mean_neighbor_delta": float(joined["neighbor_delta_vs_anchor"].mean()),
        "mean_biology_delta": mean_biology_delta,
        "mean_runtime_ratio_vs_anchor": float(joined["runtime_ratio_vs_anchor"].mean()),
    }
    stability_available = bool(joined["stability_delta_vs_anchor"].notna().any())
    structure_delta = (
        benchmark_row["mean_cluster_delta"] + benchmark_row["mean_stability_delta"]
        if stability_available
        else benchmark_row["mean_cluster_delta"]
    )
    benchmark_row["benchmark_pass"] = bool(
        benchmark_row["positive_headroom_mean_delta"] > 0.0
        and benchmark_row["atlas_control_mean_delta"] >= -0.05
        and biology_pass
        and benchmark_row["mean_runtime_ratio_vs_anchor"] <= 1.80
        and structure_delta > 0.0
    )
    return raw_df, joined, pd.DataFrame([benchmark_row])


def render_final_status(
    *,
    anchor_method: str,
    candidate_method: str,
    analysis_summary_df: pd.DataFrame,
    benchmark_summary_df: pd.DataFrame | None,
    output_dir: Path,
) -> str:
    analysis_row = analysis_summary_df.iloc[0]
    benchmark_ran = benchmark_summary_df is not None and not benchmark_summary_df.empty
    benchmark_pass = bool(benchmark_summary_df.iloc[0]["benchmark_pass"]) if benchmark_ran else False
    if not bool(analysis_row["analysis_pass"]):
        decision = "no-go for now"
        narrative = "analysis gate failed, so the benchmark stopped before full target/control evaluation."
    elif benchmark_ran and benchmark_pass:
        decision = "go"
        narrative = "analysis gate passed and the minimal target/control benchmark stayed above the requested floor."
    elif benchmark_ran:
        decision = "no-go for now"
        narrative = "analysis gate passed but the minimal target/control benchmark did not clear the benchmark gate."
    else:
        decision = "analysis-only provisional"
        narrative = "analysis gate passed but no benchmark output was produced."

    artifact_paths = sorted(path.name for path in output_dir.iterdir() if path.is_file())
    lines = [
        "# Final Status",
        "",
        "## Summary",
        f"- Anchor method: `{anchor_method}`",
        f"- Candidate method: `{candidate_method}`",
        f"- Positive-headroom datasets: {', '.join(POSITIVE_HEADROOM_DATASETS)}",
        f"- Atlas controls: {', '.join(ATLAS_CONTROL_DATASETS)}",
        f"- Analysis gates passed: {bool(analysis_row['analysis_pass'])}",
        f"- Entered benchmark: {benchmark_ran}",
        f"- Current decision: {decision} ({narrative})",
        "",
        "## Analysis Gate",
        f"- targeted_shift_gap={fmt_float(analysis_row['positive_minus_control_shift'])} pass={bool(analysis_row['condition_targeted_shift_gap'])}",
        f"- winner_overlap_pull_positive={fmt_float(analysis_row['positive_overlap_pull_vs_anchor'])} pass={bool(analysis_row['condition_winner_overlap_pull_positive'])}",
        f"- winner_overlap_pull_gap={fmt_float(analysis_row['overlap_pull_gap'])} pass={bool(analysis_row['condition_winner_overlap_pull_gap'])}",
        f"- winner_corr_pull_gap={fmt_float(analysis_row['corr_pull_gap'])} pass={bool(analysis_row['condition_winner_corr_pull_gap'])}",
        f"- control_guard={fmt_float(analysis_row['control_overlap_pull_vs_anchor'])} pass={bool(analysis_row['condition_control_guard'])}",
    ]
    if benchmark_ran:
        benchmark_row = benchmark_summary_df.iloc[0]
        lines.extend(
            [
                "",
                "## Benchmark",
                f"- positive_headroom_mean_delta={fmt_float(benchmark_row['positive_headroom_mean_delta'])}",
                f"- atlas_control_mean_delta={fmt_float(benchmark_row['atlas_control_mean_delta'])}",
                f"- mean_cluster_delta={fmt_float(benchmark_row['mean_cluster_delta'])}",
                f"- mean_stability_delta={fmt_float(benchmark_row['mean_stability_delta'])}",
                f"- mean_biology_delta={fmt_float(benchmark_row['mean_biology_delta'])}",
                f"- mean_runtime_ratio_vs_anchor={fmt_float(benchmark_row['mean_runtime_ratio_vs_anchor'])}",
                f"- benchmark_pass={bool(benchmark_row['benchmark_pass'])}",
            ]
        )
    lines.extend(
        [
            "",
            "## Artifacts",
        ]
    )
    for name in artifact_paths:
        lines.append(f"- `{name}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    split_df = build_dataset_split()
    split_df.to_csv(output_dir / "dataset_split.csv", index=False)
    rr1.save_json(output_dir / "compute_context.json", rr1.resolve_device_info())

    headroom_df = load_headroom_table()
    resources = load_dataset_resources(Path(args.real_data_root))
    dataset_cache = pkg.DatasetCache(resources)

    analysis_df = build_analysis_rows(
        dataset_cache=dataset_cache,
        headroom_df=headroom_df,
        top_k=args.top_k,
        seed=args.seed,
        refine_epochs=args.refine_epochs,
        gate_model_path=args.gate_model_path,
        anchor_method=args.anchor_method,
        candidate_method=args.candidate_method,
    )
    analysis_summary_df = summarize_analysis(analysis_df, candidate_method=args.candidate_method)
    analysis_df.to_csv(output_dir / "analysis_dataset_metrics.csv", index=False)
    analysis_summary_df.to_csv(output_dir / "analysis_gate_summary.csv", index=False)

    benchmark_summary_df: pd.DataFrame | None = None
    if bool(analysis_summary_df.iloc[0]["analysis_pass"]):
        benchmark_raw_df, benchmark_delta_df, benchmark_summary_df = run_minimal_benchmark(
            dataset_cache=dataset_cache,
            resources=resources,
            top_k=args.top_k,
            seed=args.seed,
            refine_epochs=args.refine_epochs,
            bootstrap_samples=args.bootstrap_samples,
            gate_model_path=args.gate_model_path,
            anchor_method=args.anchor_method,
            candidate_method=args.candidate_method,
        )
        benchmark_raw_df.to_csv(output_dir / "benchmark_candidate_vs_anchor_raw.csv", index=False)
        benchmark_delta_df.to_csv(output_dir / "benchmark_candidate_vs_anchor_dataset_deltas.csv", index=False)
        benchmark_summary_df.to_csv(output_dir / "benchmark_candidate_vs_anchor_summary.csv", index=False)

    (output_dir / "final_status.md").write_text(
        render_final_status(
            anchor_method=args.anchor_method,
            candidate_method=args.candidate_method,
            analysis_summary_df=analysis_summary_df,
            benchmark_summary_df=benchmark_summary_df,
            output_dir=output_dir,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
