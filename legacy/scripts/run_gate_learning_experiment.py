from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from hvg_research import (
    GateLearningConfig,
    RefineMoEHVGSelector,
    build_default_method_registry,
    evaluate_real_selection,
    evaluate_selection,
    generate_synthetic_scrna,
    load_scrna_dataset,
    subsample_dataset,
    train_gate_model,
)
from hvg_research.eval import timed_call
from hvg_research.methods import (
    FRONTIER_ANCHORLESS_INTERNAL_MODE,
    FRONTIER_ANCHORLESS_METHOD,
    FRONTIER_LITE_INTERNAL_MODE,
    FRONTIER_LITE_METHOD,
    canonicalize_method_name,
    canonicalize_method_names,
)
from hvg_research.refine_moe_hvg import EXPERT_NAMES


FRONTIER_FULL_METHOD = "learnable_gate_bank_pairregret_permissioned_escapecert_frontier"
ESCAPECERT_FULL_METHOD = "learnable_gate_bank_pairregret_permissioned_escapecert"
PERMISSIONED_METHOD = "learnable_gate_bank_pairregret_permissioned"
PAIRREGRET_METHOD = "learnable_gate_bank_pairregret"
BANK_METHOD = "learnable_gate_bank"
MV_RESIDUAL_METHOD = "mv_residual"

EXPERIMENT_PRESETS: dict[str, dict[str, int]] = {
    "smoke": {
        "train_seeds": 1,
        "eval_seeds": 1,
        "real_top_k": 40,
    },
    "full": {
        "train_seeds": 3,
        "eval_seeds": 2,
        "real_top_k": 80,
    },
}

FOCUS_POSITIVE_METRICS = (
    "ari",
    "nmi",
    "label_silhouette",
    "batch_mixing",
    "neighbor_preservation",
    "informative_recall",
    "stability",
)
FOCUS_NEGATIVE_METRICS = (
    "batch_gene_fraction",
    "runtime_sec",
)
REAL_FOCUS_POSITIVE_METRICS = (
    "ari",
    "nmi",
    "label_silhouette",
    "batch_mixing",
    "neighbor_preservation",
    "cluster_silhouette",
    "stability",
)
REAL_FOCUS_NEGATIVE_METRICS = ("runtime_sec",)


def extract_method_metadata(method_fn: object) -> dict[str, float | str]:
    metadata: dict[str, float | str] = {}
    gate_source = getattr(method_fn, "last_gate_source", None)
    if gate_source is not None:
        metadata["gate_source"] = str(gate_source)

    dataset_stats = getattr(method_fn, "last_dataset_stats", None)
    if isinstance(dataset_stats, dict):
        for key, value in dataset_stats.items():
            try:
                metadata[f"stat_{key}"] = float(value)
            except (TypeError, ValueError):
                continue

    gate_metadata = getattr(method_fn, "last_gate_metadata", None)
    if isinstance(gate_metadata, dict):
        for key, value in gate_metadata.items():
            try:
                metadata[key] = float(value)
            except (TypeError, ValueError):
                metadata[key] = str(value)
    return metadata


def explicit_arg_names(argv: list[str]) -> set[str]:
    names: set[str] = set()
    idx = 0
    while idx < len(argv):
        token = argv[idx]
        if token.startswith("--"):
            option = token[2:]
            if "=" in option:
                option = option.split("=", maxsplit=1)[0]
            names.add(option.replace("-", "_"))
        idx += 1
    return names


def apply_experiment_preset(args: argparse.Namespace, *, argv: list[str]) -> None:
    preset_name = getattr(args, "preset", None)
    if not preset_name:
        return
    preset = EXPERIMENT_PRESETS[preset_name]
    explicit_names = explicit_arg_names(argv)
    for key, value in preset.items():
        current_value = getattr(args, key)
        if key in explicit_names:
            continue
        if current_value is None or current_value == parser_default_value(key):
            setattr(args, key, value)


def parser_default_value(arg_name: str) -> object:
    defaults: dict[str, object] = {
        "train_seeds": 6,
        "eval_seeds": 3,
        "real_top_k": None,
    }
    return defaults.get(arg_name)


def maybe_filter_methods(df: pd.DataFrame, *, method_names: tuple[str, ...] | None, column: str = "method") -> pd.DataFrame:
    if method_names is None or column not in df.columns:
        return df
    allowed_methods = set(method_names)
    return df[df[column].isin(allowed_methods)].reset_index(drop=True)


def canonicalize_methods_in_frame(df: pd.DataFrame) -> pd.DataFrame:
    if "method" not in df.columns:
        return df
    normalized = df.copy()
    normalized["method"] = normalized["method"].map(lambda value: canonicalize_method_name(str(value)))
    return normalized


def min_max_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0 or np.allclose(values.max(), values.min()):
        return np.zeros_like(values, dtype=np.float64)
    return (values - values.min()) / (values.max() - values.min())


def build_method_score_frame(
    df: pd.DataFrame,
    *,
    positive_metrics: tuple[str, ...],
    negative_metrics: tuple[str, ...],
    score_col: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=["method", score_col])
    score = np.zeros(len(df), dtype=np.float64)
    for metric in positive_metrics:
        if metric in df.columns:
            values = df[metric].to_numpy(dtype=np.float64)
            finite_mask = np.isfinite(values)
            if not finite_mask.any():
                continue
            scaled = np.zeros(len(df), dtype=np.float64)
            scaled[finite_mask] = min_max_scale(values[finite_mask])
            score += scaled
    for metric in negative_metrics:
        if metric in df.columns:
            values = df[metric].to_numpy(dtype=np.float64)
            finite_mask = np.isfinite(values)
            if not finite_mask.any():
                continue
            scaled = np.zeros(len(df), dtype=np.float64)
            scaled[finite_mask] = min_max_scale(values[finite_mask])
            score -= 0.15 * scaled
    scored = df[["method"]].copy()
    scored[score_col] = score
    return scored


def focus_method_is_eligible(method_name: str) -> bool:
    return method_name in {FRONTIER_LITE_METHOD, FRONTIER_ANCHORLESS_METHOD} or not method_name.startswith("ablation_")


def build_focus_selection_table(
    *,
    synthetic_mean: pd.DataFrame,
    real_df: pd.DataFrame,
) -> pd.DataFrame:
    synthetic_scores = build_method_score_frame(
        synthetic_mean,
        positive_metrics=FOCUS_POSITIVE_METRICS,
        negative_metrics=FOCUS_NEGATIVE_METRICS,
        score_col="synthetic_score",
    )
    real_scores = build_method_score_frame(
        real_df,
        positive_metrics=REAL_FOCUS_POSITIVE_METRICS,
        negative_metrics=REAL_FOCUS_NEGATIVE_METRICS,
        score_col="real_score",
    )

    methods = sorted(set(synthetic_scores["method"].tolist()) | set(real_scores["method"].tolist()))
    if not methods:
        return pd.DataFrame(columns=["method", "synthetic_score", "real_score", "combined_score", "focus_eligible"])

    selection = pd.DataFrame({"method": methods})
    selection = selection.merge(synthetic_scores, on="method", how="left")
    selection = selection.merge(real_scores, on="method", how="left")
    selection["synthetic_score"] = selection["synthetic_score"].fillna(0.0)
    selection["real_score"] = selection["real_score"].fillna(0.0)
    selection["combined_score"] = selection["synthetic_score"] + selection["real_score"]
    selection["focus_eligible"] = selection["method"].map(focus_method_is_eligible)
    return selection.sort_values(
        ["focus_eligible", "combined_score", "real_score", "synthetic_score", "method"],
        ascending=[False, False, False, False, True],
    ).reset_index(drop=True)


def resolve_focus_method(
    *,
    focus_selection: pd.DataFrame,
    available_methods: list[str],
    requested_focus: str | None,
) -> str:
    available_method_set = set(available_methods)
    if requested_focus is not None:
        if requested_focus not in available_method_set:
            raise ValueError(f"Requested focus method '{requested_focus}' is not available in this run.")
        return requested_focus

    if not focus_selection.empty:
        eligible = focus_selection[focus_selection["focus_eligible"]]
        if not eligible.empty:
            return str(eligible.iloc[0]["method"])
    return available_methods[0] if available_methods else FRONTIER_LITE_METHOD


def load_cached_training_summary(output_dir: Path) -> dict[str, object] | None:
    summary_path = output_dir / "gate_training_summary.json"
    checkpoint_path = output_dir / "learnable_gate.pt"
    if not (summary_path.exists() and checkpoint_path.exists()):
        return None
    return json.loads(summary_path.read_text(encoding="utf-8"))


def load_cached_synthetic_results(
    *,
    output_dir: Path,
    method_names: tuple[str, ...] | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
    raw_path = output_dir / "synthetic_eval" / "synthetic_holdout_results.csv"
    if not raw_path.exists():
        return None
    synthetic_df = canonicalize_methods_in_frame(pd.read_csv(raw_path))
    synthetic_df = maybe_filter_methods(synthetic_df, method_names=method_names)
    synthetic_mean = summarize_synthetic_mean(synthetic_df)
    synthetic_scenario = summarize_synthetic_by_scenario(synthetic_df)
    return synthetic_df, synthetic_mean, synthetic_scenario


def load_cached_real_results(
    *,
    output_dir: Path,
    method_names: tuple[str, ...] | None,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    real_path = output_dir / "real_benchmark_results.csv"
    gate_snapshot_path = output_dir / "real_gate_snapshot.csv"
    if not (real_path.exists() and gate_snapshot_path.exists()):
        return None
    real_df = canonicalize_methods_in_frame(pd.read_csv(real_path))
    real_df = summarize_real_results(maybe_filter_methods(real_df, method_names=method_names))
    gate_snapshot = pd.read_csv(gate_snapshot_path)
    if "mode" in gate_snapshot.columns:
        gate_snapshot["mode"] = gate_snapshot["mode"].map(lambda value: canonicalize_method_name(str(value)))
    gate_snapshot = maybe_filter_methods(gate_snapshot, method_names=method_names, column="mode")
    return real_df, gate_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and benchmark learnable RefineMoE-HVG gating.")
    parser.add_argument("--output-dir", type=str, default="artifacts_gate_learning")
    parser.add_argument("--preset", type=str, default=None, choices=sorted(EXPERIMENT_PRESETS))
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--refine-epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--gate-type", type=str, default="mlp", choices=["mlp", "linear"])
    parser.add_argument("--train-seeds", type=int, default=6)
    parser.add_argument("--eval-seeds", type=int, default=3)
    parser.add_argument("--candidate-random-gates", type=int, default=6)
    parser.add_argument("--train-with-refine", action="store_true")
    parser.add_argument("--real-input-path", type=str, default=None)
    parser.add_argument("--real-format", type=str, default="auto", choices=["auto", "csv", "tsv", "txt", "mtx", "h5ad"])
    parser.add_argument("--real-obs-path", type=str, default=None)
    parser.add_argument("--real-labels-col", type=str, default=None)
    parser.add_argument("--real-batches-col", type=str, default=None)
    parser.add_argument("--real-max-cells", type=int, default=1000)
    parser.add_argument("--real-max-genes", type=int, default=1200)
    parser.add_argument("--real-top-k", type=int, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=3)
    parser.add_argument("--methods", type=str, default=None)
    parser.add_argument("--focus-method", type=str, default=None)
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--force-synthetic", action="store_true")
    parser.add_argument("--force-real", action="store_true")
    args = parser.parse_args()
    apply_experiment_preset(args, argv=sys.argv[1:])

    requested_methods = None
    if args.methods is not None:
        requested_methods = canonicalize_method_names(
            tuple(part.strip() for part in args.methods.split(",") if part.strip())
        )
    requested_focus = canonicalize_method_name(args.focus_method) if args.focus_method else None

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_dir = output_dir / "synthetic_eval"
    real_dir = output_dir / "real_eval"
    synthetic_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    gate_config = GateLearningConfig(
        top_k=args.top_k,
        random_state=args.seed,
        train_seeds=tuple(range(args.train_seeds)),
        candidate_random_gates=args.candidate_random_gates,
        train_with_refine=args.train_with_refine,
        gate_type=args.gate_type,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(
        f"Device={device} cuda_available={torch.cuda.is_available()} "
        f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0}"
    )
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

    training_summary = None
    if not args.force_retrain:
        training_summary = load_cached_training_summary(output_dir)
    if training_summary is None:
        print("\n[1/3] Training learnable gate...")
        training_summary = train_gate_model(output_dir=str(output_dir), config=gate_config)
    else:
        print("\n[1/3] Reusing cached learnable gate checkpoint...")
    gate_model_path = str(training_summary["checkpoint_path"])
    print(json.dumps(training_summary, indent=2))

    synthetic_bundle = None
    if not args.force_synthetic:
        synthetic_bundle = load_cached_synthetic_results(output_dir=output_dir, method_names=requested_methods)
    if synthetic_bundle is None:
        print("\n[2/3] Running synthetic holdout benchmark...")
        synthetic_df = run_synthetic_holdout(
            output_dir=synthetic_dir,
            gate_model_path=gate_model_path,
            top_k=args.top_k,
            refine_epochs=args.refine_epochs,
            random_state=args.seed,
            eval_seeds=args.eval_seeds,
            bootstrap_samples=args.bootstrap_samples,
            method_names=requested_methods,
        )
        synthetic_mean = summarize_synthetic_mean(synthetic_df)
        synthetic_scenario = summarize_synthetic_by_scenario(synthetic_df)
    else:
        print("\n[2/3] Reusing cached synthetic holdout benchmark...")
        synthetic_df, synthetic_mean, synthetic_scenario = synthetic_bundle
    synthetic_mean.to_csv(output_dir / "synthetic_mean_metrics.csv", index=False)
    synthetic_scenario.to_csv(output_dir / "synthetic_by_scenario_metrics.csv", index=False)

    print(synthetic_mean.round(4).to_string(index=False))
    print("\nPer-scenario summary:")
    print(synthetic_scenario.round(4).to_string(index=False))

    real_bundle = None
    if not args.force_real:
        real_bundle = load_cached_real_results(output_dir=output_dir, method_names=requested_methods)
    if real_bundle is None:
        print("\n[3/3] Running real benchmark...")
        real_df, gate_snapshot = run_real_benchmark(
            output_dir=real_dir,
            gate_model_path=gate_model_path,
            top_k=args.real_top_k or args.top_k,
            refine_epochs=args.refine_epochs,
            random_state=args.seed,
            real_input_path=args.real_input_path,
            real_format=args.real_format,
            real_obs_path=args.real_obs_path,
            real_labels_col=args.real_labels_col,
            real_batches_col=args.real_batches_col,
            real_max_cells=args.real_max_cells,
            real_max_genes=args.real_max_genes,
            bootstrap_samples=args.bootstrap_samples,
            method_names=requested_methods,
        )
        real_df = summarize_real_results(real_df)
    else:
        print("\n[3/3] Reusing cached real benchmark...")
        real_df, gate_snapshot = real_bundle
    real_df.to_csv(output_dir / "real_benchmark_results.csv", index=False)
    gate_snapshot.to_csv(output_dir / "real_gate_snapshot.csv", index=False)
    focus_selection = build_focus_selection_table(
        synthetic_mean=synthetic_mean,
        real_df=real_df,
    )
    focus_method = resolve_focus_method(
        focus_selection=focus_selection,
        available_methods=[str(method) for method in synthetic_mean["method"].tolist()],
        requested_focus=requested_focus,
    )
    synthetic_rel = relative_gain_table(
        synthetic_df,
        group_cols=["method"],
        label="overall",
        focus_method=focus_method,
    )
    synthetic_rel_scenario = relative_gain_table(
        synthetic_df,
        group_cols=["scenario", "method"],
        label="scenario",
        focus_method=focus_method,
    )
    synthetic_rel.to_csv(output_dir / "learnable_vs_baselines_overall.csv", index=False)
    synthetic_rel_scenario.to_csv(output_dir / "learnable_vs_baselines_by_scenario.csv", index=False)
    comparison_vs_history = build_history_comparison(
        root=Path(ROOT),
        synthetic_mean=synthetic_mean,
        synthetic_scenario=synthetic_scenario,
        real_df=real_df,
        focus_method=focus_method,
    )
    ablation_deltas = build_ablation_delta_table(
        synthetic_mean=synthetic_mean,
        synthetic_scenario=synthetic_scenario,
        real_df=real_df,
        focus_method=focus_method,
    )
    comparison_vs_history.to_csv(output_dir / "comparison_vs_history.csv", index=False)
    ablation_deltas.to_csv(output_dir / "ablation_deltas.csv", index=False)

    print(real_df.round(4).to_string(index=False))
    print("\nGate snapshot:")
    print(gate_snapshot.round(4).to_string(index=False))

    report_path = output_dir / "gate_learning_report.txt"
    report_path.write_text(
        build_report(
            training_summary=training_summary,
            synthetic_mean=synthetic_mean,
            synthetic_scenario=synthetic_scenario,
            synthetic_rel=synthetic_rel,
            synthetic_rel_scenario=synthetic_rel_scenario,
            real_df=real_df,
            gate_snapshot=gate_snapshot,
            focus_method=focus_method,
            focus_selection=focus_selection,
            comparison_vs_history=comparison_vs_history,
            ablation_deltas=ablation_deltas,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved report to: {report_path}")


def run_synthetic_holdout(
    *,
    output_dir: Path,
    gate_model_path: str,
    top_k: int,
    refine_epochs: int,
    random_state: int,
    eval_seeds: int,
    bootstrap_samples: int,
    method_names: tuple[str, ...] | None,
) -> pd.DataFrame:
    methods = build_default_method_registry(
        top_k=top_k,
        refine_epochs=refine_epochs,
        random_state=random_state,
        gate_model_path=gate_model_path,
    )
    if method_names is not None:
        allowed_methods = set(method_names)
        methods = {name: fn for name, fn in methods.items() if name in allowed_methods}
    rows: list[dict[str, float | int | str]] = []
    scenarios = ["discrete", "trajectory", "batch_shift"]
    cell_options = (700, 1000, 1300)
    gene_options = (1400, 1900, 2400)

    for scenario_idx, scenario in enumerate(scenarios):
        for eval_offset in range(eval_seeds):
            seed = 100 + random_state + 11 * scenario_idx + eval_offset
            n_cells = cell_options[(eval_offset + scenario_idx) % len(cell_options)]
            n_genes = gene_options[(2 * eval_offset + scenario_idx) % len(gene_options)]
            data = generate_synthetic_scrna(
                scenario=scenario,
                n_cells=n_cells,
                n_genes=n_genes,
                random_state=seed,
            )
            current_top_k = min(top_k, data.counts.shape[1])

            for method_name, method_fn in methods.items():
                scores, elapsed = timed_call(method_fn, data.counts, data.batches, current_top_k)
                method_metadata = extract_method_metadata(method_fn)
                selected = np.argsort(scores)[-current_top_k:]

                metrics = evaluate_selection(
                    counts=data.counts,
                    selected_genes=selected,
                    cell_types=data.cell_types,
                    informative_genes=data.informative_genes,
                    batch_genes=data.batch_genes,
                    scorer_fn=lambda subset_counts, subset_batches, fn=method_fn, k=current_top_k: fn(
                        subset_counts,
                        subset_batches,
                        k,
                    ),
                    batches=data.batches,
                    top_k=current_top_k,
                    random_state=seed,
                    n_bootstrap=bootstrap_samples,
                )
                weak_metrics = evaluate_real_selection(
                    counts=data.counts,
                    selected_genes=selected,
                    scorer_fn=lambda subset_counts, subset_batches, fn=method_fn, k=current_top_k: fn(
                        subset_counts,
                        subset_batches,
                        k,
                    ),
                    labels=data.cell_types,
                    batches=data.batches,
                    top_k=current_top_k,
                    random_state=seed,
                    n_bootstrap=bootstrap_samples,
                )

                row: dict[str, float | int | str] = {
                    "scenario": scenario,
                    "seed": seed,
                    "n_cells": int(n_cells),
                    "n_genes": int(n_genes),
                    "method": method_name,
                    "runtime_sec": float(elapsed),
                }
                row.update({key: float(value) for key, value in metrics.items()})
                row.update({key: float(value) for key, value in weak_metrics.items()})
                row.update(method_metadata)
                rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "synthetic_holdout_results.csv", index=False)
    return df


def run_real_benchmark(
    *,
    output_dir: Path,
    gate_model_path: str,
    top_k: int,
    refine_epochs: int,
    random_state: int,
    real_input_path: str | None,
    real_format: str,
    real_obs_path: str | None,
    real_labels_col: str | None,
    real_batches_col: str | None,
    real_max_cells: int | None,
    real_max_genes: int | None,
    bootstrap_samples: int,
    method_names: tuple[str, ...] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_path, resolved_format, obs_path, labels_col, batches_col = resolve_real_input(
        real_input_path=real_input_path,
        real_format=real_format,
        real_obs_path=real_obs_path,
        real_labels_col=real_labels_col,
        real_batches_col=real_batches_col,
    )
    dataset = load_scrna_dataset(
        data_path=input_path,
        file_format=resolved_format,
        obs_path=obs_path,
        labels_col=labels_col,
        batches_col=batches_col,
    )
    dataset = subsample_dataset(
        dataset,
        max_cells=real_max_cells,
        max_genes=real_max_genes,
        random_state=random_state,
    )
    current_top_k = min(top_k, dataset.counts.shape[1])
    methods = build_default_method_registry(
        top_k=current_top_k,
        refine_epochs=refine_epochs,
        random_state=random_state,
        gate_model_path=gate_model_path,
    )
    if method_names is not None:
        allowed_methods = set(method_names)
        methods = {name: fn for name, fn in methods.items() if name in allowed_methods}

    rows: list[dict[str, float | int | str]] = []
    for method_name, method_fn in methods.items():
        scores, elapsed = timed_call(method_fn, dataset.counts, dataset.batches, current_top_k)
        method_metadata = extract_method_metadata(method_fn)
        selected = np.argsort(scores)[-current_top_k:]
        metrics = evaluate_real_selection(
            counts=dataset.counts,
            selected_genes=selected,
            scorer_fn=lambda subset_counts, subset_batches, fn=method_fn, k=current_top_k: fn(
                subset_counts,
                subset_batches,
                k,
            ),
            labels=dataset.labels,
            batches=dataset.batches,
            top_k=current_top_k,
            random_state=random_state,
            n_bootstrap=bootstrap_samples,
        )
        row: dict[str, float | int | str] = {
            "dataset": dataset.name,
            "method": method_name,
            "top_k": int(current_top_k),
            "runtime_sec": float(elapsed),
        }
        row.update({key: float(value) for key, value in metrics.items()})
        row.update(method_metadata)
        rows.append(row)

        ordered = selected[np.argsort(scores[selected])[::-1]]
        pd.DataFrame(
            {
                "rank": np.arange(1, len(ordered) + 1),
                "gene": dataset.gene_names[ordered],
                "score": scores[ordered],
                "gene_index": ordered,
            }
        ).to_csv(output_dir / f"{method_name}_selected_genes.csv", index=False)

    gate_snapshot = snapshot_gates(
        counts=dataset.counts,
        batches=dataset.batches,
        gate_model_path=gate_model_path,
        top_k=current_top_k,
        random_state=random_state,
    )
    return pd.DataFrame(rows), gate_snapshot


def snapshot_gates(
    *,
    counts: np.ndarray,
    batches: np.ndarray | None,
    gate_model_path: str,
    top_k: int,
    random_state: int,
) -> pd.DataFrame:
    snapshots = []
    snapshot_specs = [
        ("full", "full"),
        ("learnable_gate", "learnable_gate"),
        ("learnable_gate_bank", "learnable_gate_bank"),
        ("learnable_gate_bank_reliable_refine", "learnable_gate_bank_reliable_refine"),
        ("learnable_gate_bank_curr_refine", "learnable_gate_bank_curr_refine"),
        ("learnable_gate_bank_curr_no_regret", "learnable_gate_bank_curr_no_regret"),
        ("learnable_gate_bank_curr_no_refine_policy", "learnable_gate_bank_curr_no_refine_policy"),
        ("learnable_gate_bank_pairregret", "learnable_gate_bank_pairregret"),
        ("learnable_gate_bank_pairregret_calibrated", "learnable_gate_bank_pairregret_calibrated"),
        ("learnable_gate_bank_pairregret_permissioned", "learnable_gate_bank_pairregret_permissioned"),
        ("ablation_pairregret_no_risk", "ablation_pairregret_no_risk"),
        ("ablation_pairregret_no_regret", "ablation_pairregret_no_regret"),
        ("ablation_pairregret_no_pairwise_term", "ablation_pairregret_no_pairwise_term"),
        ("ablation_pairregret_no_conservative_routing", "ablation_pairregret_no_conservative_routing"),
        ("ablation_pairregret_cal_no_regret", "ablation_pairregret_cal_no_regret"),
        ("ablation_pairregret_cal_no_consistency", "ablation_pairregret_cal_no_consistency"),
        ("ablation_pairregret_cal_no_route_constraint", "ablation_pairregret_cal_no_route_constraint"),
        ("ablation_pairregret_cal_utility_only", "ablation_pairregret_cal_utility_only"),
        ("ablation_pairregret_cal_no_conservative_routing", "ablation_pairregret_cal_no_conservative_routing"),
        ("ablation_pairperm_no_permission_head", "ablation_pairperm_no_permission_head"),
        ("ablation_pairperm_fixed_budget", "ablation_pairperm_fixed_budget"),
        ("ablation_pairperm_no_anchor_adaptation", "ablation_pairperm_no_anchor_adaptation"),
        ("ablation_pairperm_no_permission_value_decoupling", "ablation_pairperm_no_permission_value_decoupling"),
        ("ablation_pairperm_no_regret_aux", "ablation_pairperm_no_regret_aux"),
        ("ablation_pairperm_permission_only", "ablation_pairperm_permission_only"),
        ("ablation_pairperm_refine_on", "ablation_pairperm_refine_on"),
        (ESCAPECERT_FULL_METHOD, ESCAPECERT_FULL_METHOD),
        ("ablation_escapecert_no_anchor_escape_head", "ablation_escapecert_no_anchor_escape_head"),
        ("ablation_escapecert_no_set_supervision", "ablation_escapecert_no_set_supervision"),
        ("ablation_escapecert_no_candidate_admissibility", "ablation_escapecert_no_candidate_admissibility"),
        ("ablation_escapecert_fixed_budget", "ablation_escapecert_fixed_budget"),
        ("ablation_escapecert_no_escape_uncertainty", "ablation_escapecert_no_escape_uncertainty"),
        ("ablation_escapecert_no_regret_aux", "ablation_escapecert_no_regret_aux"),
        ("ablation_escapecert_anchor_escape_only", "ablation_escapecert_anchor_escape_only"),
        ("ablation_escapecert_admissibility_only", "ablation_escapecert_admissibility_only"),
        (FRONTIER_FULL_METHOD, FRONTIER_FULL_METHOD),
        ("ablation_frontier_no_teacher_distill", "ablation_frontier_no_teacher_distill"),
        ("ablation_frontier_no_frontier_head", "ablation_frontier_no_frontier_head"),
        ("ablation_frontier_no_frontier_uncertainty", "ablation_frontier_no_frontier_uncertainty"),
        ("ablation_frontier_fixed_frontier", "ablation_frontier_fixed_frontier"),
        (FRONTIER_ANCHORLESS_METHOD, FRONTIER_ANCHORLESS_INTERNAL_MODE),
        ("ablation_frontier_no_regret_aux", "ablation_frontier_no_regret_aux"),
        ("ablation_frontier_teacher_only", "ablation_frontier_teacher_only"),
        (FRONTIER_LITE_METHOD, FRONTIER_LITE_INTERNAL_MODE),
    ]
    learned_modes = {
        mode
        for _, mode in snapshot_specs
        if mode != "full"
    }
    for method_name, mode in snapshot_specs:
        selector = RefineMoEHVGSelector(
            top_k=top_k,
            refine_epochs=4,
            random_state=random_state,
            mode=mode,
            gate_model_path=gate_model_path if mode in learned_modes else None,
        )
        selector.score_genes(counts, batches)
        row: dict[str, float | str] = {
            "mode": method_name,
            "internal_mode": mode,
            "gate_source": selector.last_gate_source or method_name,
        }
        row.update({f"stat_{key}": float(value) for key, value in (selector.last_dataset_stats or {}).items()})
        if selector.last_gate is not None:
            row.update({f"weight_{expert}": float(weight) for expert, weight in zip(EXPERT_NAMES, selector.last_gate, strict=False)})
        if selector.last_gate_metadata is not None:
            for key, value in selector.last_gate_metadata.items():
                try:
                    row[key] = float(value)  # type: ignore[arg-type]
                except (TypeError, ValueError):
                    row[key] = str(value)
        snapshots.append(row)
    return pd.DataFrame(snapshots)


def summarize_synthetic_mean(df: pd.DataFrame) -> pd.DataFrame:
    metric_order = [
        "ari",
        "nmi",
        "label_silhouette",
        "batch_mixing",
        "neighbor_preservation",
        "stability",
        "informative_recall",
        "batch_gene_fraction",
        "pairwise_routed_away_from_stage1",
        "anchor_escape_calibrated",
        "allowed_set_total_mass",
        "runtime_sec",
    ]
    cols = [col for col in metric_order if col in df.columns]
    return df.groupby("method", as_index=False)[cols].mean().sort_values(["ari", "nmi"], ascending=False)


def summarize_synthetic_by_scenario(df: pd.DataFrame) -> pd.DataFrame:
    metric_order = [
        "ari",
        "nmi",
        "label_silhouette",
        "batch_mixing",
        "neighbor_preservation",
        "stability",
        "informative_recall",
        "batch_gene_fraction",
        "pairwise_routed_away_from_stage1",
        "anchor_escape_calibrated",
        "allowed_set_total_mass",
        "runtime_sec",
    ]
    cols = [col for col in metric_order if col in df.columns]
    return df.groupby(["scenario", "method"], as_index=False)[cols].mean().sort_values(["scenario", "ari"], ascending=[True, False])


def summarize_real_results(df: pd.DataFrame) -> pd.DataFrame:
    metric_order = [
        "ari",
        "nmi",
        "label_silhouette",
        "batch_mixing",
        "neighbor_preservation",
        "cluster_silhouette",
        "stability",
        "runtime_sec",
    ]
    sort_cols = [col for col in metric_order if col in df.columns]
    if not sort_cols:
        return df
    ascending = [False] * (len(sort_cols) - 1) + [True]
    return df.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)


def ordered_baseline_methods(focus_method: str, available_methods: set[str]) -> list[str]:
    priority = [
        FRONTIER_LITE_METHOD,
        FRONTIER_FULL_METHOD,
        ESCAPECERT_FULL_METHOD,
        PERMISSIONED_METHOD,
        PAIRREGRET_METHOD,
        "learnable_gate_bank_pairregret_calibrated",
        "learnable_gate_bank_curr_refine",
        "learnable_gate_bank_reliable_refine",
        BANK_METHOD,
        "learnable_gate",
        MV_RESIDUAL_METHOD,
        "refine_moe_hvg",
        "ablation_no_moe",
        "variance",
        "fano",
    ]
    if focus_method in {FRONTIER_LITE_METHOD, FRONTIER_ANCHORLESS_METHOD, FRONTIER_FULL_METHOD}:
        priority = [
            FRONTIER_ANCHORLESS_METHOD,
            FRONTIER_LITE_METHOD,
            FRONTIER_FULL_METHOD,
            ESCAPECERT_FULL_METHOD,
            "ablation_frontier_no_teacher_distill",
            "ablation_frontier_no_frontier_head",
            "ablation_frontier_no_frontier_uncertainty",
            "ablation_frontier_fixed_frontier",
            FRONTIER_ANCHORLESS_METHOD,
            "ablation_frontier_no_regret_aux",
            "ablation_frontier_teacher_only",
            "ablation_escapecert_no_anchor_escape_head",
            "ablation_escapecert_no_set_supervision",
            "ablation_escapecert_no_candidate_admissibility",
            "ablation_escapecert_fixed_budget",
            "ablation_escapecert_no_escape_uncertainty",
            "ablation_escapecert_no_regret_aux",
            "ablation_escapecert_anchor_escape_only",
            "ablation_escapecert_admissibility_only",
            BANK_METHOD,
            MV_RESIDUAL_METHOD,
        ] + priority
    elif focus_method == ESCAPECERT_FULL_METHOD:
        priority = [
            FRONTIER_LITE_METHOD,
            FRONTIER_FULL_METHOD,
            ESCAPECERT_FULL_METHOD,
            "ablation_escapecert_no_anchor_escape_head",
            "ablation_escapecert_no_set_supervision",
            "ablation_escapecert_no_candidate_admissibility",
            "ablation_escapecert_fixed_budget",
            "ablation_escapecert_no_escape_uncertainty",
            "ablation_escapecert_no_regret_aux",
            "ablation_escapecert_anchor_escape_only",
            "ablation_escapecert_admissibility_only",
            BANK_METHOD,
            MV_RESIDUAL_METHOD,
        ] + priority
    ordered: list[str] = []
    seen: set[str] = {focus_method}
    for method in priority:
        canonical_method = canonicalize_method_name(method)
        if canonical_method in seen or canonical_method not in available_methods:
            continue
        seen.add(canonical_method)
        ordered.append(canonical_method)
    return ordered


def relative_gain_table(
    df: pd.DataFrame,
    *,
    group_cols: list[str],
    label: str,
    focus_method: str,
) -> pd.DataFrame:
    metrics = [
        "ari",
        "nmi",
        "label_silhouette",
        "batch_mixing",
        "neighbor_preservation",
        "stability",
        "informative_recall",
        "batch_gene_fraction",
        "pairwise_routed_away_from_stage1",
        "anchor_escape_calibrated",
        "allowed_set_total_mass",
        "runtime_sec",
    ]
    available_metrics = [metric for metric in metrics if metric in df.columns]

    if group_cols == ["method"]:
        grouped = df.groupby("method")[available_metrics].mean().reset_index()
        lookup = {row["method"]: row for row in grouped.to_dict(orient="records")}
        rows = []
        if focus_method not in lookup:
            return pd.DataFrame(rows)
        for baseline in ordered_baseline_methods(focus_method, set(lookup)):
            for metric in available_metrics:
                learnable = float(lookup[focus_method][metric])
                base = float(lookup[baseline][metric])
                rows.append(
                    {
                        "scope": label,
                        "target_method": focus_method,
                        "baseline": baseline,
                        "metric": metric,
                        "learnable_minus_baseline": learnable - base,
                        "relative_change_pct": 100.0 * (learnable - base) / max(abs(base), 1e-8),
                    }
                )
        return pd.DataFrame(rows)

    grouped = df.groupby(group_cols)[available_metrics].mean().reset_index()
    rows = []
    for scenario in sorted(grouped["scenario"].unique()):
        subset = grouped[grouped["scenario"] == scenario]
        lookup = {row["method"]: row for row in subset.to_dict(orient="records")}
        if focus_method not in lookup:
            continue
        for baseline in ordered_baseline_methods(focus_method, set(lookup)):
            for metric in available_metrics:
                learnable = float(lookup[focus_method][metric])
                base = float(lookup[baseline][metric])
                rows.append(
                    {
                        "scope": scenario,
                        "target_method": focus_method,
                        "baseline": baseline,
                        "metric": metric,
                        "learnable_minus_baseline": learnable - base,
                        "relative_change_pct": 100.0 * (learnable - base) / max(abs(base), 1e-8),
                    }
                )
    return pd.DataFrame(rows)


def build_history_comparison(
    *,
    root: Path,
    synthetic_mean: pd.DataFrame,
    synthetic_scenario: pd.DataFrame,
    real_df: pd.DataFrame,
    focus_method: str,
) -> pd.DataFrame:
    metric_order = [
        "ari",
        "nmi",
        "label_silhouette",
        "batch_mixing",
        "neighbor_preservation",
        "informative_recall",
        "batch_gene_fraction",
        "pairwise_routed_away_from_stage1",
        "anchor_escape_calibrated",
        "allowed_set_total_mass",
        "runtime_sec",
    ]
    rows: list[dict[str, float | str]] = []

    def append_rows(
        *,
        scope_tag: str,
        baseline_method: str,
        baseline_synthetic_mean: pd.DataFrame,
        baseline_synthetic_scenario: pd.DataFrame,
        baseline_real_df: pd.DataFrame,
    ) -> None:
        if baseline_method == focus_method:
            return
        focus_overall = synthetic_mean[synthetic_mean["method"] == focus_method]
        focus_real = real_df[real_df["method"] == focus_method]
        baseline_overall = baseline_synthetic_mean[baseline_synthetic_mean["method"] == baseline_method]
        baseline_real = baseline_real_df[baseline_real_df["method"] == baseline_method]

        if not focus_overall.empty and not baseline_overall.empty:
            focus_overall_row = focus_overall.iloc[0]
            baseline_overall_row = baseline_overall.iloc[0]
            for metric in metric_order:
                if metric in focus_overall_row.index and metric in baseline_overall_row.index:
                    rows.append(
                        {
                            "scope": f"synthetic_overall_{scope_tag}",
                            "metric": metric,
                            "focus_method": focus_method,
                            "baseline_method": baseline_method,
                            "focus_value": float(focus_overall_row[metric]),
                            "baseline_value": float(baseline_overall_row[metric]),
                            "delta": float(focus_overall_row[metric]) - float(baseline_overall_row[metric]),
                        }
                    )

        for scenario in sorted(synthetic_scenario["scenario"].unique()):
            focus_scenario = synthetic_scenario[
                (synthetic_scenario["method"] == focus_method) & (synthetic_scenario["scenario"] == scenario)
            ]
            baseline_scenario = baseline_synthetic_scenario[
                (baseline_synthetic_scenario["method"] == baseline_method)
                & (baseline_synthetic_scenario["scenario"] == scenario)
            ]
            if focus_scenario.empty or baseline_scenario.empty:
                continue
            focus_scenario_row = focus_scenario.iloc[0]
            baseline_scenario_row = baseline_scenario.iloc[0]
            for metric in metric_order:
                if metric in focus_scenario_row.index and metric in baseline_scenario_row.index:
                    rows.append(
                        {
                            "scope": f"synthetic_{scenario}_{scope_tag}",
                            "metric": metric,
                            "focus_method": focus_method,
                            "baseline_method": baseline_method,
                            "focus_value": float(focus_scenario_row[metric]),
                            "baseline_value": float(baseline_scenario_row[metric]),
                            "delta": float(focus_scenario_row[metric]) - float(baseline_scenario_row[metric]),
                        }
                    )

        if not focus_real.empty and not baseline_real.empty:
            focus_real_row = focus_real.iloc[0]
            baseline_real_row = baseline_real.iloc[0]
            for metric in metric_order:
                if metric in focus_real_row.index and metric in baseline_real_row.index:
                    rows.append(
                        {
                            "scope": f"real_{scope_tag}",
                            "metric": metric,
                            "focus_method": focus_method,
                            "baseline_method": baseline_method,
                            "focus_value": float(focus_real_row[metric]),
                            "baseline_value": float(baseline_real_row[metric]),
                            "delta": float(focus_real_row[metric]) - float(baseline_real_row[metric]),
                        }
                    )

    current_specs = [
        ("current_learnable_gate_bank", BANK_METHOD),
        ("current_mv_residual", MV_RESIDUAL_METHOD),
        ("current_escapecert_full", ESCAPECERT_FULL_METHOD),
        ("current_frontier_full", FRONTIER_FULL_METHOD),
    ]
    for scope_tag, baseline_method in current_specs:
        append_rows(
            scope_tag=scope_tag,
            baseline_method=baseline_method,
            baseline_synthetic_mean=synthetic_mean,
            baseline_synthetic_scenario=synthetic_scenario,
            baseline_real_df=real_df,
        )

    def load_artifact_frames(artifact_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame] | None:
        synthetic_mean_path = artifact_dir / "synthetic_mean_metrics.csv"
        synthetic_scenario_path = artifact_dir / "synthetic_by_scenario_metrics.csv"
        real_path = artifact_dir / "real_benchmark_results.csv"
        if not (synthetic_mean_path.exists() and synthetic_scenario_path.exists() and real_path.exists()):
            return None
        return (
            canonicalize_methods_in_frame(pd.read_csv(synthetic_mean_path)),
            canonicalize_methods_in_frame(pd.read_csv(synthetic_scenario_path)),
            canonicalize_methods_in_frame(pd.read_csv(real_path)),
        )

    history_specs = [
        ("v_next", root / "artifacts_gate_learning_v_next", BANK_METHOD),
        ("v_next2", root / "artifacts_gate_learning_v_next2", "learnable_gate_bank_reliable_refine"),
        ("v_next3", root / "artifacts_gate_learning_v_next3", "learnable_gate_bank_curr_refine"),
        ("v_next4", root / "artifacts_gate_learning_v_next4", PAIRREGRET_METHOD),
        ("v_next5", root / "artifacts_gate_learning_v_next5", "learnable_gate_bank_pairregret_calibrated"),
        ("v_next6", root / "artifacts_gate_learning_v_next6", PERMISSIONED_METHOD),
        ("v_next7_main", root / "artifacts_gate_learning_v_next7", ESCAPECERT_FULL_METHOD),
        ("v_next7_learnable_gate_bank", root / "artifacts_gate_learning_v_next7", BANK_METHOD),
    ]
    for scope_tag, artifact_dir, baseline_method in history_specs:
        artifact_frames = load_artifact_frames(artifact_dir)
        if artifact_frames is None:
            continue
        baseline_synthetic_mean, baseline_synthetic_scenario, baseline_real_df = artifact_frames
        append_rows(
            scope_tag=scope_tag,
            baseline_method=baseline_method,
            baseline_synthetic_mean=baseline_synthetic_mean,
            baseline_synthetic_scenario=baseline_synthetic_scenario,
            baseline_real_df=baseline_real_df,
        )

    v7_frames = load_artifact_frames(root / "artifacts_gate_learning_v_next7")
    if v7_frames is not None:
        v7_synthetic_mean, v7_synthetic_scenario, v7_real_df = v7_frames
        v7_focus_table = build_focus_selection_table(
            synthetic_mean=v7_synthetic_mean,
            real_df=v7_real_df,
        )
        v7_best_overall = resolve_focus_method(
            focus_selection=v7_focus_table,
            available_methods=[str(method) for method in v7_synthetic_mean["method"].tolist()],
            requested_focus=None,
        )
        append_rows(
            scope_tag="v_next7_best_overall",
            baseline_method=v7_best_overall,
            baseline_synthetic_mean=v7_synthetic_mean,
            baseline_synthetic_scenario=v7_synthetic_scenario,
            baseline_real_df=v7_real_df,
        )

    return pd.DataFrame(rows)


def build_ablation_delta_table(
    *,
    synthetic_mean: pd.DataFrame,
    synthetic_scenario: pd.DataFrame,
    real_df: pd.DataFrame,
    focus_method: str,
) -> pd.DataFrame:
    available_methods = set(synthetic_mean["method"].tolist()) | set(real_df["method"].tolist())
    if focus_method == FRONTIER_LITE_METHOD:
        ablations = [
            FRONTIER_ANCHORLESS_METHOD,
            FRONTIER_FULL_METHOD,
            ESCAPECERT_FULL_METHOD,
            "ablation_frontier_no_teacher_distill",
            "ablation_frontier_no_frontier_head",
            "ablation_frontier_no_frontier_uncertainty",
            "ablation_frontier_fixed_frontier",
            "ablation_frontier_no_regret_aux",
            "ablation_frontier_teacher_only",
            "ablation_escapecert_no_anchor_escape_head",
            "ablation_escapecert_no_set_supervision",
            "ablation_escapecert_no_candidate_admissibility",
            "ablation_escapecert_fixed_budget",
            "ablation_escapecert_no_escape_uncertainty",
            "ablation_escapecert_no_regret_aux",
            "ablation_escapecert_anchor_escape_only",
            "ablation_escapecert_admissibility_only",
        ]
    elif focus_method == FRONTIER_ANCHORLESS_METHOD:
        ablations = [
            FRONTIER_LITE_METHOD,
            FRONTIER_FULL_METHOD,
            ESCAPECERT_FULL_METHOD,
            "ablation_frontier_no_teacher_distill",
            "ablation_frontier_no_frontier_head",
            "ablation_frontier_no_frontier_uncertainty",
            "ablation_frontier_fixed_frontier",
            "ablation_frontier_no_regret_aux",
            "ablation_frontier_teacher_only",
            "ablation_escapecert_no_anchor_escape_head",
            "ablation_escapecert_no_set_supervision",
            "ablation_escapecert_no_candidate_admissibility",
            "ablation_escapecert_fixed_budget",
            "ablation_escapecert_no_escape_uncertainty",
            "ablation_escapecert_no_regret_aux",
            "ablation_escapecert_anchor_escape_only",
            "ablation_escapecert_admissibility_only",
        ]
    elif focus_method == FRONTIER_FULL_METHOD:
        ablations = [
            FRONTIER_ANCHORLESS_METHOD,
            FRONTIER_LITE_METHOD,
            ESCAPECERT_FULL_METHOD,
            "ablation_frontier_no_teacher_distill",
            "ablation_frontier_no_frontier_head",
            "ablation_frontier_no_frontier_uncertainty",
            "ablation_frontier_fixed_frontier",
            "ablation_frontier_no_regret_aux",
            "ablation_frontier_teacher_only",
        ]
    elif focus_method == ESCAPECERT_FULL_METHOD:
        ablations = [
            FRONTIER_ANCHORLESS_METHOD,
            FRONTIER_LITE_METHOD,
            FRONTIER_FULL_METHOD,
            "ablation_escapecert_no_anchor_escape_head",
            "ablation_escapecert_no_set_supervision",
            "ablation_escapecert_no_candidate_admissibility",
            "ablation_escapecert_fixed_budget",
            "ablation_escapecert_no_escape_uncertainty",
            "ablation_escapecert_no_regret_aux",
            "ablation_escapecert_anchor_escape_only",
            "ablation_escapecert_admissibility_only",
        ]
    elif focus_method == PERMISSIONED_METHOD:
        ablations = [
            FRONTIER_ANCHORLESS_METHOD,
            ESCAPECERT_FULL_METHOD,
            FRONTIER_LITE_METHOD,
            FRONTIER_FULL_METHOD,
            "ablation_pairperm_no_permission_head",
            "ablation_pairperm_fixed_budget",
            "ablation_pairperm_no_anchor_adaptation",
            "ablation_pairperm_no_permission_value_decoupling",
            "ablation_pairperm_no_regret_aux",
            "ablation_pairperm_permission_only",
        ]
    else:
        ablations = [
            FRONTIER_ANCHORLESS_METHOD,
            PAIRREGRET_METHOD,
            PERMISSIONED_METHOD,
            ESCAPECERT_FULL_METHOD,
            FRONTIER_FULL_METHOD,
            FRONTIER_LITE_METHOD,
            "ablation_pairregret_cal_no_regret",
            "ablation_pairregret_cal_no_consistency",
            "ablation_pairregret_cal_no_route_constraint",
            "ablation_pairregret_cal_utility_only",
            "ablation_pairregret_cal_no_conservative_routing",
        ]
    deduped_ablations: list[str] = []
    seen_ablations: set[str] = set()
    for method_name in ablations:
        canonical_method = canonicalize_method_name(method_name)
        if canonical_method == focus_method or canonical_method not in available_methods or canonical_method in seen_ablations:
            continue
        seen_ablations.add(canonical_method)
        deduped_ablations.append(canonical_method)
    ablations = deduped_ablations
    metric_order = [
        "ari",
        "nmi",
        "label_silhouette",
        "batch_mixing",
        "neighbor_preservation",
        "informative_recall",
        "batch_gene_fraction",
        "pairwise_routed_away_from_stage1",
        "anchor_escape_calibrated",
        "allowed_set_total_mass",
        "runtime_sec",
    ]
    rows: list[dict[str, float | str]] = []

    def append_rows(scope: str, focus_row: pd.Series, baseline_row: pd.Series) -> None:
        for metric in metric_order:
            if metric in focus_row.index and metric in baseline_row.index:
                rows.append(
                    {
                        "scope": scope,
                        "focus_method": focus_method,
                        "ablation_method": str(baseline_row["method"]),
                        "metric": metric,
                        "focus_value": float(focus_row[metric]),
                        "ablation_value": float(baseline_row[metric]),
                        "delta": float(focus_row[metric]) - float(baseline_row[metric]),
                    }
                )

    focus_overall = synthetic_mean[synthetic_mean["method"] == focus_method]
    if not focus_overall.empty:
        focus_row = focus_overall.iloc[0]
        for ablation in ablations:
            baseline = synthetic_mean[synthetic_mean["method"] == ablation]
            if not baseline.empty:
                append_rows("synthetic_overall", focus_row, baseline.iloc[0])

    for scenario in sorted(synthetic_scenario["scenario"].unique()):
        focus_scenario = synthetic_scenario[
            (synthetic_scenario["method"] == focus_method) & (synthetic_scenario["scenario"] == scenario)
        ]
        if focus_scenario.empty:
            continue
        focus_row = focus_scenario.iloc[0]
        for ablation in ablations:
            baseline = synthetic_scenario[
                (synthetic_scenario["method"] == ablation) & (synthetic_scenario["scenario"] == scenario)
            ]
            if not baseline.empty:
                append_rows(f"synthetic_{scenario}", focus_row, baseline.iloc[0])

    focus_real = real_df[real_df["method"] == focus_method]
    if not focus_real.empty:
        focus_row = focus_real.iloc[0]
        for ablation in ablations:
            baseline = real_df[real_df["method"] == ablation]
            if not baseline.empty:
                append_rows("real", focus_row, baseline.iloc[0])

    return pd.DataFrame(rows)


def resolve_real_input(
    *,
    real_input_path: str | None,
    real_format: str,
    real_obs_path: str | None,
    real_labels_col: str | None,
    real_batches_col: str | None,
) -> tuple[str, str, str | None, str | None, str | None]:
    if real_input_path is not None:
        return real_input_path, real_format, real_obs_path, real_labels_col, real_batches_col

    base = Path(ROOT) / "tmp_real_input"
    h5ad_path = base / "minimal.h5ad"
    csv_path = base / "counts.csv"
    obs_path = base / "obs.csv"

    if h5ad_path.exists():
        return str(h5ad_path), "h5ad", None, real_labels_col or "cell_type", real_batches_col or "batch"
    if csv_path.exists():
        return str(csv_path), "csv", str(obs_path), real_labels_col or "cell_type", real_batches_col or "batch"
    raise FileNotFoundError("No default real input found. Expected tmp_real_input/minimal.h5ad or counts.csv.")


def build_report(
    *,
    training_summary: dict[str, object],
    synthetic_mean: pd.DataFrame,
    synthetic_scenario: pd.DataFrame,
    synthetic_rel: pd.DataFrame,
    synthetic_rel_scenario: pd.DataFrame,
    real_df: pd.DataFrame,
    gate_snapshot: pd.DataFrame,
    focus_method: str,
    focus_selection: pd.DataFrame,
    comparison_vs_history: pd.DataFrame | None = None,
    ablation_deltas: pd.DataFrame | None = None,
) -> str:
    sections = [
        "Gate Training Summary",
        json.dumps(training_summary, indent=2),
        "",
        "Focus Method",
        focus_method,
        "",
        "Focus Method Selection",
        focus_selection.round(4).to_string(index=False) if not focus_selection.empty else "N/A",
        "",
        "Synthetic Mean Metrics",
        synthetic_mean.round(4).to_string(index=False),
        "",
        "Synthetic Scenario Metrics",
        synthetic_scenario.round(4).to_string(index=False),
        "",
        "Focus Method vs Baselines Overall",
        synthetic_rel.round(4).to_string(index=False) if not synthetic_rel.empty else "N/A",
        "",
        "Focus Method vs Baselines By Scenario",
        synthetic_rel_scenario.round(4).to_string(index=False) if not synthetic_rel_scenario.empty else "N/A",
        "",
        "Comparison Vs History",
        comparison_vs_history.round(4).to_string(index=False)
        if comparison_vs_history is not None and not comparison_vs_history.empty
        else "N/A",
        "",
        "Ablation Deltas",
        ablation_deltas.round(4).to_string(index=False) if ablation_deltas is not None and not ablation_deltas.empty else "N/A",
        "",
        "Real Benchmark",
        real_df.round(4).to_string(index=False),
        "",
        "Gate Snapshot",
        gate_snapshot.round(4).to_string(index=False),
    ]
    return "\n".join(sections)


if __name__ == "__main__":
    main()
