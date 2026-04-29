from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from itertools import product
import math
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from spatial_context.simulation import (  # noqa: E402
    HierarchicalSimulationScenario,
    build_method_long_results,
    evaluate_simulated_replicate,
    get_simulation_runtime_info,
    simulate_hierarchical_motif_replicate,
    summarize_simulation_metrics,
)


EXPERIMENT_DIR = ROOT / "experiments" / "09_sample_permutation_simulation"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Simulate sample-aware motif differential testing and compare naive Fisher against sample-level permutation."
    )
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR))
    parser.add_argument("--n-replicates", type=int, default=100)
    parser.add_argument("--n-case", nargs="+", type=int, default=[3, 6])
    parser.add_argument("--n-control", nargs="+", type=int, default=[3, 6])
    parser.add_argument("--effect-size", nargs="+", type=float, default=[0.0, 0.8, 1.2])
    parser.add_argument("--sample-random-effect-sd", nargs="+", type=float, default=[0.0, 0.8])
    parser.add_argument("--patch-random-effect-sd", nargs="+", type=float, default=[0.0, 1.0])
    parser.add_argument("--patches-per-sample", nargs="+", type=int, default=[8])
    parser.add_argument("--spots-per-patch", nargs="+", type=int, default=[64])
    parser.add_argument("--n-motifs", type=int, default=100)
    parser.add_argument("--n-signal-motifs", type=int, default=20)
    parser.add_argument("--baseline-prevalence-low", type=float, default=0.02)
    parser.add_argument("--baseline-prevalence-high", type=float, default=0.18)
    parser.add_argument("--fdr-alpha", type=float, default=0.10)
    parser.add_argument("--sample-permutation-max-permutations", type=int, default=8192)
    parser.add_argument("--random-state", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    figures_dir = output_dir / "figures"
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runtime_info = get_simulation_runtime_info()
    scenarios = build_scenarios(args)
    if not scenarios:
        raise ValueError("No simulation scenarios were generated from the provided arguments.")

    scenario_manifest = build_scenario_manifest(scenarios)
    scenario_manifest.to_csv(results_dir / "scenario_manifest.csv", index=False)

    motif_frames: list[pd.DataFrame] = []
    for scenario_index, scenario in enumerate(scenarios):
        for replicate_id in range(int(args.n_replicates)):
            replicate_seed = int(args.random_state + scenario_index * 100_000 + replicate_id)
            replicate = simulate_hierarchical_motif_replicate(
                scenario=scenario,
                replicate_id=replicate_id,
                random_state=replicate_seed,
                runtime_info=runtime_info,
            )
            motif_result = evaluate_simulated_replicate(
                replicate=replicate,
                fdr_alpha=float(args.fdr_alpha),
                sample_permutation_max_permutations=int(args.sample_permutation_max_permutations),
            )
            if motif_result.empty:
                continue
            for key, value in scenario.to_dict().items():
                motif_result[key] = value
            motif_frames.append(motif_result)

    motif_results = pd.concat(motif_frames, ignore_index=True) if motif_frames else pd.DataFrame()
    if motif_results.empty:
        raise RuntimeError("Simulation completed without producing motif-level results.")

    method_results = build_method_long_results(motif_results)
    replicate_summary, scenario_summary = summarize_simulation_metrics(method_results)

    method_results = method_results.merge(scenario_manifest, on="scenario_id", how="left")
    replicate_summary = replicate_summary.merge(scenario_manifest, on="scenario_id", how="left")
    scenario_summary = scenario_summary.merge(scenario_manifest, on="scenario_id", how="left")
    comparison = build_method_comparison(scenario_summary)

    motif_results.to_csv(results_dir / "motif_level_results.csv", index=False)
    method_results.to_csv(results_dir / "method_level_results.csv", index=False)
    replicate_summary.to_csv(results_dir / "replicate_summary.csv", index=False)
    scenario_summary.to_csv(results_dir / "scenario_summary.csv", index=False)
    comparison.to_csv(results_dir / "method_comparison.csv", index=False)

    write_runtime_manifest(
        path=results_dir / "runtime_manifest.json",
        runtime_info=runtime_info,
        args=args,
        n_scenarios=len(scenarios),
    )
    write_protocol(
        path=output_dir / "protocol.md",
        args=args,
        runtime_info=runtime_info,
        scenarios=scenarios,
    )
    write_analysis(
        path=output_dir / "analysis.md",
        args=args,
        scenario_summary=scenario_summary,
        comparison=comparison,
    )

    plot_metric_by_scenario(
        scenario_summary=scenario_summary,
        metric="empirical_fdr",
        output_path=figures_dir / "empirical_fdr_by_scenario.png",
        title="Empirical FDR by Scenario",
        y_label="Empirical FDR",
    )
    plot_metric_by_scenario(
        scenario_summary=scenario_summary,
        metric="empirical_power",
        output_path=figures_dir / "empirical_power_by_scenario.png",
        title="Empirical Power by Scenario",
        y_label="Empirical Power",
    )
    plot_fdr_power_tradeoff(
        scenario_summary=scenario_summary,
        output_path=figures_dir / "fdr_power_tradeoff.png",
    )
    if scenario_summary["effect_size"].nunique() > 1:
        plot_metric_vs_effect_size(
            scenario_summary=scenario_summary,
            metric="empirical_fdr",
            output_path=figures_dir / "empirical_fdr_vs_effect_size.png",
            title="Empirical FDR vs Effect Size",
            y_label="Empirical FDR",
        )
        plot_metric_vs_effect_size(
            scenario_summary=scenario_summary,
            metric="empirical_power",
            output_path=figures_dir / "empirical_power_vs_effect_size.png",
            title="Empirical Power vs Effect Size",
            y_label="Empirical Power",
        )


def build_scenarios(args: argparse.Namespace) -> list[HierarchicalSimulationScenario]:
    scenarios: list[HierarchicalSimulationScenario] = []
    for n_case, n_control, effect_size, sample_sd, patch_sd, patches_per_sample, spots_per_patch in product(
        args.n_case,
        args.n_control,
        args.effect_size,
        args.sample_random_effect_sd,
        args.patch_random_effect_sd,
        args.patches_per_sample,
        args.spots_per_patch,
    ):
        scenario_id = (
            f"case{int(n_case)}_ctrl{int(n_control)}"
            f"_eff{format_float_slug(effect_size)}"
            f"_sample{format_float_slug(sample_sd)}"
            f"_patch{format_float_slug(patch_sd)}"
            f"_pps{int(patches_per_sample)}"
            f"_spp{int(spots_per_patch)}"
        )
        scenarios.append(
            HierarchicalSimulationScenario(
                scenario_id=scenario_id,
                n_case=int(n_case),
                n_control=int(n_control),
                n_motifs=int(args.n_motifs),
                n_signal_motifs=int(args.n_signal_motifs),
                patches_per_sample=int(patches_per_sample),
                spots_per_patch=int(spots_per_patch),
                sample_random_effect_sd=float(sample_sd),
                patch_random_effect_sd=float(patch_sd),
                effect_size=float(effect_size),
                baseline_prevalence_low=float(args.baseline_prevalence_low),
                baseline_prevalence_high=float(args.baseline_prevalence_high),
            )
        )
    return scenarios


def build_method_comparison(scenario_summary: pd.DataFrame) -> pd.DataFrame:
    if scenario_summary.empty:
        return pd.DataFrame()
    value_cols = [
        "empirical_fdr",
        "empirical_power",
        "raw_empirical_fdr",
        "raw_empirical_power",
        "mean_discoveries",
        "mean_true_positives",
        "mean_false_positives",
        "null_rejection_rate",
    ]
    comparison = (
        scenario_summary.pivot_table(index="scenario_id", columns="method", values=value_cols, aggfunc="first")
        .sort_index()
        .reset_index()
    )
    comparison.columns = [_flatten_pivot_column(column) for column in comparison.columns.to_flat_index()]
    scenario_cols = [
        "scenario_id",
        "n_case",
        "n_control",
        "n_motifs",
        "n_signal_motifs",
        "patches_per_sample",
        "spots_per_patch",
        "sample_random_effect_sd",
        "patch_random_effect_sd",
        "effect_size",
        "baseline_prevalence_low",
        "baseline_prevalence_high",
        "total_sample_labelings",
        "sample_permutation_min_pvalue",
    ]
    scenario_manifest = scenario_summary.loc[:, scenario_cols].drop_duplicates().reset_index(drop=True)
    comparison = scenario_manifest.merge(comparison, on="scenario_id", how="left")
    comparison["naive_minus_sample_empirical_fdr"] = (
        comparison["naive_fisher_empirical_fdr"] - comparison["sample_permutation_empirical_fdr"]
    )
    comparison["naive_minus_sample_empirical_power"] = (
        comparison["naive_fisher_empirical_power"] - comparison["sample_permutation_empirical_power"]
    )
    return comparison.sort_values(
        [
            "sample_random_effect_sd",
            "patch_random_effect_sd",
            "effect_size",
            "n_case",
            "n_control",
            "scenario_id",
        ]
    ).reset_index(drop=True)


def build_scenario_manifest(scenarios: list[HierarchicalSimulationScenario]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        row = scenario.to_dict()
        total_labelings = int(math.comb(int(scenario.n_samples), int(scenario.n_case)))
        row["total_sample_labelings"] = total_labelings
        row["sample_permutation_min_pvalue"] = float(1.0 / total_labelings) if total_labelings > 0 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def write_runtime_manifest(
    *,
    path: Path,
    runtime_info: object,
    args: argparse.Namespace,
    n_scenarios: int,
) -> None:
    payload = {
        "runtime_info": asdict(runtime_info),
        "args": vars(args),
        "n_scenarios": int(n_scenarios),
    }
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")


def write_protocol(
    *,
    path: Path,
    args: argparse.Namespace,
    runtime_info: object,
    scenarios: list[HierarchicalSimulationScenario],
) -> None:
    lines = [
        "# Protocol",
        "",
        "## Goal",
        "",
        "Simulate hierarchical spot-level motif membership with explicit sample and patch dependence, then compare pooled spot-level Fisher exact testing against sample-level permutation on motif fractions.",
        "",
        "## Generative Model",
        "",
        "For motif `m`, sample `s`, patch `p`, and spot `t`:",
        "",
        "```text",
        "y_mspt ~ Bernoulli(sigmoid(alpha_m + beta_m * I[case_s] + u_ms + v_msp))",
        "u_ms ~ Normal(0, sigma_sample^2)",
        "v_msp ~ Normal(0, sigma_patch^2)",
        "```",
        "",
        "- `n_case` / `n_control` set the number of independent biological samples.",
        "- `sigma_sample` controls sample-level random effects.",
        "- `sigma_patch` controls within-sample patch dependence; spots within a patch share the same patch effect.",
        "- `effect_size` is a logit-scale shift applied to the signal motifs only.",
        "- Half of the signal motifs are case-enriched and half are control-enriched so the permutation test remains two-sided.",
        "- When `effect_size = 0`, the simulator becomes a pure-null calibration scenario even if `n_signal_motifs` is configured above zero.",
        "",
        "## Inference",
        "",
        "- `naive_fisher`: pooled spot-level Fisher exact test, ignoring sample boundaries.",
        "- `sample_permutation`: exact or Monte Carlo label permutation over sample-level motif fractions using the existing `spatial_context.cross_sample_differential.exact_sample_permutation_statistics` function.",
        "- Multiple testing is controlled per replicate with Benjamini-Hochberg at `q <= "
        f"{float(args.fdr_alpha):.2f}`.",
        "- `results/scenario_manifest.csv` records the exact number of sample labelings and the theoretical minimum exact p-value for each `(n_case, n_control)` setting.",
        "",
        "## Scenario Grid",
        "",
        f"- Scenarios: `{len(scenarios)}`",
        f"- Replicates per scenario: `{int(args.n_replicates)}`",
        f"- Motifs per replicate: `{int(args.n_motifs)}` with `{int(args.n_signal_motifs)}` signals",
        f"- Device detection: `torch.cuda.is_available()` -> `{runtime_info.cuda_available}`",
        f"- Runtime device: `{runtime_info.device}` (`{runtime_info.cuda_name}`)",
        "",
        "## Outputs",
        "",
        "- `results/scenario_summary.csv`: scenario-level empirical FDR / power for each method.",
        "- `results/method_comparison.csv`: naive-vs-sample deltas per scenario.",
        "- `results/replicate_summary.csv`: replicate-level variability.",
        "- `results/motif_level_results.csv`: raw motif-level p-values, q-values, truth labels, and effect summaries.",
        "- `figures/*.png`: scenario summaries and FDR/power trade-off plots.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(
    *,
    path: Path,
    args: argparse.Namespace,
    scenario_summary: pd.DataFrame,
    comparison: pd.DataFrame,
) -> None:
    lines = ["# Analysis", ""]
    if scenario_summary.empty or comparison.empty:
        lines.extend(["No scenario summary was produced.", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    alternative_rows = comparison.loc[comparison["effect_size"].to_numpy(dtype=np.float64) > 0.0].copy()
    headline_pool = alternative_rows if not alternative_rows.empty else comparison
    worst_fdr = headline_pool.sort_values("naive_minus_sample_empirical_fdr", ascending=False).iloc[0]
    best_sample_power = headline_pool.sort_values("sample_permutation_empirical_power", ascending=False).iloc[0]

    lines.extend(
        [
            "## Headline",
            "",
            f"- Largest naive-vs-sample FDR gap: `{worst_fdr['scenario_id']}` naive FDR=`{float(worst_fdr['naive_fisher_empirical_fdr']):.3f}` vs sample permutation FDR=`{float(worst_fdr['sample_permutation_empirical_fdr']):.3f}`.",
            f"- Highest sample-permutation power: `{best_sample_power['scenario_id']}` with power=`{float(best_sample_power['sample_permutation_empirical_power']):.3f}` and FDR=`{float(best_sample_power['sample_permutation_empirical_fdr']):.3f}`.",
            "",
        ]
    )

    null_rows = comparison.loc[np.isclose(comparison["effect_size"].to_numpy(dtype=np.float64), 0.0)]
    if not null_rows.empty:
        worst_null = null_rows.sort_values("naive_fisher_null_rejection_rate", ascending=False).iloc[0]
        worst_sample_null = null_rows.sort_values("sample_permutation_null_rejection_rate", ascending=False).iloc[0]
        lines.extend(
            [
                "## Null Calibration",
                "",
                f"- Worst pure-null naive row: `{worst_null['scenario_id']}` null rejection rate=`{float(worst_null['naive_fisher_null_rejection_rate']):.3f}`, mean discoveries=`{float(worst_null['naive_fisher_mean_discoveries']):.2f}`.",
                f"- Worst pure-null sample-permutation row: `{worst_sample_null['scenario_id']}` null rejection rate=`{float(worst_sample_null['sample_permutation_null_rejection_rate']):.3f}`, mean discoveries=`{float(worst_sample_null['sample_permutation_mean_discoveries']):.2f}`.",
                "",
            ]
        )

    ranked = headline_pool.sort_values(
        ["naive_minus_sample_empirical_fdr", "sample_permutation_empirical_power"],
        ascending=[False, False],
    ).head(5)
    lines.extend(["## Top FDR Reductions", ""])
    for _, row in ranked.iterrows():
        lines.append(
            f"- `{row['scenario_id']}`: naive FDR=`{float(row['naive_fisher_empirical_fdr']):.3f}`, sample FDR=`{float(row['sample_permutation_empirical_fdr']):.3f}`, naive power=`{float(row['naive_fisher_empirical_power']):.3f}`, sample power=`{float(row['sample_permutation_empirical_power']):.3f}`."
        )

    bh_floor_limited = comparison.loc[
        comparison["sample_permutation_min_pvalue"].to_numpy(dtype=np.float64)
        * comparison["n_motifs"].to_numpy(dtype=np.float64)
        > float(args.fdr_alpha)
    ].copy()
    if not bh_floor_limited.empty:
        limited = bh_floor_limited.sort_values(
            ["sample_permutation_min_pvalue", "scenario_id"],
            ascending=[False, True],
        ).iloc[0]
        lines.extend(
            [
                "",
                "## Discreteness Limit",
                "",
                f"- Exact permutation is resolution-limited for rows like `{limited['scenario_id']}`: `total_sample_labelings={int(limited['total_sample_labelings'])}` gives a minimum attainable p-value of `{float(limited['sample_permutation_min_pvalue']):.4f}`.",
                f"- With `m={int(limited['n_motifs'])}` motifs and BH target `q <= {float(args.fdr_alpha):.2f}`, that p-value floor alone can prevent any controlled discovery even when raw sample-level p-values move in the right direction.",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- In this simulator, `sample_random_effect_sd`, `patch_random_effect_sd`, and patch/spot replication are the knobs that strengthen pseudoreplication pressure, so naive Fisher power should be interpreted together with empirical FDR rather than in isolation.",
            f"- All headline metrics use BH control at `q <= {float(args.fdr_alpha):.2f}`; the raw-call variants remain available in `results/scenario_summary.csv` when you want to inspect calibration without multiplicity correction.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric_by_scenario(
    *,
    scenario_summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    pivot = (
        scenario_summary.pivot_table(index="scenario_id", columns="method", values=metric, aggfunc="first")
        .sort_index()
        .fillna(0.0)
    )
    if pivot.empty:
        return
    x = np.arange(pivot.shape[0], dtype=np.float64)
    width = 0.36
    fig_width = max(10.0, 0.35 * pivot.shape[0] + 6.0)
    fig, ax = plt.subplots(figsize=(fig_width, 5.5))
    for offset, method_name, color in (
        (-width / 2.0, "naive_fisher", "#CC5A3E"),
        (width / 2.0, "sample_permutation", "#2F6B8A"),
    ):
        if method_name not in pivot.columns:
            continue
        ax.bar(x + offset, pivot[method_name].to_numpy(dtype=np.float64), width=width, label=method_name, color=color, alpha=0.9)
    ax.set_title(title)
    ax.set_ylabel(y_label)
    ax.set_xticks(x)
    ax.set_xticklabels(pivot.index.tolist(), rotation=70, ha="right", fontsize=8)
    ax.set_ylim(0.0, max(1.0, float(np.nanmax(pivot.to_numpy(dtype=np.float64))) * 1.12))
    ax.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_fdr_power_tradeoff(
    *,
    scenario_summary: pd.DataFrame,
    output_path: Path,
) -> None:
    if scenario_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, 5.5))
    colors = {"naive_fisher": "#CC5A3E", "sample_permutation": "#2F6B8A"}
    markers = {"naive_fisher": "o", "sample_permutation": "s"}
    for method_name, subset in scenario_summary.groupby("method", observed=False):
        ax.scatter(
            subset["empirical_fdr"].to_numpy(dtype=np.float64),
            subset["empirical_power"].to_numpy(dtype=np.float64),
            label=str(method_name),
            color=colors.get(str(method_name), "#333333"),
            marker=markers.get(str(method_name), "o"),
            s=64,
            alpha=0.85,
        )
    ax.set_xlabel("Empirical FDR")
    ax.set_ylabel("Empirical Power")
    ax.set_title("FDR / Power Trade-off")
    ax.set_xlim(left=0.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_metric_vs_effect_size(
    *,
    scenario_summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> None:
    if scenario_summary.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5.8))
    color_map = {"naive_fisher": "#CC5A3E", "sample_permutation": "#2F6B8A"}
    grouped = scenario_summary.copy()
    grouped["settings_label"] = grouped.apply(build_settings_label, axis=1)
    for (method_name, settings_label), subset in grouped.groupby(["method", "settings_label"], observed=False):
        subset = subset.sort_values("effect_size")
        ax.plot(
            subset["effect_size"].to_numpy(dtype=np.float64),
            subset[metric].to_numpy(dtype=np.float64),
            marker="o",
            linewidth=1.8,
            label=f"{method_name} | {settings_label}",
            color=color_map.get(str(method_name), "#333333"),
            alpha=0.8,
        )
    ax.set_title(title)
    ax.set_xlabel("Effect Size (logit shift)")
    ax.set_ylabel(y_label)
    ax.set_ylim(0.0, max(1.0, float(np.nanmax(grouped[metric].to_numpy(dtype=np.float64))) * 1.12))
    ax.grid(alpha=0.25, linewidth=0.7)
    ax.legend(frameon=False, fontsize=7, ncol=1, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_settings_label(row: pd.Series) -> str:
    return (
        f"case={int(row['n_case'])}, ctrl={int(row['n_control'])}, "
        f"sample_sd={float(row['sample_random_effect_sd']):.2f}, "
        f"patch_sd={float(row['patch_random_effect_sd']):.2f}, "
        f"pps={int(row['patches_per_sample'])}, spp={int(row['spots_per_patch'])}"
    )


def format_float_slug(value: float) -> str:
    text = f"{float(value):.2f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _flatten_pivot_column(column: object) -> str:
    if isinstance(column, str):
        return column
    if not isinstance(column, tuple):
        return str(column)
    left = str(column[0]).strip("_")
    right = str(column[1]).strip("_") if len(column) > 1 else ""
    if left == "scenario_id":
        return "scenario_id"
    if right:
        return f"{right}_{left}"
    return left


if __name__ == "__main__":
    main()
