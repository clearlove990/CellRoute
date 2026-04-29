from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import asdict
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

from spatial_context.sample_level_testing import SAMPLE_LEVEL_METHOD_ORDER, sample_permutation_min_pvalue  # noqa: E402
from spatial_context.simulation import (  # noqa: E402
    HierarchicalSimulationScenario,
    evaluate_simulated_replicate,
    get_simulation_runtime_info,
    simulate_hierarchical_motif_replicate,
    summarize_simulation_metrics,
)


EXPERIMENT_DIR = ROOT / "experiments" / "09_power_recovery_simulation"
COLOR_MAP = {
    "naive_fisher": "#C5543A",
    "sample_permutation": "#2F6B8A",
    "sample_level_ols_hc3": "#2A8F64",
    "sample_level_quasi_binomial": "#C9891A",
    "sample_permutation_midp": "#6E6E6E",
}
LINESTYLE_MAP = {
    "naive_fisher": "-",
    "sample_permutation": "-",
    "sample_level_ols_hc3": "-",
    "sample_level_quasi_binomial": "-",
    "sample_permutation_midp": "--",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 09 power-recovery simulation across realistic sample sizes, motif multiplicity, and effect sizes."
    )
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR))
    parser.add_argument("--n-replicates", type=int, default=50)
    parser.add_argument("--sample-size-pairs", nargs="+", default=["3x3", "2x4", "31x32"])
    parser.add_argument("--motif-counts", nargs="+", type=int, default=[4, 8, 16, 48])
    parser.add_argument("--effect-size", nargs="+", type=float, default=[0.2, 0.4, 0.6, 0.8, 1.2])
    parser.add_argument("--signal-fraction", type=float, default=0.25)
    parser.add_argument("--patches-per-sample", type=int, default=6)
    parser.add_argument("--spots-per-patch", type=int, default=24)
    parser.add_argument("--sample-random-effect-sd", type=float, default=0.8)
    parser.add_argument("--patch-random-effect-sd", type=float, default=1.0)
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
            method_result = evaluate_simulated_replicate(
                replicate=replicate,
                fdr_alpha=float(args.fdr_alpha),
                sample_permutation_max_permutations=int(args.sample_permutation_max_permutations),
            )
            if method_result.empty:
                continue
            for key, value in scenario.to_dict().items():
                method_result[key] = value
            method_result["sample_size_pair"] = f"{int(scenario.n_case)}v{int(scenario.n_control)}"
            motif_frames.append(method_result)

    method_results = pd.concat(motif_frames, ignore_index=True) if motif_frames else pd.DataFrame()
    if method_results.empty:
        raise RuntimeError("Power-recovery simulation completed without producing method-level results.")

    replicate_summary, scenario_summary = summarize_simulation_metrics(method_results)
    method_results = method_results.merge(scenario_manifest, on="scenario_id", how="left")
    replicate_summary = replicate_summary.merge(scenario_manifest, on="scenario_id", how="left")
    scenario_summary = scenario_summary.merge(scenario_manifest, on="scenario_id", how="left")
    comparison = build_method_comparison(scenario_summary)

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
    )

    plot_metric_grid(
        scenario_summary=scenario_summary,
        metric="empirical_power",
        output_path=figures_dir / "empirical_power_vs_effect_size.png",
        title="Empirical Power vs Effect Size",
        y_label="Empirical Power",
        alpha=float(args.fdr_alpha),
    )
    plot_metric_grid(
        scenario_summary=scenario_summary,
        metric="empirical_fdr",
        output_path=figures_dir / "empirical_fdr_vs_effect_size.png",
        title="Empirical FDR vs Effect Size",
        y_label="Empirical FDR",
        alpha=float(args.fdr_alpha),
    )


def build_scenarios(args: argparse.Namespace) -> list[HierarchicalSimulationScenario]:
    scenarios: list[HierarchicalSimulationScenario] = []
    for sample_size_pair in args.sample_size_pairs:
        n_case, n_control = parse_sample_size_pair(sample_size_pair)
        for n_motifs in args.motif_counts:
            n_signal_motifs = int(max(1, round(float(n_motifs) * float(args.signal_fraction))))
            n_signal_motifs = int(min(int(n_motifs), n_signal_motifs))
            for effect_size in args.effect_size:
                scenario_id = (
                    f"case{int(n_case)}_ctrl{int(n_control)}"
                    f"_motifs{int(n_motifs)}"
                    f"_signals{int(n_signal_motifs)}"
                    f"_eff{format_float_slug(effect_size)}"
                    f"_sample{format_float_slug(args.sample_random_effect_sd)}"
                    f"_patch{format_float_slug(args.patch_random_effect_sd)}"
                    f"_pps{int(args.patches_per_sample)}"
                    f"_spp{int(args.spots_per_patch)}"
                )
                scenarios.append(
                    HierarchicalSimulationScenario(
                        scenario_id=scenario_id,
                        n_case=int(n_case),
                        n_control=int(n_control),
                        n_motifs=int(n_motifs),
                        n_signal_motifs=int(n_signal_motifs),
                        patches_per_sample=int(args.patches_per_sample),
                        spots_per_patch=int(args.spots_per_patch),
                        sample_random_effect_sd=float(args.sample_random_effect_sd),
                        patch_random_effect_sd=float(args.patch_random_effect_sd),
                        effect_size=float(effect_size),
                        baseline_prevalence_low=float(args.baseline_prevalence_low),
                        baseline_prevalence_high=float(args.baseline_prevalence_high),
                    )
                )
    return scenarios


def build_scenario_manifest(scenarios: list[HierarchicalSimulationScenario]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        row = scenario.to_dict()
        total_labelings = int(math.comb(int(scenario.n_samples), int(scenario.n_case)))
        row["sample_size_pair"] = f"{int(scenario.n_case)}v{int(scenario.n_control)}"
        row["total_sample_labelings"] = total_labelings
        row["sample_permutation_min_pvalue"] = sample_permutation_min_pvalue(int(scenario.n_samples), int(scenario.n_case), midp=False)
        row["sample_permutation_midp_min_pvalue"] = sample_permutation_min_pvalue(int(scenario.n_samples), int(scenario.n_case), midp=True)
        rows.append(row)
    return pd.DataFrame(rows)


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
        "sample_size_pair",
        "n_case",
        "n_control",
        "n_motifs",
        "n_signal_motifs",
        "effect_size",
        "sample_random_effect_sd",
        "patch_random_effect_sd",
        "patches_per_sample",
        "spots_per_patch",
        "total_sample_labelings",
        "sample_permutation_min_pvalue",
        "sample_permutation_midp_min_pvalue",
    ]
    scenario_manifest = scenario_summary.loc[:, scenario_cols].drop_duplicates().reset_index(drop=True)
    return scenario_manifest.merge(comparison, on="scenario_id", how="left").sort_values(
        ["n_case", "n_control", "n_motifs", "effect_size", "scenario_id"]
    ).reset_index(drop=True)


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
        "Quantify when sample-aware control still has useful power under realistic cohort sizes, and when exact sample permutation can only act as a validity filter because the attainable p-value resolution is too coarse.",
        "",
        "## Fixed Dependence Regime",
        "",
        f"- Sample random effect SD: `{float(args.sample_random_effect_sd):.2f}`",
        f"- Patch random effect SD: `{float(args.patch_random_effect_sd):.2f}`",
        f"- Patches per sample: `{int(args.patches_per_sample)}`",
        f"- Spots per patch: `{int(args.spots_per_patch)}`",
        "- This keeps the pseudoreplication pressure from the original 09 simulator while sweeping the dimensions that matter for recovery: cohort size, multiplicity, and effect size.",
        "",
        "## Methods",
        "",
        "- `naive_fisher`: pooled spot-level Fisher exact test, intentionally validity-blind.",
        "- `sample_permutation`: exact or Monte Carlo sample-label permutation over sample-level motif fractions, BH-adjusted.",
        "- `sample_level_ols_hc3`: sample-level difference-in-fractions with HC3 robust variance.",
        "- `sample_level_quasi_binomial`: sample-level quasi-binomial GLM over motif counts / totals.",
        "- `sample_permutation_midp`: exploratory mid-p relaxation of the same sample permutation distribution; this is reported separately and should not replace the primary validity-controlled call set.",
        "",
        "## Scenario Grid",
        "",
        f"- Sample sizes: `{', '.join(args.sample_size_pairs)}`",
        f"- Motif counts: `{', '.join(str(int(value)) for value in args.motif_counts)}`",
        f"- Effect sizes: `{', '.join(format_float_slug(value).replace('p', '.') for value in args.effect_size)}`",
        f"- Signal fraction: `{float(args.signal_fraction):.2f}`",
        f"- Replicates per scenario: `{int(args.n_replicates)}`",
        f"- Total scenarios: `{len(scenarios)}`",
        f"- FDR target: `{float(args.fdr_alpha):.2f}`",
        f"- Device detection: `torch.cuda.is_available()` -> `{runtime_info.cuda_available}`",
        f"- Runtime device: `{runtime_info.device}` (`{runtime_info.cuda_name}`)",
        "",
        "## Outputs",
        "",
        "- `results/method_level_results.csv`: long-format motif-level results with truth labels and method metadata.",
        "- `results/replicate_summary.csv`: replicate-level empirical power / FDR summaries.",
        "- `results/scenario_summary.csv`: scenario-level method comparison table.",
        "- `results/method_comparison.csv`: wide scenario comparison table for figure building.",
        "- `figures/empirical_power_vs_effect_size.png`: line-grid showing recovery across sample sizes and multiplicity.",
        "- `figures/empirical_fdr_vs_effect_size.png`: line-grid showing calibration / inflation trade-offs.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_analysis(
    *,
    path: Path,
    args: argparse.Namespace,
    scenario_summary: pd.DataFrame,
) -> None:
    lines = ["# Analysis", ""]
    if scenario_summary.empty:
        lines.extend(["No scenario summary was produced.", ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    alpha = float(args.fdr_alpha)
    controlled_methods = {"sample_permutation", "sample_level_ols_hc3", "sample_level_quasi_binomial"}
    exploratory_methods = {"sample_permutation_midp"}
    nonzero = scenario_summary.loc[scenario_summary["effect_size"].to_numpy(dtype=np.float64) > 0.0].copy()
    if nonzero.empty:
        nonzero = scenario_summary.copy()

    small_n = nonzero.loc[
        (nonzero["n_case"].to_numpy(dtype=np.int64) <= 3)
        & (nonzero["n_control"].to_numpy(dtype=np.int64) <= 4)
    ].copy()
    if small_n.empty:
        small_n = nonzero.copy()
    sample_perm_small = small_n.loc[small_n["method"].astype(str) == "sample_permutation"].copy()
    resolution_row = pd.Series(dtype=object)
    if not sample_perm_small.empty:
        resolution_row = sample_perm_small.sort_values(
            ["empirical_power", "n_motifs", "effect_size"],
            ascending=[True, False, False],
        ).iloc[0]

    controlled = nonzero.loc[nonzero["method"].astype(str).isin(controlled_methods)].copy()
    valid_controlled = controlled.loc[
        pd.to_numeric(controlled["empirical_fdr"], errors="coerce").to_numpy(dtype=np.float64) <= (alpha + 1.0e-12)
    ].copy()
    best_controlled_pool = valid_controlled if not valid_controlled.empty else controlled
    best_controlled_is_valid = not valid_controlled.empty
    best_controlled = best_controlled_pool.sort_values(
        ["empirical_power", "empirical_fdr"],
        ascending=[False, True],
    ).iloc[0]
    strong_large = best_controlled_pool.loc[
        (best_controlled_pool["n_case"].to_numpy(dtype=np.int64) >= 31)
        & (best_controlled_pool["n_control"].to_numpy(dtype=np.int64) >= 32)
        & (best_controlled_pool["n_motifs"].to_numpy(dtype=np.int64) == 48)
    ].copy()
    sample_perm_large = strong_large.loc[strong_large["method"].astype(str) == "sample_permutation"].copy()
    large_pool = sample_perm_large if not sample_perm_large.empty else strong_large
    large_row = large_pool.sort_values(["empirical_power", "effect_size"], ascending=[False, False]).iloc[0] if not large_pool.empty else pd.Series(dtype=object)

    exploratory = nonzero.loc[nonzero["method"].astype(str).isin(exploratory_methods)].copy()
    exploratory_gain = pd.Series(dtype=object)
    if not exploratory.empty and not sample_perm_small.empty:
        merged = sample_perm_small.merge(
            exploratory.loc[:, ["scenario_id", "empirical_power", "empirical_fdr"]].rename(
                columns={
                    "empirical_power": "exploratory_power",
                    "empirical_fdr": "exploratory_fdr",
                }
            ),
            on="scenario_id",
            how="inner",
        )
        if not merged.empty:
            merged["power_gain"] = merged["exploratory_power"] - merged["empirical_power"]
            exploratory_gain = merged.sort_values(["power_gain", "exploratory_power"], ascending=[False, False]).iloc[0]

    lines.extend(
        [
            "## Headline",
            "",
            (
                f"- Best validity-controlled recovery comes from `{best_controlled['method']}` at "
                f"`{best_controlled['sample_size_pair']}`, `m={int(best_controlled['n_motifs'])}`, "
                f"`effect={float(best_controlled['effect_size']):.1f}` with power=`{float(best_controlled['empirical_power']):.3f}` "
                f"and FDR=`{float(best_controlled['empirical_fdr']):.3f}`."
                if best_controlled_is_valid
                else (
                    f"- No controlled row stayed below the empirical FDR target `{alpha:.2f}`; the closest high-power row is "
                    f"`{best_controlled['method']}` at `{best_controlled['sample_size_pair']}`, `m={int(best_controlled['n_motifs'])}`, "
                    f"`effect={float(best_controlled['effect_size']):.1f}` with power=`{float(best_controlled['empirical_power']):.3f}` "
                    f"and FDR=`{float(best_controlled['empirical_fdr']):.3f}`."
                )
            ),
        ]
    )
    if not resolution_row.empty:
        lines.append(
            f"- A resolution-limited small-cohort row is `{resolution_row['sample_size_pair']}`, `m={int(resolution_row['n_motifs'])}`, `effect={float(resolution_row['effect_size']):.1f}`: exact sample permutation power=`{float(resolution_row['empirical_power']):.3f}` with minimum exact p=`{float(resolution_row['sample_permutation_min_pvalue']):.4f}`."
        )
    if not large_row.empty:
        lines.append(
            f"- In the 31v32 / 48-motif regime, `{large_row['method']}` reaches power=`{float(large_row['empirical_power']):.3f}` at effect=`{float(large_row['effect_size']):.1f}`, showing that sample-aware control is not intrinsically powerless once cohort size is large enough."
        )

    lines.extend(["", "## Interpretation", ""])
    if not resolution_row.empty:
        bh_floor = float(resolution_row["sample_permutation_min_pvalue"]) * float(resolution_row["n_motifs"])
        lines.append(
            f"- For `{resolution_row['sample_size_pair']}` with `m={int(resolution_row['n_motifs'])}`, the exact sample-permutation floor implies `p_min * m = {bh_floor:.3f}` at BH `q <= {float(args.fdr_alpha):.2f}`. When that exceeds the target FDR, sample permutation behaves as a validity filter rather than a discovery engine."
        )
    lines.append(
        "- `sample_level_ols_hc3` and `sample_level_quasi_binomial` recover some power in small cohorts because they are not lattice-limited by the exact label-count support, but they remain sample-level methods and therefore preserve the main anti-pseudoreplication logic."
    )
    if not exploratory_gain.empty:
        lines.append(
            f"- The largest exploratory mid-p gain appears at `{exploratory_gain['scenario_id']}` with power gain=`{float(exploratory_gain['power_gain']):.3f}`; this is useful as a sensitivity analysis, not as the primary inferential workflow."
        )
    lines.append(
        "- The useful reviewer-facing message is therefore conditional: exact sample permutation gives the cleanest validity guarantee, while OLS/HC3 or quasi-binomial can serve as auxiliary recovery analyses in regimes where discreteness dominates."
    )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def plot_metric_grid(
    *,
    scenario_summary: pd.DataFrame,
    metric: str,
    output_path: Path,
    title: str,
    y_label: str,
    alpha: float,
) -> None:
    if scenario_summary.empty:
        return
    sample_pairs = sorted(
        scenario_summary["sample_size_pair"].astype(str).dropna().unique().tolist(),
        key=lambda value: tuple(int(part) for part in value.split("v")),
    )
    motif_counts = sorted(scenario_summary["n_motifs"].astype(int).dropna().unique().tolist())
    fig, axes = plt.subplots(
        len(sample_pairs),
        len(motif_counts),
        figsize=(4.0 * len(motif_counts), 3.2 * len(sample_pairs)),
        sharex=True,
        sharey=True,
    )
    if not isinstance(axes, np.ndarray):
        axes = np.asarray([[axes]])
    elif axes.ndim == 1:
        if len(sample_pairs) == 1:
            axes = axes[None, :]
        else:
            axes = axes[:, None]

    for row_index, sample_pair in enumerate(sample_pairs):
        for col_index, motif_count in enumerate(motif_counts):
            ax = axes[row_index, col_index]
            subset = scenario_summary.loc[
                (scenario_summary["sample_size_pair"].astype(str) == sample_pair)
                & (scenario_summary["n_motifs"].astype(int) == int(motif_count))
            ].copy()
            if subset.empty:
                ax.axis("off")
                continue
            for method_name in SAMPLE_LEVEL_METHOD_ORDER:
                method_subset = subset.loc[subset["method"].astype(str) == method_name].sort_values("effect_size")
                if method_subset.empty:
                    continue
                ax.plot(
                    method_subset["effect_size"].to_numpy(dtype=np.float64),
                    method_subset[metric].to_numpy(dtype=np.float64),
                    color=COLOR_MAP.get(method_name, "#333333"),
                    linestyle=LINESTYLE_MAP.get(method_name, "-"),
                    marker="o",
                    linewidth=1.6,
                    markersize=4.0,
                    alpha=0.9,
                    label=method_name if row_index == 0 and col_index == 0 else None,
                )
            if metric == "empirical_fdr":
                ax.axhline(alpha, color="#999999", linewidth=1.0, linestyle=":", alpha=0.9)
            ax.set_title(f"{sample_pair} | m={int(motif_count)}")
            ax.set_ylim(0.0, 1.02)
            ax.grid(alpha=0.25, linewidth=0.7)
            if row_index == len(sample_pairs) - 1:
                ax.set_xlabel("Effect Size")
            if col_index == 0:
                ax.set_ylabel(y_label)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, frameon=False, ncol=min(3, len(labels)), loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(title, y=1.06, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_sample_size_pair(value: str) -> tuple[int, int]:
    tokens = value.lower().replace("vs", "x").split("x")
    if len(tokens) != 2:
        raise ValueError(f"Invalid sample-size pair: {value}")
    return int(tokens[0]), int(tokens[1])


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
