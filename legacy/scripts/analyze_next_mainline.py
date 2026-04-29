from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts_next_direction"
ANCHOR_METHOD = "adaptive_hybrid_hvg"
ADAPTIVE_STAT_METHOD = "adaptive_stat_hvg"
FRONTIER_METHOD = "learnable_gate_bank_pairregret_permissioned_escapecert_frontier_lite"
ATLAS_REGIME = "atlas-like / large homogeneous panel"
NEAR_ORACLE_THRESHOLD = 0.10
MEANINGFUL_HEADROOM_THRESHOLD = 0.20


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate post-mortem and next-mainline analysis artifacts.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def fmt_float(value: float | int | None, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_dataset_headroom_table(
    benchmark_df: pd.DataFrame,
    failure_df: pd.DataFrame,
    biology_df: pd.DataFrame,
) -> pd.DataFrame:
    bench_idx = benchmark_df.set_index(["dataset", "method"])
    biology_idx = biology_df.groupby(["dataset", "method"])["weighted_marker_recall_at_50"].mean()
    failure_idx = failure_df.set_index("dataset")

    rows: list[dict[str, object]] = []
    excluded_methods = {ANCHOR_METHOD, ADAPTIVE_STAT_METHOD, FRONTIER_METHOD}

    for dataset, group in benchmark_df.groupby("dataset", sort=True):
        anchor_row = group[group["method"] == ANCHOR_METHOD].iloc[0]
        adaptive_stat_row = group[group["method"] == ADAPTIVE_STAT_METHOD].iloc[0]
        candidate_pool = group[~group["method"].isin(excluded_methods)].copy()
        best_single_row = candidate_pool.sort_values("overall_score", ascending=False).iloc[0]
        failure_row = failure_idx.loc[dataset]
        best_published_method = str(failure_row["best_published_expert"])
        best_published_row = bench_idx.loc[(dataset, best_published_method)]

        anchor_bio = biology_idx.get((dataset, ANCHOR_METHOD))
        best_published_bio = biology_idx.get((dataset, best_published_method))

        rows.append(
            {
                "row_type": "dataset",
                "dataset": dataset,
                "regime": str(failure_row["regime"]),
                "anchor_score": float(anchor_row["overall_score"]),
                "adaptive_stat_score": float(adaptive_stat_row["overall_score"]),
                "anchor_minus_adaptive_stat": float(anchor_row["overall_score"] - adaptive_stat_row["overall_score"]),
                "best_single_expert": str(best_single_row["method"]),
                "best_single_score": float(best_single_row["overall_score"]),
                "headroom_vs_best_single": float(best_single_row["overall_score"] - anchor_row["overall_score"]),
                "best_published_expert": best_published_method,
                "best_published_score": float(best_published_row["overall_score"]),
                "headroom_vs_best_published": float(best_published_row["overall_score"] - anchor_row["overall_score"]),
                "biology_delta_best_published_minus_anchor": (
                    float(best_published_bio - anchor_bio)
                    if best_published_bio is not None and anchor_bio is not None
                    else float("nan")
                ),
                "ari_delta_best_single_minus_anchor": float(best_single_row["ari"] - anchor_row["ari"]),
                "nmi_delta_best_single_minus_anchor": float(best_single_row["nmi"] - anchor_row["nmi"]),
                "label_silhouette_delta_best_single_minus_anchor": float(
                    best_single_row["label_silhouette"] - anchor_row["label_silhouette"]
                ),
                "neighbor_preservation_delta_best_single_minus_anchor": float(
                    best_single_row["neighbor_preservation"] - anchor_row["neighbor_preservation"]
                ),
                "cluster_silhouette_delta_best_single_minus_anchor": float(
                    best_single_row["cluster_silhouette"] - anchor_row["cluster_silhouette"]
                ),
                "stability_delta_best_single_minus_anchor": float(best_single_row["stability"] - anchor_row["stability"]),
                "near_oracle_le_0_1": int(best_single_row["overall_score"] - anchor_row["overall_score"] <= NEAR_ORACLE_THRESHOLD),
                "meaningful_headroom_gt_0_2": int(best_single_row["overall_score"] - anchor_row["overall_score"] > MEANINGFUL_HEADROOM_THRESHOLD),
            }
        )

    return pd.DataFrame(rows).sort_values(["headroom_vs_best_single", "dataset"], ascending=[False, True]).reset_index(drop=True)


def build_combined_table(dataset_df: pd.DataFrame) -> pd.DataFrame:
    regime_summary = (
        dataset_df.groupby("regime", as_index=False)
        .agg(
            dataset_count=("dataset", "count"),
            mean_headroom_vs_best_single=("headroom_vs_best_single", "mean"),
            median_headroom_vs_best_single=("headroom_vs_best_single", "median"),
            max_headroom_vs_best_single=("headroom_vs_best_single", "max"),
            mean_headroom_vs_best_published=("headroom_vs_best_published", "mean"),
            mean_anchor_minus_adaptive_stat=("anchor_minus_adaptive_stat", "mean"),
            near_oracle_count=("near_oracle_le_0_1", "sum"),
            meaningful_headroom_count=("meaningful_headroom_gt_0_2", "sum"),
        )
        .assign(row_type="regime_summary")
    )

    winner_summary = (
        dataset_df.groupby("best_single_expert", as_index=False)
        .agg(
            winner_count=("dataset", "count"),
            positive_headroom_count=("headroom_vs_best_single", lambda s: int((s > 0).sum())),
            mean_headroom_when_winner=("headroom_vs_best_single", "mean"),
        )
        .rename(columns={"best_single_expert": "winner_method"})
        .assign(row_type="winner_summary")
        .sort_values(["winner_count", "winner_method"], ascending=[False, True])
    )

    global_summary = pd.DataFrame(
        [
            {
                "row_type": "global_summary",
                "dataset_count": int(len(dataset_df)),
                "mean_headroom_vs_best_single": float(dataset_df["headroom_vs_best_single"].mean()),
                "mean_headroom_vs_best_published": float(dataset_df["headroom_vs_best_published"].mean()),
                "mean_anchor_minus_adaptive_stat": float(dataset_df["anchor_minus_adaptive_stat"].mean()),
                "near_oracle_count": int((dataset_df["headroom_vs_best_single"] <= NEAR_ORACLE_THRESHOLD).sum()),
                "meaningful_headroom_count": int((dataset_df["headroom_vs_best_single"] > MEANINGFUL_HEADROOM_THRESHOLD).sum()),
                "anchor_best_or_tie_count": int((dataset_df["headroom_vs_best_single"] <= 0).sum()),
            }
        ]
    )

    return pd.concat([dataset_df, regime_summary, winner_summary, global_summary], ignore_index=True, sort=False)


def build_topk_overlap_summary(dataset_df: pd.DataFrame, topk_raw_df: pd.DataFrame) -> dict[str, object]:
    overlap_rows: list[dict[str, object]] = []
    for row in dataset_df.itertuples(index=False):
        anchor_group = topk_raw_df[(topk_raw_df["dataset"] == row.dataset) & (topk_raw_df["method"] == ANCHOR_METHOD)]
        best_group = topk_raw_df[(topk_raw_df["dataset"] == row.dataset) & (topk_raw_df["method"] == row.best_single_expert)]
        if anchor_group.empty or best_group.empty:
            continue
        overlap_rows.append(
            {
                "dataset": row.dataset,
                "best_single_expert": row.best_single_expert,
                "headroom_vs_best_single": row.headroom_vs_best_single,
                "anchor_setting_std": float(anchor_group["overall_score"].std(ddof=0)),
                "best_setting_std": float(best_group["overall_score"].std(ddof=0)),
                "std_delta_best_minus_anchor": float(best_group["overall_score"].std(ddof=0) - anchor_group["overall_score"].std(ddof=0)),
            }
        )
    overlap_df = pd.DataFrame(overlap_rows)
    positive_overlap = overlap_df[overlap_df["headroom_vs_best_single"] > 0].copy() if not overlap_df.empty else overlap_df
    return {
        "overlap_df": overlap_df,
        "overlap_count": int(len(overlap_df)),
        "positive_overlap_count": int(len(positive_overlap)),
        "positive_mean_std_delta": float(positive_overlap["std_delta_best_minus_anchor"].mean()) if not positive_overlap.empty else float("nan"),
    }


def render_selector_postmortem(
    phase3_holdout_summary_df: pd.DataFrame,
    phase3_selector_summary_df: pd.DataFrame,
    phase3_selector_results_df: pd.DataFrame,
    phase3_bank_audit_df: pd.DataFrame,
) -> str:
    admissible_df = phase3_bank_audit_df[phase3_bank_audit_df["selector_admissible"] == True].copy()  # noqa: E712
    admissible_methods = ", ".join(admissible_df["method"].astype(str).tolist())
    best_single_row = phase3_holdout_summary_df.sort_values("mean_delta_vs_anchor", ascending=False).iloc[1]
    selector_row = phase3_selector_summary_df[phase3_selector_summary_df["policy"] == "holdout_risk_selector"].iloc[0]
    abstained_count = int((phase3_selector_results_df["abstained"] == 1).sum())
    dataset_count = int(len(phase3_selector_results_df))
    fallback_count = int(phase3_holdout_summary_df["fallback_use_count"].fillna(0).sum())

    lines = [
        "# Selector Line Post-mortem",
        "",
        "## Decision",
        "`holdout_risk_release_hvg` 已经在 phase3 被正式封存，不应继续作为主方法方向推进。",
        "",
        "## Why This Is No-Go",
        f"- strict/reproduced selector bank 只有 {len(admissible_df)} 个 admissible experts：{admissible_methods}。",
        f"- strict/reproduced bank 中最强单专家是 `{best_single_row['method']}`，但相对 anchor 的 held-out mean delta 仍为 {fmt_float(best_single_row['mean_delta_vs_anchor'])}。",
        f"- refit selector 在 held-out 上 `{abstained_count}/{dataset_count}` 全部 abstain，release coverage={fmt_float(selector_row['override_rate'], 3)}，mean delta vs anchor={fmt_float(selector_row['mean_score_delta_vs_anchor'])}。",
        f"- strict mode fallback usage 已经为 {fallback_count}，所以失败不是由 silent fallback 或不干净实现导致，而是方法方向本身在当前 benchmark 和 bank 质量下不成立。",
        "",
        "## Evidence Base",
        "- `artifacts_codex_selector_phase3_official_repro/phase3_go_no_go.md`",
        "- `artifacts_codex_selector_phase3_official_repro/strict_repro_holdout_summary.csv`",
        "- `artifacts_codex_selector_phase3_official_repro/strict_repro_selector_summary.csv`",
        "- `artifacts_codex_selector_phase3_official_repro/strict_repro_selector_results.csv`",
        "- `artifacts_codex_selector_phase3_official_repro/bank_audit.csv`",
        "",
        "## What Still Remains Valid",
        "- `adaptive_hybrid_hvg` 仍是当前 repo 中最安全、最可复用的 anchor。",
        "- strict/reproduced bank audit、availability 报告和 reproduction notes 仍然是方法学资产。",
        "- round2/phase2/phase3 benchmark harness、biology proxy、regime taxonomy 仍可直接复用到后续 anchor-centered 分析和新主线验证。",
        "",
        "## Reusable Code And Artifacts",
        "- `src/hvg_research/adaptive_stat.py`",
        "- `src/hvg_research/methods.py`",
        "- `scripts/run_topconf_selector_round2.py`",
        "- `artifacts_topconf_selector_round2/`",
        "- `artifacts_codex_selector_phase3_official_repro/`",
        "",
        "## Why We Should Not Keep Patching The Selector",
        "- 当 admissible bank 本身整体弱于 anchor 时，阈值微调无法制造真实的正向 release。",
        "- phase3 refit 后没有任何 release event，可用于再校准的正例支持已经消失，继续调 selector 只会变成 under-identified 的 operating-point 游戏。",
        "- 继续围绕 selective release 写新故事，只会重述已经被 phase3 否定的方向，而不会解决真正的瓶颈。",
        "",
        "## Conclusion",
        "后续分析与实验应从 selector 线完全转向 anchor 内核本身，或者在证据不足时诚实暂停 benchmark 上的新方法搜索。",
    ]
    return "\n".join(lines) + "\n"


def render_anchor_headroom_analysis(
    dataset_df: pd.DataFrame,
    topk_overlap_summary: dict[str, object],
) -> str:
    near_oracle_count = int((dataset_df["headroom_vs_best_single"] <= NEAR_ORACLE_THRESHOLD).sum())
    near_oracle_0_2_count = int((dataset_df["headroom_vs_best_single"] <= 0.20).sum())
    anchor_best_or_tie = dataset_df[dataset_df["headroom_vs_best_single"] <= 0]["dataset"].astype(str).tolist()
    meaningful_df = dataset_df[dataset_df["headroom_vs_best_single"] > MEANINGFUL_HEADROOM_THRESHOLD].copy()
    positive_df = dataset_df[dataset_df["headroom_vs_best_single"] > 0].copy()
    regime_summary = (
        dataset_df.groupby("regime")["headroom_vs_best_single"]
        .agg(["count", "mean", "median", "max"])
        .reset_index()
        .sort_values("mean", ascending=False)
    )
    metric_means = positive_df[
        [
            "ari_delta_best_single_minus_anchor",
            "nmi_delta_best_single_minus_anchor",
            "label_silhouette_delta_best_single_minus_anchor",
            "neighbor_preservation_delta_best_single_minus_anchor",
            "cluster_silhouette_delta_best_single_minus_anchor",
            "stability_delta_best_single_minus_anchor",
        ]
    ].mean()
    hybrid_vs_stat_material = dataset_df.loc[
        dataset_df["anchor_minus_adaptive_stat"].abs() > 0.01,
        ["dataset", "anchor_minus_adaptive_stat"],
    ].sort_values("anchor_minus_adaptive_stat", ascending=False)
    best_expert_counts = dataset_df["best_single_expert"].value_counts()

    regime_lines = [
        f"- {row['regime']}: {int(row['count'])} 个数据集，mean headroom={fmt_float(row['mean'])}，median={fmt_float(row['median'])}，max={fmt_float(row['max'])}。"
        for _, row in regime_summary.iterrows()
    ]
    meaningful_lines = [
        f"- {row.dataset}: headroom={fmt_float(row.headroom_vs_best_single)}，best single expert=`{row.best_single_expert}`。"
        for row in meaningful_df.itertuples(index=False)
    ]
    hybrid_material_lines = [
        f"- {row.dataset}: anchor - adaptive_stat = {fmt_float(row.anchor_minus_adaptive_stat)}。"
        for row in hybrid_vs_stat_material.itertuples(index=False)
    ]
    best_expert_lines = [
        f"- `{method}`: {int(count)} 个数据集。"
        for method, count in best_expert_counts.items()
    ]

    lines = [
        "# Anchor-Centered Headroom Analysis",
        "",
        "## Scope",
        "- 主分析使用 `artifacts_topconf_selector_round2/benchmark_dataset_summary.csv` 与 `artifacts_topconf_selector_round2/failure_taxonomy.csv`。",
        "- biology proxy 使用 `artifacts_codex_selector_mvp/biology_proxy_raw.csv`。",
        "- phase2 top-k/seed artifact 只作为辅助证据，不再拿来替 selector 翻案。",
        "- 这里的 oracle 明确定义为“best non-selector single expert”，排除 `adaptive_hybrid_hvg`、`adaptive_stat_hvg` 与 frontier selector，以避免把下一主线又绕回 routing。",
        "",
        "## Findings",
        f"1. `adaptive_hybrid_hvg` 不是完全饱和，但 headroom 是结构化的。相对 best non-selector single expert 的 mean headroom 为 {fmt_float(dataset_df['headroom_vs_best_single'].mean())}；{near_oracle_count}/12 个数据集 headroom <= 0.1，{near_oracle_0_2_count}/12 个数据集 headroom <= 0.2。",
        f"2. anchor 只在 {len(anchor_best_or_tie)}/12 个数据集上已经达到或超过 non-selector bank oracle：{', '.join(anchor_best_or_tie)}。",
        "3. 剩余 headroom 明显集中在非 atlas-like regime，而不是平均撒在所有数据集上：",
        *regime_lines,
        "4. 具有明确 headroom 的数据集主要是：",
        *meaningful_lines,
        "5. 这批 headroom 更像 clustering/stability 信号，而不是 biology proxy 或 neighbor-preservation 冲突。正 headroom 数据集上，best expert minus anchor 的平均增量为：",
        f"- ARI: {fmt_float(metric_means['ari_delta_best_single_minus_anchor'])}",
        f"- NMI: {fmt_float(metric_means['nmi_delta_best_single_minus_anchor'])}",
        f"- cluster silhouette: {fmt_float(metric_means['cluster_silhouette_delta_best_single_minus_anchor'])}",
        f"- stability: {fmt_float(metric_means['stability_delta_best_single_minus_anchor'])}",
        f"- label silhouette: {fmt_float(metric_means['label_silhouette_delta_best_single_minus_anchor'])}",
        f"- neighbor preservation: {fmt_float(metric_means['neighbor_preservation_delta_best_single_minus_anchor'])}",
        f"6. 相对 best published expert，anchor 仍有 mean score headroom {fmt_float(dataset_df['headroom_vs_best_published'].mean())}，但 mean biology delta 反而是 {fmt_float(dataset_df['biology_delta_best_published_minus_anchor'].mean())}。这说明剩余 headroom 不是简单的 biology guardrail 校准问题。",
        f"7. 当前 `adaptive_hybrid_hvg` 在 10/12 个数据集上几乎等同于 `adaptive_stat_hvg`，只有 {len(hybrid_material_lines)} 个数据集出现实质性差异：",
        *hybrid_material_lines,
        "8. 正向 headroom 的赢家方法是分散的，而不是单一专家主导：",
        *best_expert_lines,
        "9. 现有 phase2 top-k/seed artifact 与当前 dataset-oracle 配对只重叠了 "
        f"{topk_overlap_summary['overlap_count']}/12 个数据集；重叠的正 headroom 数据集只有 "
        f"{topk_overlap_summary['positive_overlap_count']} 个，而且没有显示出一致的“anchor 特别不稳定”模式。现有证据不足以把主要瓶颈归因于 top-k 稳定性。",
        "",
        "## Interpretation",
        "- atlas-like 数据集上，anchor 基本已经接近 bank oracle，没有明显继续优化空间。",
        "- 真正的 headroom 集中在 batch-heavy、high-dropout、count-model-friendly 这些非 atlas-like regime，而且主要体现为 cluster separation / stability 改善。",
        "- 因为 `adaptive_hybrid_hvg` 几乎就是 `adaptive_stat_hvg` 的外层包装，剩余空间本质上是 anchor core 的空间，而不是再加一个 selector layer 的空间。",
        "",
        "## Decision Implication",
        "如果还要给这个 benchmark 一次新的主线机会，目标必须是“单一 anchor scorer 的内核升级”，而不是 selector、routing、abstain、bank cleanup 或阈值校准。",
    ]
    return "\n".join(lines) + "\n"


def render_next_mainline_proposal(dataset_df: pd.DataFrame) -> str:
    meaningful_df = dataset_df[dataset_df["headroom_vs_best_single"] > MEANINGFUL_HEADROOM_THRESHOLD].copy()
    candidate_datasets = ", ".join(meaningful_df["dataset"].astype(str).tolist())

    lines = [
        "# Next Mainline Proposal",
        "",
        "## Recommendation",
        "唯一值得继续的一条新主线是：把当前 anchor 升级成一个固定的、非路由的 multi-signal consensus scorer，而不是继续做 selector 或 bank 工程。",
        "",
        "建议的新候选方法名可以是 `adaptive_core_consensus_hvg`，定位是 `adaptive_hybrid_hvg` 的后继 anchor，而不是它上面的 release layer。",
        "",
        "## Why This Is The Only Credible Line",
        "- selector 线已经被 phase3 关掉，不能再重开。",
        "- official reproduction / strict bank 现在已经足够诚实，但它不是当前瓶颈。",
        "- 现有 headroom 不是集中在某一个单专家上，所以“换一个更强 expert 当主方法”没有数据支撑。",
        "- 现有 headroom 又确实不是纯噪声，而是稳定地集中在非 atlas-like regime，并主要体现在 clustering/stability 指标上。",
        "- 在不重新回到 dataset-level routing 的前提下，唯一合理的机制假设就是：把多个互补的 gene-level 信号融合进一个单一 scorer，让 anchor 内核本身更强。",
        "",
        "## Mechanism Hypothesis",
        "- 当前 anchor core 主要还是 variance / mean-variance residual / fano 的 profile-conditioned blend。",
        "- 非 atlas-like regime 的赢家方法提供了互补信号：count-model residual、batch-aware variance standardization、局部结构富集。",
        "- 如果一个基因在这些互补信号上都稳定靠前，那么它更可能贡献 cluster separation 与 cross-run stability，而不需要 dataset-level 路由来决定是否 release。",
        "- 因此，固定的 consensus / rank-fusion scorer 有机会在以下数据集上超过当前 anchor："
        f" {candidate_datasets}。",
        "",
        "## What This Is Not",
        "- 不是 selector/routing/abstain 的再包装。",
        "- 不是继续扩 expert bank。",
        "- 不是阈值扫参。",
        "- 不是重新讲 official reproduction 的基础设施故事。",
        "",
        "## Main Risk",
        "风险也很清楚：赢家方法目前是异质的，固定 consensus 可能把专家优势平均掉，最后只退化回现在的 anchor。如果这一点在最小实验里发生，就应该直接停止在这个 benchmark 上继续找新方法。",
    ]
    return "\n".join(lines) + "\n"


def render_minimal_experiment_plan(dataset_df: pd.DataFrame) -> str:
    focus_df = dataset_df[dataset_df["headroom_vs_best_single"] > NEAR_ORACLE_THRESHOLD].copy()
    atlas_df = dataset_df[dataset_df["regime"] == ATLAS_REGIME].copy()
    focus_datasets = ", ".join(focus_df["dataset"].astype(str).tolist())
    atlas_datasets = ", ".join(atlas_df["dataset"].astype(str).tolist())

    lines = [
        "# Next Mainline Minimal Experiment Plan",
        "",
        "## Hypothesis",
        "一个固定的 anchor-internal consensus scorer，若直接融合当前 anchor core 与少量互补信号，可以在不引入 routing 的前提下，提升非 atlas-like regime 的 clustering/stability，从而在 held-out overall behavior 上超过 `adaptive_hybrid_hvg`。",
        "",
        "## Minimal Method Definition",
        "- 新方法：`adaptive_core_consensus_hvg`。",
        "- 形式：单一 scorer；不做 dataset-level routing；不做 abstain；不做阈值校准。",
        "- 最小可测实现建议：对以下组件的 gene score 先转成 percentile rank，再做固定的 rank average：",
        "- `adaptive_stat_hvg` 的当前 core score",
        "- `multinomial_deviance_hvg`",
        "- `scanpy_seurat_v3_hvg`",
        "- `triku_hvg`",
        "- 先用完全固定的 fusion rule 跑一次，不做 sweep；如果第一次失败，直接 no-go，而不是继续调权重。",
        "",
        "## Code Changes",
        "- `src/hvg_research/adaptive_stat.py`：加入 rank/percentile 标准化与 consensus scorer。",
        "- `src/hvg_research/methods.py`：注册 `adaptive_core_consensus_hvg`，并写入清晰 metadata。",
        "- `scripts/run_topconf_selector_round2.py`：把新方法加入一轮最小 benchmark 调用，复用现有 dataset manifest 与评估管线。",
        "",
        "## Reused Assets",
        "- 直接复用 `artifacts_topconf_selector_round2` 对应的 12 个 held-out 数据集与评估指标。",
        "- 直接复用 phase3 已验证过的 strict worker 环境，不把 official reproduction 重新变成主任务。",
        "- 对照基线固定为 `adaptive_hybrid_hvg`。",
        "",
        "## New Output Directory",
        "- `D:\\code_py\\sc-RNAseq\\artifacts_next_direction_mainline_minimal`",
        "",
        "## Minimal Outputs",
        "- `minimal_holdout_results.csv`",
        "- `minimal_holdout_summary.csv`",
        "- `minimal_regime_summary.csv`",
        "- `minimal_go_no_go.md`",
        "",
        "## Go / No-Go",
        "- Go only if all of the following are true after a single top_k=200, seed=7 pass on all 12 datasets.",
        f"- Full held-out mean delta vs `adaptive_hybrid_hvg` > 0.",
        f"- Mean delta on the current positive-headroom subset ({focus_datasets}) > 0.15.",
        f"- Mean delta on atlas-like controls ({atlas_datasets}) >= -0.05.",
        "- Mean biology proxy delta vs `adaptive_hybrid_hvg` >= -0.02.",
        "- If any of these fail, stop. Do not continue with weight sweeps, selectorization, or another expert-bank rewrite.",
        "",
        "## Smallest Unblock",
        "不需要新数据。最小 unblock 只是沿用当前 repo 已有的 benchmark harness 与 phase3 worker 环境，把新 scorer 跑完一次 12 数据集 held-out 评估即可。",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    phase3_dir = ROOT / "artifacts_codex_selector_phase3_official_repro"
    round2_dir = ROOT / "artifacts_topconf_selector_round2"
    mvp_dir = ROOT / "artifacts_codex_selector_mvp"
    phase2_dir = ROOT / "artifacts_codex_selector_phase2"

    phase3_holdout_summary_df = load_csv(phase3_dir / "strict_repro_holdout_summary.csv")
    phase3_selector_summary_df = load_csv(phase3_dir / "strict_repro_selector_summary.csv")
    phase3_selector_results_df = load_csv(phase3_dir / "strict_repro_selector_results.csv")
    phase3_bank_audit_df = load_csv(phase3_dir / "bank_audit.csv")

    benchmark_df = load_csv(round2_dir / "benchmark_dataset_summary.csv")
    failure_df = load_csv(round2_dir / "failure_taxonomy.csv")
    biology_df = load_csv(mvp_dir / "biology_proxy_raw.csv")
    topk_raw_df = load_csv(phase2_dir / "_topk_seed_method_raw.csv")

    dataset_df = build_dataset_headroom_table(
        benchmark_df=benchmark_df,
        failure_df=failure_df,
        biology_df=biology_df,
    )
    combined_table_df = build_combined_table(dataset_df)
    topk_overlap_summary = build_topk_overlap_summary(dataset_df, topk_raw_df)

    combined_table_df.to_csv(output_dir / "anchor_headroom_tables.csv", index=False)
    (output_dir / "selector_line_postmortem.md").write_text(
        render_selector_postmortem(
            phase3_holdout_summary_df=phase3_holdout_summary_df,
            phase3_selector_summary_df=phase3_selector_summary_df,
            phase3_selector_results_df=phase3_selector_results_df,
            phase3_bank_audit_df=phase3_bank_audit_df,
        ),
        encoding="utf-8",
    )
    (output_dir / "anchor_headroom_analysis.md").write_text(
        render_anchor_headroom_analysis(dataset_df=dataset_df, topk_overlap_summary=topk_overlap_summary),
        encoding="utf-8",
    )
    (output_dir / "next_mainline_proposal.md").write_text(
        render_next_mainline_proposal(dataset_df=dataset_df),
        encoding="utf-8",
    )
    (output_dir / "next_mainline_minimal_experiment_plan.md").write_text(
        render_minimal_experiment_plan(dataset_df=dataset_df),
        encoding="utf-8",
    )

    print(f"Wrote next-direction artifacts to {output_dir}")


if __name__ == "__main__":
    main()
