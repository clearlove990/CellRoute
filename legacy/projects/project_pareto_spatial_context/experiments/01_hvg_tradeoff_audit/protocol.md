# HVG Tradeoff Audit Protocol

## 目标

把四类已有 `benchmarkHVG` official partial artifact 汇总成统一结果表，并用新的 Pareto 口径回答一个更清晰的问题：

是否存在某个 HVG candidate，能在 strong `baseline_mix` 上带来 target metric 收益，同时不牺牲 protected metric？

## 数据源

1. `artifacts_benchmarkhvg_formal_partial_20260424_v1`
2. `artifacts_benchmarkhvg_formal_partial_20260424_v3`
3. `artifacts_benchmarkhvg_rank_blend_frontier_20260424_v1`
4. `artifacts_benchmarkhvg_anchor_repair_20260425_v1`

## 统一 baseline 语义

- 这些 artifact 的评估 baseline 都是官方 `baseline_mix`。
- `rank_blend` 与 `anchor_repair` 中的 `adaptive_hybrid_hvg` 是构造 `extra.rank` 的 anchor，不是最后比较的 benchmark baseline。

## 两套判定口径

### A. Artifact 原始口径

沿用原 artifact 的 `wins/losses` 语义：

- 所有指标统一按 “delta > 0 记 win，delta < 0 记 loss”；
- 因此保留原始 `artifact_clean_win` / `artifact_mixed_tradeoff` 字段，仅用于复盘历史决策文本。

### B. Pareto 审计口径

用 `src/pareto_eval/clean_win.py` 中的新定义：

- `discrete`:
  - target metrics: `ARI`, `NMI`
  - protected metrics: `var_ratio`, `LISI`
- `continuous`:
  - target metrics: `max_ARI`, `max_NMI`
  - protected metrics: `var_ratio`, `dist_cor`, `knn_ratio`, `three_nn`

定义：

- `clean_win`:
  - 所有 protected metric 不低于 baseline；
  - 所有 target metric 不低于 baseline；
  - 至少一个 target metric 明显高于 baseline。
- `near_clean`:
  - protected metric 回吐不超过小容忍区间；
  - 至少一个 target metric 达到实质收益阈值；
  - 其余 target metric 没有明显回吐。
- `mixed_tradeoff`:
  - 至少一个 metric 上涨，且至少一个 metric 下跌。
- `dominated`:
  - 没有任何 metric 上涨，且至少一个 metric 下跌。

说明：

- 这里的 `clean_win` 比一个“只要求 protected 不跌、任一 target 上涨”的宽松定义更严格。
- 这样做的原因是：宽松定义会把一类 “一个 target 涨、另一个 target 跌” 的结果误记为 clean，但这类结果并不满足本项目要表达的“publication-clean domination”。

## 结果表字段

主表：

- `experiment_family`
- `method`
- `dataset`
- `setting`
- `budget_or_alpha`
- `metric_name`
- `baseline_value`
- `candidate_value`
- `delta`
- `direction`
- `is_win`
- `is_loss`
- `clean_win`
- `near_clean`
- `mixed_tradeoff`
- `dominated`
- `artifact_clean_win`
- `artifact_mixed_tradeoff`
- `artifact_path`

## 输出

- `results/hvg_tradeoff_master_table.csv`
- `results/hvg_tradeoff_config_summary.csv`
- `analysis.md`
- `figures/fig1_metric_tradeoff_heatmap.png`
- `figures/fig2_clean_win_failure_summary.png`
- `figures/fig3_rank_blend_frontier.png`
