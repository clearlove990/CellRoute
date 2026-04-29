# Pareto Spatial Context Toolkit

本项目把 Week 1 的一次性 Pareto 审计脚本整理成了可复用工具，核心代码位于 `project_pareto_spatial_context/src/pareto_eval/`。

## 目录

- `src/pareto_eval/metrics.py`
  - metric direction registry
  - protected / target metric grouping
  - epsilon / tolerance / gain-threshold 配置
- `src/pareto_eval/frontier.py`
  - Pareto dominance
  - Pareto frontier extraction
  - dominance depth ranking
- `src/pareto_eval/clean_win.py`
  - clean_win / near_clean / mixed_tradeoff / dominated 判定
- `src/pareto_eval/rank_blend.py`
  - alpha grid
  - rank interpolation
- `src/pareto_eval/replacement_frontier.py`
  - anchor replacement budget simulation
  - replacement frontier summary
- `src/pareto_eval/reporting.py`
  - markdown summary
  - CSV summary
- `src/pareto_eval/plotting.py`
  - heatmaps
  - frontier plots
  - win/loss barplots

## 输入格式

Toolkit 支持两种常见输入方式。

1. 长表 delta 输入

最常用，适合直接从实验日志或 metric audit table 接入。

必需列：

- `config_id`
- `evaluation_mode`
- `metric_name`
- `delta`

推荐附带列：

- `baseline_value`
- `candidate_value`
- `experiment_family`
- `display_method`
- `dataset`
- `budget_or_alpha`

说明：

- `delta` 约定为 `candidate_value - baseline_value`。
- 若某个 metric 在 registry 中配置为 `lower_is_better`，toolkit 会自动转成“正值代表改进”的内部方向。

2. 宽表 baseline/candidate 输入

如果已经拿到单条 baseline/candidate 指标字典，可以直接调用：

```python
from pareto_eval.clean_win import classify_candidate_against_baseline
from pareto_eval.metrics import registry_for_evaluation_mode

registry = registry_for_evaluation_mode("discrete")
result = classify_candidate_against_baseline(
    baseline_metrics={"var_ratio": 1.30, "lisi": 1.41, "ari": 0.74, "nmi": 0.82},
    candidate_metrics={"var_ratio": 1.31, "lisi": 1.39, "ari": 0.75, "nmi": 0.83},
    profile=registry,
)
```

## Metric Direction 配置

默认 registry 在 `metrics.py` 中定义：

- `DISCRETE_REGISTRY`
  - protected: `var_ratio`, `lisi`
  - target: `ari`, `nmi`
- `CONTINUOUS_REGISTRY`
  - protected: `var_ratio`, `dist_cor`, `knn_ratio`, `three_nn`
  - target: `max_ari`, `max_nmi`

如果要扩展到新任务，可用 `build_metric_registry(...)` 自定义：

```python
from pareto_eval.metrics import build_metric_registry

custom_registry = build_metric_registry(
    "custom",
    protected_metrics=("latency", "memory"),
    target_metrics=("accuracy", "f1"),
    directions={"latency": "lower_is_better", "memory": "lower_is_better"},
    gain_thresholds={"accuracy": 0.002, "f1": 0.002},
    loss_tolerances={"latency": 5.0, "memory": 50.0, "accuracy": 0.001, "f1": 0.001},
)
```

## Clean Win 判定

给定“正值代表改进”的 delta 后，判定规则为：

- `clean_win`
  - 至少一个 target metric 明确提升
  - 所有 protected metric 不低于 baseline
  - 所有 target metric 不低于 baseline
- `near_clean`
  - 至少一个 target metric 达到 gain threshold
  - protected 回吐不超过 metric-specific loss tolerance
  - 其余 target 不出现超容忍回吐
- `mixed_tradeoff`
  - 至少一个 metric 提升，且至少一个 metric 回吐
- `dominated`
  - 没有任何 metric 提升，且至少一个 metric 回吐
- `flat_or_tied`
  - 全部在 epsilon 以内

## 如何运行 Frontier Analysis

最直接的入口是：

```python
from pareto_eval.frontier import rank_candidates_by_dominance_depth
from pareto_eval.metrics import registry_for_evaluation_mode

registry = registry_for_evaluation_mode("continuous")
ranked = rank_candidates_by_dominance_depth(candidate_frame, registry)
frontier = ranked[ranked["is_pareto_frontier"]]
```

如果输入是 Week 1 风格的长表：

1. 先用 `classify_frame(...)` 得到 clean/mixed/dominated。
2. 再把长表 pivot 成宽表。
3. 用 `rank_candidates_by_dominance_depth(...)` 提取 frontier。

## 如何生成报告

### Week 2 tool validation

PowerShell:

```powershell
python D:\code_py\sc-RNAseq\project_pareto_spatial_context\experiments\02_pareto_tool_validation\run_validation.py
```

输出：

- `experiments/02_pareto_tool_validation/results/tool_validation_summary.csv`
- `experiments/02_pareto_tool_validation/results/smoke_test_results.csv`
- `experiments/02_pareto_tool_validation/analysis.md`
- `experiments/02_pareto_tool_validation/figures/*.png`

### 代码内生成 markdown / CSV

```python
from pareto_eval.reporting import (
    build_candidate_summary,
    generate_csv_summary,
    generate_markdown_summary,
    write_markdown_summary,
)

summary = build_candidate_summary(metric_frame, classification_frame)
markdown_text = generate_markdown_summary(summary, title="My Pareto Report")
generate_csv_summary(summary, "summary.csv")
write_markdown_summary(markdown_text, "summary.md")
```

### 代码内生成图

```python
from pareto_eval.plotting import plot_metric_heatmap, plot_frontier, plot_win_loss_barplot

plot_metric_heatmap(metric_frame, output_path="fig_heatmap.png")
plot_frontier(summary_frame, output_path="fig_frontier.png")
plot_win_loss_barplot(summary_frame, output_path="fig_labels.png")
```

## GPU / CPU 运行策略

按仓库要求，toolkit 会在运行时优先检查 `torch.cuda.is_available()`：

- 若 GPU 可用，`clean_win` 和 `frontier` 的批量比较会自动走 CUDA tensor。
- 若 GPU 不可用，会无缝退回 CPU，不影响结果生成。
- 图形绘制仍由 `matplotlib` 在 CPU 侧完成。

## 本周产物

- `src/pareto_eval/*.py`
- `experiments/02_pareto_tool_validation/protocol.md`
- `experiments/02_pareto_tool_validation/results/tool_validation_summary.csv`
- `experiments/02_pareto_tool_validation/analysis.md`
- `artifacts/reports/pareto_toolkit_readme.md`

