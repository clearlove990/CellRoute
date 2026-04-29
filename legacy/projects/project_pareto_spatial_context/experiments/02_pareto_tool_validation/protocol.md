# Pareto Toolkit Validation Protocol

## 目标

验证 `src/pareto_eval/` 已经足够通用，可以不依赖 Week 1 的一次性脚本，直接从已有 HVG metric table 自动产出：

- clean / near_clean / mixed_tradeoff / dominated；
- Pareto frontier；
- dominance depth；
- markdown report；
- CSV summary；
- key plots。

## 输入

使用 Week 1 已整理好的主表：

- `experiments/01_hvg_tradeoff_audit/results/hvg_tradeoff_master_table.csv`

理由：

- 该表已经统一了 baseline 语义；
- 同时覆盖 global scorer family、risk parity family、rank blend frontier、anchor repair；
- 适合直接测试 toolkit 是否能处理真实多 family、多 dataset、多 evaluation mode 的长表输入。

## 验证步骤

1. 用 `pareto_eval.clean_win.classify_frame(...)` 重算 config-level classification。
2. 把长表 pivot 成宽表 delta matrix。
3. 用 `pareto_eval.frontier.rank_candidates_by_dominance_depth(...)` 计算 family 内和 dataset 内的 Pareto frontier。
4. 对 `anchor_repair` 子集使用 `pareto_eval.replacement_frontier` 复核 budget frontier。
5. 用 `pareto_eval.reporting` 生成 summary CSV 和 markdown analysis。
6. 用 `pareto_eval.plotting` 生成：
   - metric tradeoff heatmap
   - family win/loss barplot
   - rank blend frontier plot
7. 运行轻量 smoke tests，覆盖：
   - clean / near_clean / mixed / dominated 分类
   - frontier membership
   - dominance depth
   - alpha grid
   - replacement frontier

## 通过标准

- `global_scorer_family` clean win 数量为 `0`
- `risk_parity_family` clean win 数量为 `0`
- `rank_blend_frontier` clean win 数量为 `0`
- `anchor_repair` clean win 数量为 `0`
- smoke tests 全部通过

## 运行命令

```powershell
python D:\code_py\sc-RNAseq\project_pareto_spatial_context\experiments\02_pareto_tool_validation\run_validation.py
```

