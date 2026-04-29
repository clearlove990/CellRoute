# 项目计划

## 项目问题

把已有 `benchmarkHVG` official partial 结果组织成一个清晰的负结果定义：

- 多种 HVG scorer 都能在局部指标上超过 strong `baseline_mix`；
- 但 cleanly dominate 失败；
- 因此问题更像 Pareto tradeoff，而不是单一标量目标还没调到位。

## 本周目标

1. 建立统一项目目录与实验协议。
2. 把四类已有实验汇总为一张统一 master table。
3. 定义更严格的 Pareto clean win / near-clean / mixed tradeoff / dominated 口径。
4. 产出 2-3 张图，说明：
   - 哪些指标经常上涨；
   - 哪些指标经常回吐；
   - tradeoff 是否跨方法稳定存在。

## 输入 artifact

- `artifacts_benchmarkhvg_formal_partial_20260424_v3`
- `artifacts_benchmarkhvg_rank_blend_frontier_20260424_v1`
- `artifacts_benchmarkhvg_anchor_repair_20260425_v1`
- `artifacts_benchmarkhvg_formal_partial_20260424_v1`

## 成功标准

- 一张表能复现“没有 clean win”的事实；
- 图能说明 tradeoff 的稳定模式；
- `findings.md` 能把失败现象转成可继续推进的 Pareto 问题定义。
