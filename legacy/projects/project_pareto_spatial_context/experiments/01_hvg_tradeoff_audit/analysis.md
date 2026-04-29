# HVG Tradeoff Audit Analysis

## Core Claim

- 审计共覆盖 `48` 个 family-scoped config，对应 `44` 个唯一评估 config。
- Pareto `clean_win` 数量为 `0`；`near_clean` 数量为 `0`；`mixed_tradeoff` 数量为 `43`；`dominated` 数量为 `1`。
- 结论是：在当前 official partial evidence 下，问题更像稳定出现的多目标 tradeoff，而不是单一 scorer 还没调到最优。

## Family Summary

- `anchor_repair`: clean=0, near-clean=0, mixed=8, dominated=0
- `mainline_benchmark`: clean=0, near-clean=0, mixed=12, dominated=0
- `rank_blend_frontier`: clean=0, near-clean=0, mixed=15, dominated=1
- `risk_parity_family`: clean=0, near-clean=0, mixed=12, dominated=0
- risk-parity related configs: `28` mixed / `28` unique configs; spectral-related configs: `11` mixed / `12` unique configs.

## Stable Metric Pattern

- 最常上涨的指标：`nmi` (19/22 positive), `ari` (18/22 positive), `knn_ratio` (17/22 positive)。
- 最常回吐的指标：`lisi` (21/22 negative), `three_nn` (18/22 negative), `max_ari` (15/22 negative)。
- 这说明局部收益并非不存在；真正的问题是收益与 payback 经常绑定出现。

## Interpretation

- `rank_blend_frontier` 没有找到 clean alpha，说明 tradeoff 不是简单由 candidate rank 过猛导致。
- `anchor_repair` 也没有通过小预算替换打开 clean region，说明局部修补同样受限。
- 因此更合理的下一步不是继续单标量加权，而是把问题正式建模成带 protected constraints 的 Pareto repair / context-aware route。
