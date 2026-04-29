# Findings

- 统一审计后，`44` 个唯一 official partial config 中 `clean_win=0`，核心负结果成立。
- `mixed_tradeoff` 是主导状态：`43` / `44` unique configs。
- 最常上涨的指标集中在 `nmi, ari, knn_ratio`；最常回吐的指标集中在 `lisi, three_nn, max_ari`。
- `rank_blend` 与 `anchor_repair` 都没能把 direct scorer 的局部收益转成 clean domination，因此应把后续工作重定义为 Pareto-constrained problem，而不是继续纯 scorer tuning。
