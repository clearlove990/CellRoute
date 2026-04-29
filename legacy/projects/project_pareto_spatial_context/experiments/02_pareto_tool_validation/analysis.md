# Pareto Toolkit Validation

## Overall

- total configs: `48`
- clean_win: `0`
- near_clean: `0`
- mixed_tradeoff: `47`
- dominated: `1`
- flat_or_tied: `0`

## By Group

- `anchor_repair`: clean=0, near_clean=0, mixed=8, dominated=0, flat=0
- `global_scorer_family`: clean=0, near_clean=0, mixed=12, dominated=0, flat=0
- `rank_blend_frontier`: clean=0, near_clean=0, mixed=15, dominated=1, flat=0
- `risk_parity_family`: clean=0, near_clean=0, mixed=12, dominated=0, flat=0

## Frontier Highlights

- `anchor_repair | anchor_repair:risk_parity_safe | duo8_pbmc | 25`
- `anchor_repair | anchor_repair:risk_parity_safe | duo8_pbmc | 50`
- `anchor_repair | anchor_repair:risk_parity_safe | duo8_pbmc | 100`
- `anchor_repair | anchor_repair:risk_parity_safe | duo8_pbmc | 200`
- `anchor_repair | anchor_repair:risk_parity_safe | pbmc_cite | 25`
- `anchor_repair | anchor_repair:risk_parity_safe | pbmc_cite | 100`
- `anchor_repair | anchor_repair:risk_parity_safe | pbmc_cite | 200`
- `global_scorer_family | core_consensus | duo8_pbmc | direct`
- `global_scorer_family | core_consensus | pbmc_cite | direct`
- `global_scorer_family | eb_shrinkage | duo8_pbmc | direct`
- `global_scorer_family | eb_shrinkage | pbmc_cite | direct`
- `global_scorer_family | risk_parity | duo4_pbmc | direct`

## Notes

- runtime device preference: `cuda`
- Week 2 validation reruns classification and dominance depth from the reusable toolkit instead of reusing Week 1 one-off logic.
- Validation target: reproduce the Week 1 negative result that no clean win exists in the four candidate families.

## Key Conclusions

- `anchor_repair`: clean=0, near_clean=0, mixed=8, dominated=0
- `global_scorer_family`: clean=0, near_clean=0, mixed=12, dominated=0
- `rank_blend_frontier`: clean=0, near_clean=0, mixed=15, dominated=1
- `risk_parity_family`: clean=0, near_clean=0, mixed=12, dominated=0

## Week 1 Reproduction

- `global_scorer_family` has no clean win.
- `risk_parity_family` has no clean win.
- `rank_blend_frontier` has no clean alpha.
- `anchor_repair` has no clean repair budget.

## Anchor Frontier

- `duo8_pbmc` frontier budgets: `25,50,100,200`
- `pbmc_cite` frontier budgets: `25,100,200`
