from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import pandas as pd

from .frontier import rank_candidates_by_dominance_depth
from .metrics import MetricRegistry


def simulate_anchor_replacement_budgets(
    budgets: Sequence[int],
    evaluator: Callable[[int], Mapping[str, Any]],
    *,
    budget_col: str = "budget",
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for budget in budgets:
        payload = dict(evaluator(int(budget)))
        payload[budget_col] = int(budget)
        rows.append(payload)
    return pd.DataFrame(rows)


def extract_replacement_frontier(
    frame: pd.DataFrame,
    registry: MetricRegistry,
    *,
    group_cols: Sequence[str] = ("dataset",),
    budget_col: str = "budget",
    metric_names: Sequence[str] | None = None,
    device: str = "auto",
) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()

    grouped_frames: list[pd.DataFrame] = []
    if group_cols:
        iterator = frame.groupby(list(group_cols), sort=False, dropna=False)
    else:
        iterator = [((), frame)]

    for _, group in iterator:
        ranked = rank_candidates_by_dominance_depth(
            group.copy(),
            registry,
            metric_names=metric_names,
            device=device,
        )
        if budget_col in ranked.columns:
            ranked = ranked.sort_values(
                [budget_col, "dominance_depth", "dominated_by_count"],
                ascending=[True, True, True],
                kind="mergesort",
            )
        grouped_frames.append(ranked)

    return pd.concat(grouped_frames, ignore_index=True) if grouped_frames else frame.copy()


def summarize_replacement_frontier(
    frontier_frame: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("dataset",),
    budget_col: str = "budget",
    frontier_flag_col: str = "is_pareto_frontier",
) -> pd.DataFrame:
    if frontier_frame.empty:
        return frontier_frame.copy()

    if budget_col not in frontier_frame.columns:
        raise KeyError(f"'{budget_col}' is required to summarize replacement frontier budgets.")

    summary_rows: list[dict[str, Any]] = []
    if group_cols:
        iterator = frontier_frame.groupby(list(group_cols), sort=False, dropna=False)
    else:
        iterator = [((), frontier_frame)]

    for group_key, group in iterator:
        frontier_budgets = group.loc[group[frontier_flag_col], budget_col].tolist()
        row: dict[str, Any] = {}
        if group_cols:
            if isinstance(group_key, tuple):
                for column_name, column_value in zip(group_cols, group_key, strict=False):
                    row[column_name] = column_value
            else:
                row[group_cols[0]] = group_key
        row["frontier_budget_count"] = len(frontier_budgets)
        row["frontier_budgets"] = ",".join(str(int(budget)) for budget in frontier_budgets)
        summary_rows.append(row)

    return pd.DataFrame(summary_rows)

