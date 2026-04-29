from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def build_candidate_summary(
    metric_frame: pd.DataFrame,
    classification_frame: pd.DataFrame,
    *,
    group_cols: Sequence[str] = ("config_id", "evaluation_mode"),
    carry_columns: Sequence[str] | None = None,
    sort_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    if metric_frame.empty:
        return pd.DataFrame()

    group_columns = list(group_cols)
    if carry_columns is None:
        preferred_columns = [
            "canonical_config_id",
            "experiment_family",
            "method",
            "display_method",
            "dataset",
            "setting",
            "budget_or_alpha",
            "artifact_path",
            "is_canonical",
        ]
        carry_columns = [column for column in preferred_columns if column in metric_frame.columns and column not in group_columns]

    anchor = metric_frame[group_columns + list(carry_columns)].drop_duplicates(subset=group_columns, keep="first")
    summary = anchor.merge(classification_frame, on=group_columns, how="left", validate="one_to_one")

    if sort_columns is None:
        sort_columns = [
            column
            for column in ("experiment_family", "display_method", "dataset", "budget_or_alpha")
            if column in summary.columns
        ]
    if sort_columns:
        summary = summary.sort_values(list(sort_columns), kind="mergesort").reset_index(drop=True)
    return summary


def build_group_label_summary(
    summary_frame: pd.DataFrame,
    *,
    group_col: str = "experiment_family",
    label_col: str = "pareto_label",
) -> pd.DataFrame:
    if summary_frame.empty:
        return pd.DataFrame()
    grouped = (
        summary_frame.groupby([group_col, label_col], sort=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )
    return grouped


def generate_markdown_summary(
    summary_frame: pd.DataFrame,
    *,
    title: str = "Pareto Summary",
    group_col: str = "experiment_family",
    label_col: str = "pareto_label",
    frontier_flag_col: str = "is_pareto_frontier",
    highlight_columns: Sequence[str] = ("display_method", "dataset", "budget_or_alpha"),
    max_frontier_items: int = 12,
    extra_lines: Sequence[str] | None = None,
) -> str:
    if summary_frame.empty:
        return f"# {title}\n\n- No candidate rows were provided.\n"

    label_order = ["clean_win", "near_clean", "mixed_tradeoff", "dominated", "flat_or_tied"]
    label_counts = summary_frame[label_col].value_counts().reindex(label_order, fill_value=0)

    lines = [
        f"# {title}",
        "",
        "## Overall",
        "",
        f"- total configs: `{int(summary_frame.shape[0])}`",
        f"- clean_win: `{int(label_counts['clean_win'])}`",
        f"- near_clean: `{int(label_counts['near_clean'])}`",
        f"- mixed_tradeoff: `{int(label_counts['mixed_tradeoff'])}`",
        f"- dominated: `{int(label_counts['dominated'])}`",
        f"- flat_or_tied: `{int(label_counts['flat_or_tied'])}`",
    ]

    if group_col in summary_frame.columns:
        grouped = build_group_label_summary(summary_frame, group_col=group_col, label_col=label_col).fillna(0)
        lines.extend(["", "## By Group", ""])
        for row in grouped.itertuples(index=False):
            row_dict = row._asdict()
            lines.append(
                "- "
                f"`{row_dict[group_col]}`: "
                f"clean={int(row_dict.get('clean_win', 0))}, "
                f"near_clean={int(row_dict.get('near_clean', 0))}, "
                f"mixed={int(row_dict.get('mixed_tradeoff', 0))}, "
                f"dominated={int(row_dict.get('dominated', 0))}, "
                f"flat={int(row_dict.get('flat_or_tied', 0))}"
            )

    if frontier_flag_col in summary_frame.columns:
        frontier = summary_frame.loc[summary_frame[frontier_flag_col]].copy()
        frontier = frontier.head(max_frontier_items)
        lines.extend(["", "## Frontier Highlights", ""])
        if frontier.empty:
            lines.append("- No frontier rows were flagged in this summary.")
        else:
            for row in frontier.itertuples(index=False):
                labels = [
                    str(getattr(row, column_name))
                    for column_name in highlight_columns
                    if hasattr(row, column_name)
                ]
                label = " | ".join(labels)
                if hasattr(row, "dominance_depth"):
                    label = f"{label} | depth={int(getattr(row, 'dominance_depth'))}"
                lines.append(f"- `{label}`")

    if extra_lines:
        lines.extend(["", "## Notes", ""])
        lines.extend(list(extra_lines))

    return "\n".join(lines) + "\n"


def generate_csv_summary(
    summary_frame: pd.DataFrame,
    output_path: str | Path,
    *,
    sort_columns: Sequence[str] | None = None,
) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    to_write = summary_frame.copy()
    if sort_columns:
        valid_columns = [column for column in sort_columns if column in to_write.columns]
        if valid_columns:
            to_write = to_write.sort_values(valid_columns, kind="mergesort")
    to_write.to_csv(target_path, index=False)
    return target_path


def write_markdown_summary(markdown_text: str, output_path: str | Path) -> Path:
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(markdown_text, encoding="utf-8")
    return target_path

