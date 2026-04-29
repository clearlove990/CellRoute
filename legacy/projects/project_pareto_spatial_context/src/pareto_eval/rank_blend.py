from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd


def _validate_alpha(alpha: float) -> float:
    numeric = float(alpha)
    if not 0.0 <= numeric <= 1.0:
        raise ValueError(f"alpha must be within [0, 1], got {alpha}")
    return numeric


def generate_alpha_grid(
    *,
    start: float = 0.0,
    stop: float = 1.0,
    step: float | None = None,
    num: int | None = None,
    values: Sequence[float] | None = None,
    decimals: int = 2,
) -> list[float]:
    if values is not None:
        grid = [_validate_alpha(value) for value in values]
    elif step is not None:
        start = _validate_alpha(start)
        stop = _validate_alpha(stop)
        if step <= 0:
            raise ValueError("step must be positive")
        raw = np.arange(start, stop + (step / 2.0), step, dtype=float)
        grid = [_validate_alpha(value) for value in raw.tolist()]
    else:
        if num is None:
            num = 5
        if num < 2:
            raise ValueError("num must be at least 2 when step is not provided")
        raw = np.linspace(_validate_alpha(start), _validate_alpha(stop), num=num, dtype=float)
        grid = [_validate_alpha(value) for value in raw.tolist()]

    rounded = [round(value, decimals) for value in grid]
    deduped = sorted(dict.fromkeys(rounded))
    return deduped


def interpolate_values(left: Sequence[float], right: Sequence[float], alpha: float) -> np.ndarray:
    weight = _validate_alpha(alpha)
    left_arr = np.asarray(left, dtype=float)
    right_arr = np.asarray(right, dtype=float)
    if left_arr.shape != right_arr.shape:
        raise ValueError(f"left and right must have the same shape, got {left_arr.shape} vs {right_arr.shape}")
    return ((1.0 - weight) * left_arr) + (weight * right_arr)


def interpolate_rank_columns(
    frame: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    alpha: float,
    item_id_col: str | None = None,
    smaller_rank_is_better: bool = True,
    rank_method: str = "dense",
) -> pd.DataFrame:
    weight = _validate_alpha(alpha)
    required_columns = [left_col, right_col]
    if item_id_col is not None:
        required_columns.append(item_id_col)

    missing = [column for column in required_columns if column not in frame.columns]
    if missing:
        raise KeyError(f"Missing columns for rank interpolation: {missing}")

    blended = frame[required_columns].copy()
    blended["alpha"] = weight
    blended["blended_rank_value"] = interpolate_values(frame[left_col], frame[right_col], weight)
    blended["blended_rank"] = blended["blended_rank_value"].rank(
        method=rank_method,
        ascending=smaller_rank_is_better,
    )
    return blended.sort_values("blended_rank", kind="mergesort").reset_index(drop=True)


def build_rank_blend_sweep(
    frame: pd.DataFrame,
    *,
    left_col: str,
    right_col: str,
    alphas: Sequence[float],
    item_id_col: str | None = None,
    smaller_rank_is_better: bool = True,
    rank_method: str = "dense",
) -> pd.DataFrame:
    blended_frames = [
        interpolate_rank_columns(
            frame,
            left_col=left_col,
            right_col=right_col,
            alpha=alpha,
            item_id_col=item_id_col,
            smaller_rank_is_better=smaller_rank_is_better,
            rank_method=rank_method,
        )
        for alpha in alphas
    ]
    if not blended_frames:
        return pd.DataFrame()
    return pd.concat(blended_frames, ignore_index=True)

