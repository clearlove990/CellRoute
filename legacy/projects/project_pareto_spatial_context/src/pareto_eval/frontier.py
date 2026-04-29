from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None

from .metrics import MetricRegistry, get_runtime_device


def _resolve_metric_names(
    registry: MetricRegistry,
    *,
    metric_names: Sequence[str] | None,
    columns: Sequence[str],
) -> list[str]:
    if metric_names is not None:
        resolved = [metric_name for metric_name in metric_names if metric_name in columns]
    else:
        resolved = [metric_name for metric_name in registry.metric_names if metric_name in columns]
    if not resolved:
        raise ValueError("No registered metrics were found in the input frame.")
    return resolved


def _oriented_metric_matrix(
    frame: pd.DataFrame,
    registry: MetricRegistry,
    *,
    metric_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    resolved_metric_names = _resolve_metric_names(registry, metric_names=metric_names, columns=frame.columns.tolist())
    metric_frame = frame[resolved_metric_names].astype(float)
    if metric_frame.isna().any().any():
        raise ValueError("Pareto frontier extraction requires complete metric values without NaN.")

    oriented = metric_frame.copy()
    for metric_name in resolved_metric_names:
        if registry.direction_for(metric_name) == "lower_is_better":
            oriented[metric_name] = -oriented[metric_name]

    epsilon = np.asarray([registry.epsilon_for(metric_name) for metric_name in resolved_metric_names], dtype=np.float64)
    return oriented.to_numpy(dtype=np.float64), epsilon, resolved_metric_names


def pareto_dominates(
    candidate_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
    registry: MetricRegistry,
    *,
    metric_names: Sequence[str] | None = None,
) -> bool:
    shared_columns = list(set(candidate_metrics).intersection(reference_metrics))
    resolved_metric_names = _resolve_metric_names(registry, metric_names=metric_names, columns=shared_columns)

    strictly_better = False
    for metric_name in resolved_metric_names:
        candidate_value = registry.orient_value(metric_name, candidate_metrics[metric_name])
        reference_value = registry.orient_value(metric_name, reference_metrics[metric_name])
        epsilon = registry.epsilon_for(metric_name)
        if candidate_value < reference_value - epsilon:
            return False
        if candidate_value > reference_value + epsilon:
            strictly_better = True
    return strictly_better


def pairwise_dominance_matrix(
    frame: pd.DataFrame,
    registry: MetricRegistry,
    *,
    metric_names: Sequence[str] | None = None,
    device: str = "auto",
) -> tuple[np.ndarray, list[str], str]:
    if frame.empty:
        return np.zeros((0, 0), dtype=bool), [], get_runtime_device()

    matrix, epsilon, resolved_metric_names = _oriented_metric_matrix(frame, registry, metric_names=metric_names)
    runtime_device = get_runtime_device() if device == "auto" else device

    if runtime_device == "cuda" and torch is not None and torch.cuda.is_available():
        tensor = torch.as_tensor(matrix, dtype=torch.float32, device="cuda")
        epsilon_tensor = torch.as_tensor(epsilon, dtype=torch.float32, device="cuda")
        diff = tensor[:, None, :] - tensor[None, :, :]
        not_worse = diff >= (-epsilon_tensor)
        strictly_better = diff > epsilon_tensor
        dominates = torch.all(not_worse, dim=2) & torch.any(strictly_better, dim=2)
        return dominates.detach().cpu().numpy(), resolved_metric_names, runtime_device

    diff = matrix[:, None, :] - matrix[None, :, :]
    not_worse = diff >= (-epsilon[None, None, :])
    strictly_better = diff > epsilon[None, None, :]
    dominates = np.all(not_worse, axis=2) & np.any(strictly_better, axis=2)
    return dominates, resolved_metric_names, runtime_device


def extract_pareto_frontier(
    frame: pd.DataFrame,
    registry: MetricRegistry,
    *,
    metric_names: Sequence[str] | None = None,
    device: str = "auto",
) -> pd.DataFrame:
    annotated = frame.copy()
    dominance, resolved_metric_names, runtime_device = pairwise_dominance_matrix(
        frame,
        registry,
        metric_names=metric_names,
        device=device,
    )
    annotated["dominates_count"] = dominance.sum(axis=1).astype(int)
    annotated["dominated_by_count"] = dominance.sum(axis=0).astype(int)
    annotated["is_pareto_frontier"] = annotated["dominated_by_count"] == 0
    annotated["frontier_runtime_device"] = runtime_device
    annotated["frontier_metric_names"] = ",".join(resolved_metric_names)
    return annotated


def _dominance_depth_from_matrix(dominance: np.ndarray) -> np.ndarray:
    n_items = int(dominance.shape[0])
    depth = np.full(n_items, -1, dtype=int)
    if n_items == 0:
        return depth

    remaining = np.ones(n_items, dtype=bool)
    incoming = dominance.sum(axis=0).astype(int)
    current_depth = 0

    while remaining.any():
        frontier_mask = remaining & (incoming == 0)
        if not frontier_mask.any():
            frontier_mask = remaining & (incoming == incoming[remaining].min())
        depth[frontier_mask] = current_depth
        remaining[frontier_mask] = False
        incoming = incoming - dominance[frontier_mask].sum(axis=0).astype(int)
        incoming[~remaining] = -1
        current_depth += 1

    return depth


def rank_candidates_by_dominance_depth(
    frame: pd.DataFrame,
    registry: MetricRegistry,
    *,
    metric_names: Sequence[str] | None = None,
    device: str = "auto",
) -> pd.DataFrame:
    annotated = extract_pareto_frontier(frame, registry, metric_names=metric_names, device=device)
    dominance, _, _ = pairwise_dominance_matrix(frame, registry, metric_names=metric_names, device=device)
    annotated["dominance_depth"] = _dominance_depth_from_matrix(dominance)
    annotated["dominance_rank"] = annotated["dominance_depth"] + 1
    return annotated.sort_values(
        ["dominance_depth", "dominated_by_count", "dominates_count"],
        ascending=[True, True, False],
        kind="mergesort",
    ).reset_index(drop=True)

