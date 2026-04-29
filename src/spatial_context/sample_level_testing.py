from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.stats import fisher_exact, hypergeom, t as student_t

from .differential_motif import benjamini_hochberg_qvalues


SAMPLE_LEVEL_METHOD_ORDER: tuple[str, ...] = (
    "naive_fisher",
    "sample_permutation",
    "sample_level_ols_hc3",
    "sample_level_quasi_binomial",
    "sample_permutation_midp",
)


@dataclass(frozen=True)
class SamplePermutationBatchResult:
    observed_effect: np.ndarray
    pvalue_two_sided: np.ndarray
    pvalue_one_sided: np.ndarray
    pvalue_two_sided_midp: np.ndarray
    mode: str
    n_permutations: int
    total_labelings: int


def evaluate_sample_level_methods(
    *,
    motif_ids: np.ndarray,
    sample_positive_counts: np.ndarray,
    sample_totals: np.ndarray,
    labels: np.ndarray,
    condition_a: str,
    condition_b: str,
    fdr_alpha: float,
    sample_permutation_max_permutations: int,
    random_state: int = 7,
    include_midp: bool = False,
) -> pd.DataFrame:
    motif_ids = np.asarray(motif_ids, dtype=object)
    counts = np.asarray(sample_positive_counts, dtype=np.int64)
    totals = np.asarray(sample_totals, dtype=np.int64)
    labels = np.asarray(labels, dtype=object)
    if counts.ndim != 2:
        raise ValueError("sample_positive_counts must be a 2D array of shape (n_samples, n_motifs).")
    if counts.shape[0] != totals.shape[0] or counts.shape[0] != labels.shape[0]:
        raise ValueError("Counts, totals, and labels must have the same number of samples.")
    if counts.shape[1] != motif_ids.shape[0]:
        raise ValueError("Motif ids must align with the motif dimension of sample_positive_counts.")

    fractions = np.divide(
        counts.astype(np.float64, copy=False),
        np.maximum(totals[:, None].astype(np.float64, copy=False), 1.0),
        out=np.zeros(counts.shape, dtype=np.float64),
        where=totals[:, None] > 0,
    )

    naive_frame = _build_naive_fisher_frame(
        motif_ids=motif_ids,
        sample_positive_counts=counts,
        sample_totals=totals,
        labels=labels,
        condition_a=condition_a,
        condition_b=condition_b,
    )
    permutation_frame = _build_sample_permutation_frame(
        motif_ids=motif_ids,
        sample_fractions=fractions,
        labels=labels,
        condition_a=condition_a,
        condition_b=condition_b,
        random_state=int(random_state),
        max_permutations=sample_permutation_max_permutations,
    )
    ols_frame = _build_ols_hc3_frame(
        motif_ids=motif_ids,
        sample_fractions=fractions,
        labels=labels,
        condition_a=condition_a,
        condition_b=condition_b,
    )
    quasi_frame = _build_quasi_binomial_frame(
        motif_ids=motif_ids,
        sample_positive_counts=counts,
        sample_totals=totals,
        labels=labels,
        condition_a=condition_a,
        condition_b=condition_b,
    )

    method_frames = [
        naive_frame,
        permutation_frame.loc[permutation_frame["method"].astype(str) == "sample_permutation"].copy(),
        ols_frame,
        quasi_frame,
    ]
    if include_midp:
        method_frames.append(
            permutation_frame.loc[permutation_frame["method"].astype(str) == "sample_permutation_midp"].copy()
        )

    result = pd.concat(method_frames, ignore_index=True)
    if result.empty:
        return result

    for method_name, method_df in result.groupby("method", observed=False):
        method_index = method_df.index.to_numpy(dtype=np.int64, copy=False)
        pvalues = pd.to_numeric(method_df["pvalue"], errors="coerce").fillna(1.0).to_numpy(dtype=np.float64, copy=False)
        qvalues = benjamini_hochberg_qvalues(pvalues)
        result.loc[method_index, "qvalue"] = qvalues
        result.loc[method_index, "discovery"] = qvalues <= float(fdr_alpha)
        result.loc[method_index, "raw_call"] = pvalues <= float(fdr_alpha)
    result["discovery"] = pd.Series(result["discovery"], dtype="boolean").fillna(False).astype(bool)
    result["raw_call"] = pd.Series(result["raw_call"], dtype="boolean").fillna(False).astype(bool)
    result["method_rank"] = result["method"].map({name: idx for idx, name in enumerate(SAMPLE_LEVEL_METHOD_ORDER)}).fillna(999).astype(int)
    return result.sort_values(["method_rank", "motif_id"]).drop(columns="method_rank").reset_index(drop=True)


def sample_permutation_min_pvalue(n_samples: int, n_case: int, *, midp: bool = False) -> float:
    total_labelings = int(math.comb(int(n_samples), int(n_case)))
    if total_labelings <= 0:
        return float("nan")
    numerator = 0.5 if midp else 1.0
    return float(numerator / total_labelings)


def sample_permutation_total_labelings(n_samples: int, n_case: int) -> int:
    return int(math.comb(int(n_samples), int(n_case)))


def _build_naive_fisher_frame(
    *,
    motif_ids: np.ndarray,
    sample_positive_counts: np.ndarray,
    sample_totals: np.ndarray,
    labels: np.ndarray,
    condition_a: str,
    condition_b: str,
) -> pd.DataFrame:
    mask_a = labels == condition_a
    mask_b = labels == condition_b
    total_a = int(sample_totals[mask_a].sum())
    total_b = int(sample_totals[mask_b].sum())
    rows: list[dict[str, object]] = []
    for motif_index, motif_id in enumerate(motif_ids.tolist()):
        count_a = int(sample_positive_counts[mask_a, motif_index].sum())
        count_b = int(sample_positive_counts[mask_b, motif_index].sum())
        contingency = np.asarray(
            [
                [count_b, max(total_b - count_b, 0)],
                [count_a, max(total_a - count_a, 0)],
            ],
            dtype=np.int64,
        )
        odds_ratio, pvalue = fisher_exact(contingency, alternative="two-sided")
        rows.append(
            {
                "motif_id": str(motif_id),
                "method": "naive_fisher",
                "effect_estimate": float(np.log(max(float(odds_ratio), 1.0e-12))) if np.isfinite(odds_ratio) else np.nan,
                "effect_scale": "log_odds_ratio",
                "pvalue": float(pvalue),
                "permutation_mode": "",
                "n_permutations": np.nan,
                "total_labelings": np.nan,
                "min_attainable_pvalue": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _build_sample_permutation_frame(
    *,
    motif_ids: np.ndarray,
    sample_fractions: np.ndarray,
    labels: np.ndarray,
    condition_a: str,
    condition_b: str,
    random_state: int,
    max_permutations: int,
) -> pd.DataFrame:
    stats = sample_permutation_matrix_statistics(
        values=sample_fractions,
        labels=labels,
        condition_a=condition_a,
        condition_b=condition_b,
        random_state=random_state,
        max_permutations=max_permutations,
    )
    n_samples = int(sample_fractions.shape[0])
    n_case = int(np.sum(labels == condition_b))
    exact_floor = sample_permutation_min_pvalue(n_samples, n_case, midp=False)
    exact_floor_midp = sample_permutation_min_pvalue(n_samples, n_case, midp=True)
    rows: list[dict[str, object]] = []
    for motif_index, motif_id in enumerate(motif_ids.tolist()):
        shared = {
            "motif_id": str(motif_id),
            "effect_estimate": float(stats.observed_effect[motif_index]),
            "effect_scale": "delta_fraction",
            "permutation_mode": str(stats.mode),
            "n_permutations": int(stats.n_permutations),
            "total_labelings": int(stats.total_labelings),
        }
        rows.append(
            {
                **shared,
                "method": "sample_permutation",
                "pvalue": float(stats.pvalue_two_sided[motif_index]),
                "min_attainable_pvalue": exact_floor,
            }
        )
        rows.append(
            {
                **shared,
                "method": "sample_permutation_midp",
                "pvalue": float(stats.pvalue_two_sided_midp[motif_index]),
                "min_attainable_pvalue": exact_floor_midp,
            }
        )
    return pd.DataFrame(rows)


def _build_ols_hc3_frame(
    *,
    motif_ids: np.ndarray,
    sample_fractions: np.ndarray,
    labels: np.ndarray,
    condition_a: str,
    condition_b: str,
) -> pd.DataFrame:
    stats = ols_hc3_statistics(
        values=sample_fractions,
        labels=labels,
        condition_a=condition_a,
        condition_b=condition_b,
    )
    return pd.DataFrame(
        {
            "motif_id": motif_ids.astype(str),
            "method": "sample_level_ols_hc3",
            "effect_estimate": stats["effect_estimate"],
            "effect_scale": "delta_fraction",
            "pvalue": stats["pvalue"],
            "permutation_mode": "",
            "n_permutations": np.nan,
            "total_labelings": np.nan,
            "min_attainable_pvalue": np.nan,
        }
    )


def _build_quasi_binomial_frame(
    *,
    motif_ids: np.ndarray,
    sample_positive_counts: np.ndarray,
    sample_totals: np.ndarray,
    labels: np.ndarray,
    condition_a: str,
    condition_b: str,
) -> pd.DataFrame:
    effects = np.zeros(motif_ids.shape[0], dtype=np.float64)
    pvalues = np.ones(motif_ids.shape[0], dtype=np.float64)
    dispersions = np.full(motif_ids.shape[0], np.nan, dtype=np.float64)
    for motif_index in range(motif_ids.shape[0]):
        stats = quasi_binomial_statistics(
            counts=sample_positive_counts[:, motif_index],
            totals=sample_totals,
            labels=labels,
            condition_a=condition_a,
            condition_b=condition_b,
        )
        effects[motif_index] = float(stats["effect_estimate"])
        pvalues[motif_index] = float(stats["pvalue"])
        dispersions[motif_index] = float(stats["dispersion"])
    return pd.DataFrame(
        {
            "motif_id": motif_ids.astype(str),
            "method": "sample_level_quasi_binomial",
            "effect_estimate": effects,
            "effect_scale": "delta_fraction",
            "pvalue": pvalues,
            "quasi_dispersion": dispersions,
            "permutation_mode": "",
            "n_permutations": np.nan,
            "total_labelings": np.nan,
            "min_attainable_pvalue": np.nan,
        }
    )


def sample_permutation_matrix_statistics(
    *,
    values: np.ndarray,
    labels: np.ndarray,
    condition_a: str,
    condition_b: str,
    random_state: int,
    max_permutations: int,
) -> SamplePermutationBatchResult:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    labels = np.asarray(labels, dtype=object)
    idx_a = np.flatnonzero(labels == condition_a)
    idx_b = np.flatnonzero(labels == condition_b)
    n_features = int(values.shape[1])
    if idx_a.size == 0 or idx_b.size == 0:
        nan_array = np.full(n_features, np.nan, dtype=np.float64)
        return SamplePermutationBatchResult(
            observed_effect=nan_array,
            pvalue_two_sided=nan_array.copy(),
            pvalue_one_sided=nan_array.copy(),
            pvalue_two_sided_midp=nan_array.copy(),
            mode="degenerate",
            n_permutations=0,
            total_labelings=0,
        )

    observed = values[idx_b].mean(axis=0) - values[idx_a].mean(axis=0)
    case_matrix, mode, total_labelings = _build_case_assignment_matrix(
        n_samples=int(values.shape[0]),
        n_case=int(idx_b.size),
        random_state=random_state,
        max_permutations=max_permutations,
    )
    case_sums = case_matrix @ values
    total_sums = values.sum(axis=0, keepdims=True)
    control_sums = total_sums - case_sums
    effects = case_sums / float(idx_b.size) - control_sums / float(idx_a.size)

    tol = 1.0e-12
    abs_observed = np.abs(observed)[None, :]
    abs_effects = np.abs(effects)
    two_strict = np.sum(abs_effects > abs_observed + tol, axis=0, dtype=np.int64)
    two_equal = np.sum(np.abs(abs_effects - abs_observed) <= tol, axis=0, dtype=np.int64)

    positive_mask = observed >= 0.0
    one_strict = np.zeros(n_features, dtype=np.int64)
    one_equal = np.zeros(n_features, dtype=np.int64)
    if np.any(positive_mask):
        one_strict[positive_mask] = np.sum(
            effects[:, positive_mask] > observed[None, positive_mask] + tol,
            axis=0,
            dtype=np.int64,
        )
        one_equal[positive_mask] = np.sum(
            np.abs(effects[:, positive_mask] - observed[None, positive_mask]) <= tol,
            axis=0,
            dtype=np.int64,
        )
    negative_mask = ~positive_mask
    if np.any(negative_mask):
        one_strict[negative_mask] = np.sum(
            effects[:, negative_mask] < observed[None, negative_mask] - tol,
            axis=0,
            dtype=np.int64,
        )
        one_equal[negative_mask] = np.sum(
            np.abs(effects[:, negative_mask] - observed[None, negative_mask]) <= tol,
            axis=0,
            dtype=np.int64,
        )

    n_permutations = int(case_matrix.shape[0])
    if mode == "exact":
        p_two = (two_strict + two_equal).astype(np.float64) / max(n_permutations, 1)
        p_one = (one_strict + one_equal).astype(np.float64) / max(n_permutations, 1)
    else:
        p_two = (two_strict + two_equal + 1.0).astype(np.float64) / float(n_permutations + 1)
        p_one = (one_strict + one_equal + 1.0).astype(np.float64) / float(n_permutations + 1)
    p_two_midp = (two_strict + (0.5 * two_equal)).astype(np.float64) / max(n_permutations, 1)

    return SamplePermutationBatchResult(
        observed_effect=observed.astype(np.float64, copy=False),
        pvalue_two_sided=np.clip(p_two, 0.0, 1.0),
        pvalue_one_sided=np.clip(p_one, 0.0, 1.0),
        pvalue_two_sided_midp=np.clip(p_two_midp, 0.0, 1.0),
        mode=mode,
        n_permutations=n_permutations,
        total_labelings=total_labelings,
    )


def ols_hc3_statistics(
    *,
    values: np.ndarray,
    labels: np.ndarray,
    condition_a: str,
    condition_b: str,
) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values[:, None]
    labels = np.asarray(labels, dtype=object)
    group = np.asarray(labels == condition_b, dtype=np.float64)
    if group.sum() <= 0 or group.sum() >= group.shape[0]:
        nan_array = np.full(values.shape[1], np.nan, dtype=np.float64)
        return {"effect_estimate": nan_array, "pvalue": nan_array.copy()}

    x = np.column_stack([np.ones(group.shape[0], dtype=np.float64), group])
    xtx_inv = np.linalg.pinv(x.T @ x)
    beta = xtx_inv @ x.T @ values
    residuals = values - (x @ beta)
    leverage = np.sum((x @ xtx_inv) * x, axis=1)
    adjusted = residuals / np.clip(1.0 - leverage[:, None], 1.0e-8, None)
    adjusted_sq = adjusted * adjusted

    s00 = adjusted_sq.sum(axis=0)
    s01 = (adjusted_sq * group[:, None]).sum(axis=0)
    s11 = (adjusted_sq * (group[:, None] * group[:, None])).sum(axis=0)

    a10 = float(xtx_inv[1, 0])
    a11 = float(xtx_inv[1, 1])
    variance = (a10 * a10 * s00) + (2.0 * a10 * a11 * s01) + (a11 * a11 * s11)
    variance = np.clip(variance, 0.0, None)
    se = np.sqrt(variance)
    effect = beta[1, :].astype(np.float64, copy=False)
    t_stat = np.divide(effect, se, out=np.zeros_like(effect), where=se > 1.0e-12)
    df = max(int(values.shape[0] - x.shape[1]), 1)
    pvalues = 2.0 * student_t.sf(np.abs(t_stat), df=df)
    pvalues = np.where(se > 1.0e-12, pvalues, np.where(np.abs(effect) > 1.0e-12, 0.0, 1.0))
    return {"effect_estimate": effect, "pvalue": np.clip(pvalues, 0.0, 1.0)}


def quasi_binomial_statistics(
    *,
    counts: np.ndarray,
    totals: np.ndarray,
    labels: np.ndarray,
    condition_a: str,
    condition_b: str,
) -> dict[str, float]:
    counts = np.asarray(counts, dtype=np.float64)
    totals = np.asarray(totals, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    valid = totals > 0
    counts = counts[valid]
    totals = totals[valid]
    labels = labels[valid]
    group = np.asarray(labels == condition_b, dtype=np.float64)
    if group.sum() <= 0 or group.sum() >= group.shape[0]:
        return {"effect_estimate": np.nan, "pvalue": np.nan, "dispersion": np.nan}

    y = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    x = np.column_stack([np.ones(group.shape[0], dtype=np.float64), group])

    mean_a = float(np.clip(y[group == 0.0].mean(), 1.0e-5, 1.0 - 1.0e-5))
    mean_b = float(np.clip(y[group == 1.0].mean(), 1.0e-5, 1.0 - 1.0e-5))
    beta = np.asarray([logit(mean_a), logit(mean_b) - logit(mean_a)], dtype=np.float64)

    for _ in range(64):
        eta = x @ beta
        mu = np.clip(expit(eta), 1.0e-6, 1.0 - 1.0e-6)
        variance = np.clip(mu * (1.0 - mu), 1.0e-6, None)
        weights = np.clip(totals * variance, 1.0e-8, None)
        z = eta + (y - mu) / variance
        xtw = x.T * weights
        xtwx = xtw @ x
        xtwz = xtw @ z
        beta_new = np.linalg.pinv(xtwx) @ xtwz
        beta_new = np.clip(beta_new, -12.0, 12.0)
        if np.max(np.abs(beta_new - beta)) <= 1.0e-8:
            beta = beta_new
            break
        beta = beta_new

    eta = x @ beta
    mu = np.clip(expit(eta), 1.0e-6, 1.0 - 1.0e-6)
    variance = np.clip(mu * (1.0 - mu), 1.0e-6, None)
    weights = np.clip(totals * variance, 1.0e-8, None)
    xtwx = (x.T * weights) @ x
    cov_base = np.linalg.pinv(xtwx)

    pearson = np.sum(np.square(counts - (totals * mu)) / np.clip(totals * variance, 1.0e-8, None))
    df = max(int(group.shape[0] - x.shape[1]), 1)
    dispersion = float(max(pearson / float(df), 1.0e-8))
    covariance = cov_base * dispersion
    se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    coef = float(beta[1])
    if se <= 1.0e-12:
        pvalue = 0.0 if abs(coef) > 1.0e-12 else 1.0
    else:
        t_stat = coef / se
        pvalue = float(2.0 * student_t.sf(abs(t_stat), df=df))
    effect = float(expit(beta[0] + beta[1]) - expit(beta[0]))
    return {
        "effect_estimate": effect,
        "pvalue": float(np.clip(pvalue, 0.0, 1.0)),
        "dispersion": dispersion,
    }


def fisher_mid_pvalue(contingency: np.ndarray) -> float:
    table = np.asarray(contingency, dtype=np.int64)
    if table.shape != (2, 2):
        raise ValueError("Contingency table must be 2x2.")
    total = int(table.sum())
    row_case = int(table[0, :].sum())
    col_positive = int(table[:, 0].sum())
    observed = int(table[0, 0])
    _, exact_p = fisher_exact(table, alternative="two-sided")
    point_mass = float(hypergeom.pmf(observed, total, col_positive, row_case))
    return float(np.clip(exact_p - (0.5 * point_mass), 0.0, 1.0))


def _build_case_assignment_matrix(
    *,
    n_samples: int,
    n_case: int,
    random_state: int,
    max_permutations: int,
) -> tuple[np.ndarray, str, int]:
    total_labelings = int(math.comb(int(n_samples), int(n_case)))
    if total_labelings <= max_permutations:
        case_matrix = np.zeros((total_labelings, int(n_samples)), dtype=np.float64)
        for row_index, case_idx in enumerate(itertools.combinations(range(int(n_samples)), int(n_case))):
            case_matrix[row_index, np.asarray(case_idx, dtype=np.int64)] = 1.0
        return case_matrix, "exact", total_labelings

    rng = np.random.default_rng(random_state)
    case_matrix = np.zeros((int(max_permutations), int(n_samples)), dtype=np.float64)
    all_indices = np.arange(int(n_samples), dtype=np.int64)
    for row_index in range(int(max_permutations)):
        case_idx = rng.choice(all_indices, size=int(n_case), replace=False)
        case_matrix[row_index, case_idx] = 1.0
    return case_matrix, "approx_monte_carlo", total_labelings
