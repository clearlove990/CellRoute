from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


@dataclass(frozen=True)
class DifferentialComparison:
    dataset_id: str
    dataset_name: str
    comparison_name: str
    condition_a: str
    condition_b: str


def compute_sample_motif_abundance(
    spot_table: pd.DataFrame,
    *,
    dataset_cols: tuple[str, ...] = ("dataset_id", "dataset_name"),
    sample_col: str = "sample_id",
    condition_col: str = "condition",
    motif_cols: tuple[str, ...] = ("motif_id", "motif_label"),
) -> pd.DataFrame:
    sample_meta = (
        spot_table.loc[:, list(dataset_cols) + [sample_col, condition_col]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    motif_meta = (
        spot_table.loc[:, list(dataset_cols) + list(motif_cols)]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    sample_meta["__merge_key"] = 1
    motif_meta["__merge_key"] = 1
    template = sample_meta.merge(motif_meta, on=list(dataset_cols) + ["__merge_key"], how="inner").drop(columns="__merge_key")
    sample_totals = (
        spot_table.groupby(list(dataset_cols) + [sample_col, condition_col], observed=False)
        .size()
        .rename("sample_spots")
        .reset_index()
    )
    motif_counts = (
        spot_table.groupby(list(dataset_cols) + [sample_col, condition_col] + list(motif_cols), observed=False)
        .size()
        .rename("motif_spots")
        .reset_index()
    )
    abundance = template.merge(
        sample_totals,
        on=list(dataset_cols) + [sample_col, condition_col],
        how="left",
    ).merge(
        motif_counts,
        on=list(dataset_cols) + [sample_col, condition_col] + list(motif_cols),
        how="left",
    )
    abundance["motif_spots"] = abundance["motif_spots"].fillna(0).astype(np.int64)
    abundance["sample_spots"] = abundance["sample_spots"].fillna(0).astype(np.int64)
    abundance["motif_fraction"] = np.divide(
        abundance["motif_spots"].to_numpy(dtype=np.float64),
        np.maximum(abundance["sample_spots"].to_numpy(dtype=np.float64), 1.0),
        out=np.zeros(abundance.shape[0], dtype=np.float64),
        where=abundance["sample_spots"].to_numpy(dtype=np.float64) > 0,
    )
    return abundance.sort_values(list(dataset_cols) + [sample_col, "motif_id"]).reset_index(drop=True)


def differential_motif_analysis(
    spot_table: pd.DataFrame,
    abundance_table: pd.DataFrame,
    *,
    condition_col: str = "condition",
    sample_col: str = "sample_id",
    motif_col: str = "motif_id",
    label_col: str = "motif_label",
    random_state: int = 7,
    bootstrap_iterations: int = 2000,
) -> pd.DataFrame:
    results: list[dict[str, object]] = []
    dataset_groups = abundance_table.groupby(["dataset_id", "dataset_name"], observed=False)
    for (dataset_id, dataset_name), abundance_df in dataset_groups:
        conditions = sorted(abundance_df[condition_col].astype(str).unique().tolist())
        if len(conditions) != 2:
            continue
        condition_a, condition_b = choose_condition_order(conditions)
        comparison_name = f"{condition_b}_vs_{condition_a}"
        spot_subset = spot_table.loc[
            (spot_table["dataset_id"] == dataset_id)
            & (spot_table[condition_col].isin([condition_a, condition_b]))
        ].copy()
        abundance_subset = abundance_df.loc[abundance_df[condition_col].isin([condition_a, condition_b])].copy()
        motif_rows: list[dict[str, object]] = []
        for motif_id, motif_df in abundance_subset.groupby(motif_col, observed=False):
            motif_label = str(motif_df[label_col].iloc[0])
            case_values = motif_df.loc[motif_df[condition_col] == condition_b, "motif_fraction"].to_numpy(dtype=np.float64)
            ref_values = motif_df.loc[motif_df[condition_col] == condition_a, "motif_fraction"].to_numpy(dtype=np.float64)
            if case_values.size == 0 or ref_values.size == 0:
                continue
            delta_fraction = float(case_values.mean() - ref_values.mean())
            log2_fold_change = float(np.log2((case_values.mean() + 1.0e-4) / (ref_values.mean() + 1.0e-4)))
            permutation_pvalue = exact_sample_permutation_pvalue(
                values=motif_df["motif_fraction"].to_numpy(dtype=np.float64),
                labels=motif_df[condition_col].astype(str).to_numpy(),
                condition_a=condition_a,
                condition_b=condition_b,
            )
            bootstrap_low, bootstrap_high = bootstrap_effect_interval(
                ref_values=ref_values,
                case_values=case_values,
                n_iterations=bootstrap_iterations,
                random_state=random_state,
            )
            mixedlm_effect, mixedlm_pvalue, mixedlm_method = fit_mixed_effect_model(
                spot_subset,
                motif_id=motif_id,
                condition_a=condition_a,
                condition_b=condition_b,
                sample_col=sample_col,
                condition_col=condition_col,
                motif_col=motif_col,
            )
            motif_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "comparison_name": comparison_name,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "motif_id": motif_id,
                    "motif_label": motif_label,
                    "n_samples_a": int(ref_values.size),
                    "n_samples_b": int(case_values.size),
                    "mean_fraction_a": float(ref_values.mean()),
                    "mean_fraction_b": float(case_values.mean()),
                    "delta_fraction": delta_fraction,
                    "log2_fold_change": log2_fold_change,
                    "permutation_pvalue": permutation_pvalue,
                    "bootstrap_ci_low": bootstrap_low,
                    "bootstrap_ci_high": bootstrap_high,
                    "mixedlm_effect": mixedlm_effect,
                    "mixedlm_pvalue": mixedlm_pvalue,
                    "mixedlm_method": mixedlm_method,
                }
            )
        if not motif_rows:
            continue
        motif_result_df = pd.DataFrame(motif_rows).sort_values("motif_id").reset_index(drop=True)
        motif_result_df["q_value"] = benjamini_hochberg_qvalues(motif_result_df["mixedlm_pvalue"].fillna(1.0).to_numpy(dtype=np.float64))
        motif_result_df["evidence_tier"] = motif_result_df.apply(assign_evidence_tier, axis=1)
        motif_result_df["association_call"] = motif_result_df["evidence_tier"].isin(["strong", "moderate"])
        results.append(motif_result_df)
    if not results:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "dataset_name",
                "comparison_name",
                "condition_a",
                "condition_b",
                "motif_id",
                "motif_label",
                "n_samples_a",
                "n_samples_b",
                "mean_fraction_a",
                "mean_fraction_b",
                "delta_fraction",
                "log2_fold_change",
                "permutation_pvalue",
                "bootstrap_ci_low",
                "bootstrap_ci_high",
                "mixedlm_effect",
                "mixedlm_pvalue",
                "mixedlm_method",
                "q_value",
                "evidence_tier",
                "association_call",
            ]
        )
    return pd.concat(results, ignore_index=True)


def choose_condition_order(conditions: list[str]) -> tuple[str, str]:
    def score(value: str) -> tuple[int, str]:
        lowered = value.lower()
        if any(keyword in lowered for keyword in ("control", "normal", "responder", "er")):
            return (0, lowered)
        if any(keyword in lowered for keyword in ("ad", "tumor", "tnbc", "non-responder", "disease")):
            return (2, lowered)
        return (1, lowered)

    ordered = sorted(conditions, key=score)
    return ordered[0], ordered[1]


def exact_sample_permutation_pvalue(
    *,
    values: np.ndarray,
    labels: np.ndarray,
    condition_a: str,
    condition_b: str,
) -> float:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    idx_a = np.flatnonzero(labels == condition_a)
    idx_b = np.flatnonzero(labels == condition_b)
    if idx_a.size == 0 or idx_b.size == 0:
        return float("nan")
    observed = float(values[idx_b].mean() - values[idx_a].mean())
    all_indices = np.arange(values.shape[0], dtype=np.int64)
    permutation_effects: list[float] = []
    for case_idx in itertools.combinations(all_indices.tolist(), idx_b.size):
        case_idx_array = np.asarray(case_idx, dtype=np.int64)
        ref_idx_array = np.setdiff1d(all_indices, case_idx_array, assume_unique=True)
        effect = float(values[case_idx_array].mean() - values[ref_idx_array].mean())
        permutation_effects.append(effect)
    permutation_values = np.asarray(permutation_effects, dtype=np.float64)
    return float(np.mean(np.abs(permutation_values) >= abs(observed) - 1.0e-12))


def bootstrap_effect_interval(
    *,
    ref_values: np.ndarray,
    case_values: np.ndarray,
    n_iterations: int = 2000,
    random_state: int = 7,
) -> tuple[float, float]:
    ref_values = np.asarray(ref_values, dtype=np.float64)
    case_values = np.asarray(case_values, dtype=np.float64)
    rng = np.random.default_rng(random_state)
    draws = np.zeros(n_iterations, dtype=np.float64)
    for index in range(n_iterations):
        ref_sample = rng.choice(ref_values, size=ref_values.size, replace=True)
        case_sample = rng.choice(case_values, size=case_values.size, replace=True)
        draws[index] = float(case_sample.mean() - ref_sample.mean())
    lower, upper = np.quantile(draws, [0.025, 0.975])
    return float(lower), float(upper)


def fit_mixed_effect_model(
    spot_table: pd.DataFrame,
    *,
    motif_id: str,
    condition_a: str,
    condition_b: str,
    sample_col: str = "sample_id",
    condition_col: str = "condition",
    motif_col: str = "motif_id",
) -> tuple[float, float, str]:
    model_df = spot_table.loc[spot_table[condition_col].isin([condition_a, condition_b])].copy()
    if model_df.empty:
        return float("nan"), float("nan"), "unavailable"
    model_df["motif_present"] = (model_df[motif_col].astype(str) == str(motif_id)).astype(float)
    model_df["condition_case"] = (model_df[condition_col].astype(str) == condition_b).astype(float)
    if model_df["condition_case"].nunique() < 2:
        return float("nan"), float("nan"), "unavailable"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            mixed_model = smf.mixedlm(
                "motif_present ~ condition_case",
                data=model_df,
                groups=model_df[sample_col].astype(str),
            )
            mixed_result = mixed_model.fit(reml=False, method="lbfgs", disp=False)
            effect = float(mixed_result.params.get("condition_case", np.nan))
            pvalue = float(mixed_result.pvalues.get("condition_case", np.nan))
            if np.isfinite(effect) and np.isfinite(pvalue):
                return effect, pvalue, "mixedlm_random_intercept"
        except Exception:
            pass
        try:
            ols_result = smf.ols("motif_present ~ condition_case", data=model_df).fit(
                cov_type="cluster",
                cov_kwds={"groups": model_df[sample_col].astype(str)},
            )
            effect = float(ols_result.params.get("condition_case", np.nan))
            pvalue = float(ols_result.pvalues.get("condition_case", np.nan))
            return effect, pvalue, "cluster_robust_ols_fallback"
        except Exception:
            return float("nan"), float("nan"), "failed"


def benjamini_hochberg_qvalues(pvalues: np.ndarray) -> np.ndarray:
    pvalues = np.asarray(pvalues, dtype=np.float64)
    n = pvalues.shape[0]
    if n == 0:
        return pvalues
    order = np.argsort(pvalues)
    ranked = pvalues[order]
    qvalues = np.empty(n, dtype=np.float64)
    running = 1.0
    for rank in range(n, 0, -1):
        index = rank - 1
        candidate = ranked[index] * n / rank
        running = min(running, candidate)
        qvalues[index] = running
    result = np.empty(n, dtype=np.float64)
    result[order] = np.clip(qvalues, 0.0, 1.0)
    return result


def assign_evidence_tier(row: pd.Series) -> str:
    mixedlm_pvalue = float(row.get("mixedlm_pvalue", np.nan))
    permutation_pvalue = float(row.get("permutation_pvalue", np.nan))
    q_value = float(row.get("q_value", np.nan))
    ci_low = float(row.get("bootstrap_ci_low", np.nan))
    ci_high = float(row.get("bootstrap_ci_high", np.nan))
    ci_excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
    if q_value <= 0.10 and ci_excludes_zero:
        return "strong"
    if ci_excludes_zero and (permutation_pvalue <= 0.20 or mixedlm_pvalue <= 0.05):
        return "moderate"
    if mixedlm_pvalue <= 0.10 or permutation_pvalue <= 0.20:
        return "weak"
    return "none"
