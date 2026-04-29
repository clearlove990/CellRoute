from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from scipy.stats import fisher_exact
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .differential_motif import (
    bootstrap_effect_interval,
    choose_condition_order,
    compute_sample_motif_abundance,
    fit_mixed_effect_model,
)
from .motif_embedding import MotifEmbeddingResult
from .neighborhood import NeighborhoodSummary, SpatialDataset


@dataclass(frozen=True)
class HoldoutAssignmentResult:
    spot_table: pd.DataFrame
    fold_summary: pd.DataFrame


@dataclass(frozen=True)
class DifferentialStatsResult:
    summary: pd.DataFrame
    null_controls: pd.DataFrame
    per_sample_effects: pd.DataFrame


def build_sample_level_motif_table(
    *,
    dataset: SpatialDataset,
    spot_table: pd.DataFrame,
    feature_frame: pd.DataFrame,
    expression_program_metadata: pd.DataFrame,
    neighborhood_summary: NeighborhoodSummary,
    analysis_scope: str,
) -> pd.DataFrame:
    abundance = compute_sample_motif_abundance(spot_table)
    if abundance.empty:
        abundance["analysis_scope"] = analysis_scope
        return abundance

    core_scale = neighborhood_summary.core_scale_name
    program_cols = [column for column in feature_frame.columns if column.startswith(f"{core_scale}__program_")]
    entropy_col = f"{core_scale}__entropy"
    density_col = f"{core_scale}__area_density"
    if entropy_col not in feature_frame.columns:
        entropy_col = ""
    if density_col not in feature_frame.columns:
        density_col = ""

    per_spot = spot_table.loc[
        :,
        [
            "dataset_id",
            "dataset_name",
            "sample_id",
            "condition",
            "cell_type",
            "motif_id",
            "motif_label",
        ],
    ].copy()
    per_spot["row_idx"] = np.arange(per_spot.shape[0], dtype=np.int64)
    per_spot["mean_spot_expression"] = np.asarray(dataset.expression.mean(axis=1)).ravel().astype(np.float32, copy=False)
    if "n_counts" in dataset.obs.columns:
        per_spot["mean_n_counts"] = dataset.obs["n_counts"].to_numpy(dtype=np.float32, copy=False)
    else:
        per_spot["mean_n_counts"] = np.nan
    if "n_genes_by_counts" in dataset.obs.columns:
        per_spot["mean_n_genes"] = dataset.obs["n_genes_by_counts"].to_numpy(dtype=np.float32, copy=False)
    else:
        per_spot["mean_n_genes"] = np.nan
    for column in program_cols:
        per_spot[column] = feature_frame[column].to_numpy(dtype=np.float32, copy=False)
    if entropy_col:
        per_spot[entropy_col] = feature_frame[entropy_col].to_numpy(dtype=np.float32, copy=False)
    if density_col:
        per_spot[density_col] = feature_frame[density_col].to_numpy(dtype=np.float32, copy=False)

    group_cols = ["dataset_id", "dataset_name", "sample_id", "condition", "motif_id", "motif_label"]
    positive = per_spot.groupby(group_cols, observed=False)
    summary_parts: list[pd.DataFrame] = []

    mean_columns = ["mean_spot_expression", "mean_n_counts", "mean_n_genes"] + program_cols
    if entropy_col:
        mean_columns.append(entropy_col)
    if density_col:
        mean_columns.append(density_col)
    metric_summary = positive[mean_columns].mean().reset_index()
    rename_map = {
        "mean_spot_expression": "motif_mean_expression",
        "mean_n_counts": "motif_mean_n_counts",
        "mean_n_genes": "motif_mean_n_genes",
    }
    if entropy_col:
        rename_map[entropy_col] = "motif_mean_core_entropy"
    if density_col:
        rename_map[density_col] = "motif_mean_core_density"
    for column in program_cols:
        component = column.rsplit("__", maxsplit=1)[-1]
        rename_map[column] = f"{component}_mean"
    metric_summary = metric_summary.rename(columns=rename_map)
    summary_parts.append(metric_summary)

    dominant_cell_type = (
        per_spot.groupby(group_cols + ["cell_type"], observed=False)
        .size()
        .rename("cell_type_spots")
        .reset_index()
    )
    dominant_cell_type["cell_type_fraction"] = np.divide(
        dominant_cell_type["cell_type_spots"].to_numpy(dtype=np.float64),
        np.maximum(
            dominant_cell_type.groupby(group_cols, observed=False)["cell_type_spots"].transform("sum").to_numpy(dtype=np.float64),
            1.0,
        ),
        out=np.zeros(dominant_cell_type.shape[0], dtype=np.float64),
        where=dominant_cell_type.groupby(group_cols, observed=False)["cell_type_spots"].transform("sum").to_numpy(dtype=np.float64) > 0,
    )
    dominant_cell_type = dominant_cell_type.sort_values(group_cols + ["cell_type_spots", "cell_type"], ascending=[True, True, True, True, True, True, False, True])
    dominant_cell_type = dominant_cell_type.groupby(group_cols, observed=False).tail(1)
    dominant_cell_type = dominant_cell_type.rename(
        columns={
            "cell_type": "dominant_cell_type",
            "cell_type_fraction": "dominant_cell_type_fraction",
        }
    )
    summary_parts.append(dominant_cell_type.loc[:, group_cols + ["dominant_cell_type", "dominant_cell_type_fraction"]])

    if program_cols:
        program_mean_cols = [rename_map[column] for column in program_cols]
        dominant_program = metric_summary.loc[:, group_cols + program_mean_cols].copy()
        dominant_program["dominant_program"] = ""
        dominant_program["dominant_program_score"] = np.nan
        dominant_program["dominant_program_top_genes"] = ""
        program_lookup = (
            expression_program_metadata.set_index("component")["top_genes"].astype(str).to_dict()
            if not expression_program_metadata.empty
            else {}
        )
        for row_index in dominant_program.index:
            values = dominant_program.loc[row_index, program_mean_cols].to_numpy(dtype=np.float64)
            if not np.isfinite(values).any():
                continue
            top_idx = int(np.nanargmax(values))
            component = program_mean_cols[top_idx].replace("_mean", "")
            dominant_program.at[row_index, "dominant_program"] = component
            dominant_program.at[row_index, "dominant_program_score"] = float(values[top_idx])
            dominant_program.at[row_index, "dominant_program_top_genes"] = str(program_lookup.get(component, ""))
        summary_parts.append(
            dominant_program.loc[:, group_cols + ["dominant_program", "dominant_program_score", "dominant_program_top_genes"]]
        )

    colocalization = compute_sample_motif_colocalization(
        spot_table=spot_table,
        adjacency=neighborhood_summary.scales[core_scale].adjacency,
    )
    summary_parts.append(colocalization)

    sample_level = abundance.copy()
    for part in summary_parts:
        sample_level = sample_level.merge(part, on=group_cols, how="left")
    sample_level["analysis_scope"] = analysis_scope
    return sample_level.sort_values(["dataset_id", "sample_id", "motif_id"]).reset_index(drop=True)


def compute_sample_motif_colocalization(
    *,
    spot_table: pd.DataFrame,
    adjacency: sparse.csr_matrix,
) -> pd.DataFrame:
    motifs = spot_table["motif_id"].astype(str).to_numpy()
    sample_ids = spot_table["sample_id"].astype(str).to_numpy()
    rows: list[dict[str, object]] = []
    sample_frame = spot_table.loc[:, ["dataset_id", "dataset_name", "sample_id", "condition", "motif_id", "motif_label"]].copy()
    sample_frame["row_idx"] = np.arange(sample_frame.shape[0], dtype=np.int64)
    for keys, group in sample_frame.groupby(["dataset_id", "dataset_name", "sample_id", "condition", "motif_id", "motif_label"], observed=False):
        row_indices = group["row_idx"].to_numpy(dtype=np.int64, copy=False)
        if row_indices.size == 0:
            continue
        partner_weight: dict[str, float] = {}
        same_fraction: list[float] = []
        for row_idx in row_indices:
            start = int(adjacency.indptr[row_idx])
            end = int(adjacency.indptr[row_idx + 1])
            neighbors = adjacency.indices[start:end]
            weights = adjacency.data[start:end]
            weight_sum = float(weights.sum())
            if weight_sum <= 0.0:
                continue
            neighbor_motifs = motifs[neighbors]
            same_fraction.append(float(weights[neighbor_motifs == keys[4]].sum() / weight_sum))
            for partner, weight in zip(neighbor_motifs.tolist(), weights.tolist(), strict=False):
                partner_weight[str(partner)] = partner_weight.get(str(partner), 0.0) + float(weight)
        partner_candidates = {key: value for key, value in partner_weight.items() if key != str(keys[4])}
        top_partner = max(partner_candidates, key=partner_candidates.get) if partner_candidates else str(keys[4])
        total_partner_weight = float(sum(partner_weight.values()))
        top_partner_fraction = (
            float(partner_weight.get(top_partner, 0.0) / total_partner_weight)
            if total_partner_weight > 0.0
            else np.nan
        )
        rows.append(
            {
                "dataset_id": keys[0],
                "dataset_name": keys[1],
                "sample_id": keys[2],
                "condition": keys[3],
                "motif_id": keys[4],
                "motif_label": keys[5],
                "same_motif_neighbor_fraction": float(np.mean(same_fraction)) if same_fraction else np.nan,
                "top_neighbor_motif": top_partner,
                "top_neighbor_fraction": top_partner_fraction,
            }
        )
    return pd.DataFrame(rows)


def assign_out_of_fold_motifs(
    *,
    dataset: SpatialDataset,
    full_result: MotifEmbeddingResult,
    random_state: int,
) -> HoldoutAssignmentResult:
    feature_frame = full_result.feature_frame
    feature_matrix = feature_frame.to_numpy(dtype=np.float32, copy=False)
    spot_table = full_result.spot_table.copy()
    sample_ids = spot_table["sample_id"].astype(str).to_numpy()
    motif_meta = (
        full_result.spot_table.loc[:, ["motif_id", "motif_label"]]
        .drop_duplicates()
        .sort_values("motif_id")
        .reset_index(drop=True)
    )
    reference_motif_ids = motif_meta["motif_id"].astype(str).tolist()
    reference_label_map = motif_meta.set_index("motif_id")["motif_label"].astype(str).to_dict()
    reference_centroids = (
        pd.DataFrame(feature_matrix, columns=feature_frame.columns)
        .assign(motif_id=full_result.spot_table["motif_id"].astype(str).to_numpy())
        .groupby("motif_id", observed=False)
        .mean(numeric_only=True)
        .reindex(reference_motif_ids)
        .to_numpy(dtype=np.float64)
    )

    oof_table = spot_table.copy()
    fold_rows: list[dict[str, object]] = []
    cluster_dim = int(min(8, max(2, full_result.representation_embedding.shape[1])))
    for fold_index, held_out_sample in enumerate(sorted(np.unique(sample_ids).tolist())):
        test_mask = sample_ids == held_out_sample
        train_mask = ~test_mask
        train_idx = np.flatnonzero(train_mask)
        test_idx = np.flatnonzero(test_mask)
        scaler = StandardScaler()
        train_scaled = scaler.fit_transform(feature_matrix[train_idx]).astype(np.float32, copy=False)
        test_scaled = scaler.transform(feature_matrix[test_idx]).astype(np.float32, copy=False)
        n_pca_components = int(max(2, min(12, train_scaled.shape[0] - 1, train_scaled.shape[1])))
        pca_model = PCA(n_components=n_pca_components, random_state=random_state + fold_index)
        train_embedding = pca_model.fit_transform(train_scaled).astype(np.float32, copy=False)
        test_embedding = pca_model.transform(test_scaled).astype(np.float32, copy=False)

        cluster_model = KMeans(
            n_clusters=int(full_result.n_clusters),
            n_init=20,
            random_state=random_state + fold_index,
        )
        train_cluster_ids = cluster_model.fit_predict(train_embedding[:, :cluster_dim])
        test_cluster_ids = cluster_model.predict(test_embedding[:, :cluster_dim])

        fold_centroids = (
            pd.DataFrame(feature_matrix[train_idx], columns=feature_frame.columns)
            .assign(cluster_id=train_cluster_ids.astype(np.int64, copy=False))
            .groupby("cluster_id", observed=False)
            .mean(numeric_only=True)
            .sort_index()
        )
        cost_matrix = pairwise_squared_distance(
            fold_centroids.to_numpy(dtype=np.float64),
            reference_centroids,
        )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        cluster_to_motif = {
            int(fold_centroids.index[row]): str(reference_motif_ids[col])
            for row, col in zip(row_ind.tolist(), col_ind.tolist(), strict=False)
        }
        mapped_motif_ids = np.asarray([cluster_to_motif[int(cluster_id)] for cluster_id in test_cluster_ids.tolist()], dtype=object)
        mapped_labels = np.asarray([reference_label_map[str(motif_id)] for motif_id in mapped_motif_ids.tolist()], dtype=object)
        oof_table.loc[oof_table.index[test_idx], "motif_id"] = mapped_motif_ids
        oof_table.loc[oof_table.index[test_idx], "motif_label"] = mapped_labels
        full_ids = full_result.spot_table.iloc[test_idx]["motif_id"].astype(str).to_numpy()
        fold_rows.append(
            {
                "sample_id": held_out_sample,
                "n_train_spots": int(train_idx.size),
                "n_test_spots": int(test_idx.size),
                "assignment_agreement_vs_full": float(np.mean(mapped_motif_ids == full_ids)),
                "alignment_cost_mean": float(np.mean(cost_matrix[row_ind, col_ind])),
            }
        )

    return HoldoutAssignmentResult(
        spot_table=oof_table.reset_index(drop=True),
        fold_summary=pd.DataFrame(fold_rows).sort_values("sample_id").reset_index(drop=True),
    )


def compute_differential_statistics(
    *,
    spot_table: pd.DataFrame,
    sample_level_table: pd.DataFrame,
    adjacency: sparse.csr_matrix,
    random_state: int,
    null_iterations: int,
    null_scope_label: str,
    bootstrap_iterations: int = 2000,
    sample_permutation_max_permutations: int = 8192,
    label_max_t_max_permutations: int = 4096,
) -> DifferentialStatsResult:
    if sample_level_table.empty:
        empty = pd.DataFrame()
        return DifferentialStatsResult(summary=empty, null_controls=empty, per_sample_effects=empty)

    result_rows: list[dict[str, object]] = []
    null_rows: list[dict[str, object]] = []
    loso_rows: list[dict[str, object]] = []

    for (dataset_id, dataset_name), dataset_sample_level in sample_level_table.groupby(["dataset_id", "dataset_name"], observed=False):
        conditions = sorted(dataset_sample_level["condition"].astype(str).unique().tolist())
        if len(conditions) != 2:
            continue
        condition_a, condition_b = choose_condition_order(conditions)
        comparison_name = f"{condition_b}_vs_{condition_a}"
        dataset_spot_table = spot_table.loc[spot_table["dataset_id"].astype(str) == str(dataset_id)].copy()
        dataset_indices = dataset_spot_table.index.to_numpy(dtype=np.int64, copy=False)
        dataset_adjacency = adjacency[dataset_indices][:, dataset_indices].tocsr()
        label_max_t = compute_condition_label_max_t_pvalues(
            sample_level_table=dataset_sample_level,
            condition_a=condition_a,
            condition_b=condition_b,
            random_state=random_state,
            max_permutations=label_max_t_max_permutations,
        )
        naive_stats = compute_naive_spot_level_statistics(
            spot_table=dataset_spot_table,
            condition_a=condition_a,
            condition_b=condition_b,
        )
        loso_summary, loso_detail = compute_leave_one_sample_out_summary(
            sample_level_table=dataset_sample_level,
            condition_a=condition_a,
            condition_b=condition_b,
        )
        loso_rows.append(loso_detail)
        null_summary = compute_size_matched_null_controls(
            spot_table=dataset_spot_table,
            sample_level_table=dataset_sample_level,
            adjacency=dataset_adjacency,
            condition_a=condition_a,
            condition_b=condition_b,
            n_iterations=null_iterations,
            random_state=random_state,
            analysis_scope=null_scope_label,
        )
        null_rows.append(null_summary)

        dataset_result_rows: list[dict[str, object]] = []
        for motif_id, motif_df in dataset_sample_level.groupby("motif_id", observed=False):
            motif_label = str(motif_df["motif_label"].iloc[0])
            case_values = motif_df.loc[motif_df["condition"].astype(str) == condition_b, "motif_fraction"].to_numpy(dtype=np.float64)
            ref_values = motif_df.loc[motif_df["condition"].astype(str) == condition_a, "motif_fraction"].to_numpy(dtype=np.float64)
            if case_values.size == 0 or ref_values.size == 0:
                continue
            permutation_stats = exact_sample_permutation_statistics(
                values=motif_df["motif_fraction"].to_numpy(dtype=np.float64),
                labels=motif_df["condition"].astype(str).to_numpy(),
                condition_a=condition_a,
                condition_b=condition_b,
                random_state=random_state,
                max_permutations=sample_permutation_max_permutations,
            )
            ci_low, ci_high = bootstrap_effect_interval(
                ref_values=ref_values,
                case_values=case_values,
                n_iterations=bootstrap_iterations,
                random_state=random_state,
            )
            mixedlm_effect, mixedlm_pvalue, mixedlm_method = fit_mixed_effect_model(
                dataset_spot_table,
                motif_id=str(motif_id),
                condition_a=condition_a,
                condition_b=condition_b,
            )
            dataset_result_rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": dataset_name,
                    "comparison_name": comparison_name,
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "motif_id": str(motif_id),
                    "motif_label": motif_label,
                    "n_samples_a": int(ref_values.size),
                    "n_samples_b": int(case_values.size),
                    "mean_fraction_a": float(ref_values.mean()),
                    "mean_fraction_b": float(case_values.mean()),
                    "delta_fraction": float(case_values.mean() - ref_values.mean()),
                    "log2_fold_change": float(np.log2((case_values.mean() + 1.0e-4) / (ref_values.mean() + 1.0e-4))),
                    "sample_permutation_pvalue_two_sided": permutation_stats["pvalue_two_sided"],
                    "sample_permutation_pvalue_one_sided": permutation_stats["pvalue_one_sided"],
                    "sample_permutation_mode": permutation_stats["mode"],
                    "sample_permutation_n_permutations": permutation_stats["n_permutations"],
                    "sample_permutation_total_labelings": permutation_stats["total_labelings"],
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "mixedlm_effect": mixedlm_effect,
                    "mixedlm_pvalue": mixedlm_pvalue,
                    "mixedlm_method": mixedlm_method,
                }
            )

        dataset_result = pd.DataFrame(dataset_result_rows)
        if dataset_result.empty:
            continue
        dataset_result = dataset_result.merge(label_max_t, on=["dataset_id", "motif_id"], how="left")
        dataset_result = dataset_result.merge(naive_stats, on=["dataset_id", "motif_id"], how="left")
        dataset_result = dataset_result.merge(loso_summary, on=["dataset_id", "motif_id"], how="left")
        dataset_result = dataset_result.merge(
            null_summary.loc[
                null_summary["statistic_name"].astype(str) == "abs_delta_fraction",
                ["dataset_id", "motif_id", "empirical_pvalue", "null_mean", "null_std"],
            ].rename(
                columns={
                    "empirical_pvalue": "synthetic_null_effect_pvalue",
                    "null_mean": "synthetic_null_effect_mean",
                    "null_std": "synthetic_null_effect_std",
                }
            ),
            on=["dataset_id", "motif_id"],
            how="left",
        )
        dataset_result = dataset_result.merge(
            null_summary.loc[
                null_summary["statistic_name"].astype(str) == "self_neighbor_fraction",
                ["dataset_id", "motif_id", "empirical_pvalue", "null_mean", "null_std"],
            ].rename(
                columns={
                    "empirical_pvalue": "synthetic_null_spatial_pvalue",
                    "null_mean": "synthetic_null_spatial_mean",
                    "null_std": "synthetic_null_spatial_std",
                }
            ),
            on=["dataset_id", "motif_id"],
            how="left",
        )
        dataset_result["controlled_support_tier"] = dataset_result.apply(assign_controlled_support_tier, axis=1)
        dataset_result["naive_minus_controlled_log10p"] = (
            -np.log10(np.clip(dataset_result["naive_spot_pvalue"].fillna(1.0).to_numpy(dtype=np.float64), 1.0e-12, 1.0))
            + np.log10(np.clip(dataset_result["sample_permutation_pvalue_two_sided"].fillna(1.0).to_numpy(dtype=np.float64), 1.0e-12, 1.0))
        )
        result_rows.extend(dataset_result.to_dict(orient="records"))

    summary = pd.DataFrame(result_rows)
    if not summary.empty:
        summary = summary.sort_values(
            [
                "dataset_id",
                "sample_permutation_pvalue_two_sided",
                "label_max_t_pvalue",
                "synthetic_null_effect_pvalue",
                "motif_id",
            ],
            ascending=[True, True, True, True, True],
        ).reset_index(drop=True)
    null_controls = pd.concat(null_rows, ignore_index=True) if null_rows else pd.DataFrame()
    per_sample_effects = pd.concat(loso_rows, ignore_index=True) if loso_rows else pd.DataFrame()
    return DifferentialStatsResult(
        summary=summary,
        null_controls=null_controls,
        per_sample_effects=per_sample_effects,
    )


def exact_sample_permutation_statistics(
    *,
    values: np.ndarray,
    labels: np.ndarray,
    condition_a: str,
    condition_b: str,
    random_state: int,
    max_permutations: int,
) -> dict[str, float | int | str]:
    values = np.asarray(values, dtype=np.float64)
    labels = np.asarray(labels, dtype=object)
    idx_a = np.flatnonzero(labels == condition_a)
    idx_b = np.flatnonzero(labels == condition_b)
    if idx_a.size == 0 or idx_b.size == 0:
        return {
            "observed_effect": float("nan"),
            "pvalue_two_sided": float("nan"),
            "pvalue_one_sided": float("nan"),
            "mode": "degenerate",
            "n_permutations": 0,
            "total_labelings": "0",
        }
    observed = float(values[idx_b].mean() - values[idx_a].mean())
    all_indices = np.arange(values.shape[0], dtype=np.int64)
    total_labelings = math.comb(int(values.shape[0]), int(idx_b.size))
    use_exact = total_labelings <= max_permutations
    two_sided_hits = 0.0
    one_sided_hits = 0.0
    n_permutations = 0

    if use_exact:
        for case_idx in itertools.combinations(all_indices.tolist(), idx_b.size):
            case_array = np.asarray(case_idx, dtype=np.int64)
            ref_array = np.setdiff1d(all_indices, case_array, assume_unique=True)
            effect = float(values[case_array].mean() - values[ref_array].mean())
            two_sided_hits += abs(effect) >= abs(observed) - 1.0e-12
            if observed >= 0.0:
                one_sided_hits += effect >= observed - 1.0e-12
            else:
                one_sided_hits += effect <= observed + 1.0e-12
            n_permutations += 1
        two_sided = two_sided_hits / max(n_permutations, 1)
        one_sided = one_sided_hits / max(n_permutations, 1)
        mode = "exact"
    else:
        rng = np.random.default_rng(random_state)
        for _ in range(max_permutations):
            case_array = np.sort(rng.choice(all_indices, size=idx_b.size, replace=False))
            ref_array = np.setdiff1d(all_indices, case_array, assume_unique=True)
            effect = float(values[case_array].mean() - values[ref_array].mean())
            two_sided_hits += abs(effect) >= abs(observed) - 1.0e-12
            if observed >= 0.0:
                one_sided_hits += effect >= observed - 1.0e-12
            else:
                one_sided_hits += effect <= observed + 1.0e-12
            n_permutations += 1
        two_sided = float((two_sided_hits + 1.0) / float(n_permutations + 1))
        one_sided = float((one_sided_hits + 1.0) / float(n_permutations + 1))
        mode = "approx_monte_carlo"
    return {
        "observed_effect": observed,
        "pvalue_two_sided": float(two_sided),
        "pvalue_one_sided": float(one_sided),
        "mode": mode,
        "n_permutations": int(n_permutations),
        "total_labelings": str(total_labelings),
    }


def compute_condition_label_max_t_pvalues(
    *,
    sample_level_table: pd.DataFrame,
    condition_a: str,
    condition_b: str,
    random_state: int,
    max_permutations: int,
) -> pd.DataFrame:
    pivot = (
        sample_level_table.pivot_table(
            index=["sample_id", "condition"],
            columns="motif_id",
            values="motif_fraction",
            fill_value=0.0,
        )
        .sort_index()
        .copy()
    )
    if pivot.empty:
        return pd.DataFrame(
            columns=[
                "dataset_id",
                "motif_id",
                "label_max_t_pvalue",
                "label_max_t_mode",
                "label_max_t_n_permutations",
                "label_max_t_total_labelings",
            ]
        )
    sample_labels = pivot.index.get_level_values("condition").astype(str).to_numpy()
    dataset_id = str(sample_level_table["dataset_id"].iloc[0])
    case_size = int(np.sum(sample_labels == condition_b))
    all_indices = np.arange(pivot.shape[0], dtype=np.int64)
    values = pivot.to_numpy(dtype=np.float64)
    observed = compute_effect_vector(values, sample_labels, condition_a=condition_a, condition_b=condition_b)
    abs_observed = np.abs(observed)
    total_labelings = math.comb(int(pivot.shape[0]), case_size)
    use_exact = total_labelings <= max_permutations
    hit_counts = np.zeros(abs_observed.shape[0], dtype=np.float64)
    n_permutations = 0

    if use_exact:
        for case_idx in itertools.combinations(all_indices.tolist(), case_size):
            permuted = np.asarray([condition_a] * pivot.shape[0], dtype=object)
            permuted[np.asarray(case_idx, dtype=np.int64)] = condition_b
            max_t_value = float(
                np.max(
                    np.abs(
                        compute_effect_vector(
                            values,
                            permuted,
                            condition_a=condition_a,
                            condition_b=condition_b,
                        )
                    )
                )
            )
            hit_counts += max_t_value >= (abs_observed - 1.0e-12)
            n_permutations += 1
        pvalues = hit_counts / max(n_permutations, 1)
        mode = "exact"
    else:
        rng = np.random.default_rng(random_state)
        for _ in range(max_permutations):
            case_array = np.sort(rng.choice(all_indices, size=case_size, replace=False))
            permuted = np.asarray([condition_a] * pivot.shape[0], dtype=object)
            permuted[case_array] = condition_b
            max_t_value = float(
                np.max(
                    np.abs(
                        compute_effect_vector(
                            values,
                            permuted,
                            condition_a=condition_a,
                            condition_b=condition_b,
                        )
                    )
                )
            )
            hit_counts += max_t_value >= (abs_observed - 1.0e-12)
            n_permutations += 1
        pvalues = (hit_counts + 1.0) / float(n_permutations + 1)
        mode = "approx_monte_carlo"

    return pd.DataFrame(
        {
            "dataset_id": dataset_id,
            "motif_id": pivot.columns.astype(str).tolist(),
            "label_max_t_pvalue": pvalues.astype(np.float64, copy=False),
            "label_max_t_mode": mode,
            "label_max_t_n_permutations": int(n_permutations),
            "label_max_t_total_labelings": str(total_labelings),
        }
    )


def compute_naive_spot_level_statistics(
    *,
    spot_table: pd.DataFrame,
    condition_a: str,
    condition_b: str,
) -> pd.DataFrame:
    if spot_table.empty:
        return pd.DataFrame(columns=["dataset_id", "motif_id", "naive_spot_pvalue", "naive_spot_odds_ratio"])
    dataset_id = str(spot_table["dataset_id"].iloc[0])
    total_a = int(np.sum(spot_table["condition"].astype(str) == condition_a))
    total_b = int(np.sum(spot_table["condition"].astype(str) == condition_b))
    rows: list[dict[str, object]] = []
    for motif_id, motif_df in spot_table.groupby("motif_id", observed=False):
        count_a = int(np.sum(motif_df["condition"].astype(str) == condition_a))
        count_b = int(np.sum(motif_df["condition"].astype(str) == condition_b))
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
                "dataset_id": dataset_id,
                "motif_id": str(motif_id),
                "naive_spot_pvalue": float(pvalue),
                "naive_spot_odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else np.nan,
                "naive_spot_fraction_a": float(count_a / max(total_a, 1)),
                "naive_spot_fraction_b": float(count_b / max(total_b, 1)),
            }
        )
    return pd.DataFrame(rows)


def compute_leave_one_sample_out_summary(
    *,
    sample_level_table: pd.DataFrame,
    condition_a: str,
    condition_b: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_id = str(sample_level_table["dataset_id"].iloc[0])
    full_effects = (
        sample_level_table.groupby("motif_id", observed=False)
        .apply(
            lambda group: compute_effect_from_frame(
                group,
                condition_a=condition_a,
                condition_b=condition_b,
            ),
            include_groups=False,
        )
        .rename("full_delta_fraction")
        .reset_index()
    )
    fold_rows: list[dict[str, object]] = []
    sample_ids = sorted(sample_level_table["sample_id"].astype(str).unique().tolist())
    for held_out_sample in sample_ids:
        subset = sample_level_table.loc[sample_level_table["sample_id"].astype(str) != held_out_sample].copy()
        if subset["condition"].astype(str).nunique() < 2:
            continue
        fold_effects = (
            subset.groupby("motif_id", observed=False)
            .apply(
                lambda group: compute_effect_from_frame(
                    group,
                    condition_a=condition_a,
                    condition_b=condition_b,
                ),
                include_groups=False,
            )
            .rename("fold_delta_fraction")
            .reset_index()
        )
        fold_effects["held_out_sample"] = held_out_sample
        fold_effects["dataset_id"] = dataset_id
        fold_rows.append(fold_effects)
    detail = pd.concat(fold_rows, ignore_index=True) if fold_rows else pd.DataFrame(columns=["motif_id", "fold_delta_fraction", "held_out_sample", "dataset_id"])
    if detail.empty:
        summary = pd.DataFrame(columns=["dataset_id", "motif_id", "loso_sign_consistency", "loso_delta_median", "loso_delta_std", "loso_max_abs_shift"])
        return summary, detail
    merged = detail.merge(full_effects, on="motif_id", how="left")
    merged["same_sign"] = merged.apply(
        lambda row: sign_match(float(row["fold_delta_fraction"]), float(row["full_delta_fraction"])),
        axis=1,
    )
    merged["abs_shift"] = np.abs(
        merged["fold_delta_fraction"].to_numpy(dtype=np.float64) - merged["full_delta_fraction"].to_numpy(dtype=np.float64)
    )
    summary = (
        merged.groupby("motif_id", observed=False)
        .agg(
            loso_sign_consistency=("same_sign", "mean"),
            loso_delta_median=("fold_delta_fraction", "median"),
            loso_delta_std=("fold_delta_fraction", "std"),
            loso_max_abs_shift=("abs_shift", "max"),
        )
        .reset_index()
    )
    summary["dataset_id"] = dataset_id
    return summary.loc[:, ["dataset_id", "motif_id", "loso_sign_consistency", "loso_delta_median", "loso_delta_std", "loso_max_abs_shift"]], merged


def compute_size_matched_null_controls(
    *,
    spot_table: pd.DataFrame,
    sample_level_table: pd.DataFrame,
    adjacency: sparse.csr_matrix,
    condition_a: str,
    condition_b: str,
    n_iterations: int,
    random_state: int,
    analysis_scope: str,
) -> pd.DataFrame:
    if spot_table.empty or sample_level_table.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(random_state)
    dataset_id = str(spot_table["dataset_id"].iloc[0])
    dataset_name = str(spot_table["dataset_name"].iloc[0])
    sample_values = spot_table["sample_id"].astype(str).to_numpy()
    sample_categories = pd.Categorical(sample_values)
    sample_codes = np.asarray(sample_categories.codes, dtype=np.int64)
    sample_names = sample_categories.categories.astype(str).tolist()
    sample_sizes = np.bincount(sample_codes, minlength=len(sample_names)).astype(np.float64)
    sample_condition = (
        spot_table.loc[:, ["sample_id", "condition"]]
        .drop_duplicates()
        .set_index("sample_id")["condition"]
        .astype(str)
        .to_dict()
    )
    case_mask = np.asarray([sample_condition[sample_name] == condition_b for sample_name in sample_names], dtype=bool)
    rows: list[dict[str, object]] = []
    total_obs = spot_table.shape[0]
    motif_rows = sample_level_table.loc[:, ["motif_id", "motif_label"]].drop_duplicates().sort_values("motif_id")
    motif_lookup = motif_rows.set_index("motif_id")["motif_label"].astype(str).to_dict()
    for motif_id in motif_rows["motif_id"].astype(str).tolist():
        motif_mask = spot_table["motif_id"].astype(str).to_numpy() == motif_id
        motif_size = int(np.sum(motif_mask))
        if motif_size <= 0 or motif_size >= total_obs:
            continue
        observed_sample = (
            sample_level_table.loc[sample_level_table["motif_id"].astype(str) == motif_id]
            .sort_values("sample_id")
            .set_index("sample_id")
        )
        observed_fractions = np.asarray(
            [float(observed_sample.at[sample_name, "motif_fraction"]) if sample_name in observed_sample.index else 0.0 for sample_name in sample_names],
            dtype=np.float64,
        )
        observed_effect = float(observed_fractions[case_mask].mean() - observed_fractions[~case_mask].mean())
        observed_self_neighbor = float(np.mean(adjacency[motif_mask].dot(motif_mask.astype(np.float32, copy=False)))) if np.any(motif_mask) else np.nan

        null_effects = np.zeros(n_iterations, dtype=np.float64)
        null_self_neighbor = np.zeros(n_iterations, dtype=np.float64)
        for iteration in range(n_iterations):
            sampled_idx = np.sort(rng.choice(total_obs, size=motif_size, replace=False))
            sampled_mask = np.zeros(total_obs, dtype=np.float32)
            sampled_mask[sampled_idx] = 1.0
            sampled_counts = np.bincount(sample_codes[sampled_idx], minlength=len(sample_names)).astype(np.float64)
            sampled_fractions = np.divide(
                sampled_counts,
                np.maximum(sample_sizes, 1.0),
                out=np.zeros_like(sampled_counts, dtype=np.float64),
                where=sample_sizes > 0,
            )
            null_effects[iteration] = float(sampled_fractions[case_mask].mean() - sampled_fractions[~case_mask].mean())
            null_self_neighbor[iteration] = float(np.mean(adjacency[sampled_idx].dot(sampled_mask))) if sampled_idx.size else np.nan

        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "analysis_scope": analysis_scope,
                "motif_id": motif_id,
                "motif_label": motif_lookup.get(motif_id, motif_id),
                "control_name": "size_matched_random_motif",
                "statistic_name": "abs_delta_fraction",
                "observed_value": abs(observed_effect),
                "null_mean": float(np.mean(np.abs(null_effects))),
                "null_std": float(np.std(np.abs(null_effects), ddof=0)),
                "empirical_pvalue": float((np.sum(np.abs(null_effects) >= abs(observed_effect) - 1.0e-12) + 1.0) / (n_iterations + 1.0)),
                "n_iterations": int(n_iterations),
            }
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset_name": dataset_name,
                "analysis_scope": analysis_scope,
                "motif_id": motif_id,
                "motif_label": motif_lookup.get(motif_id, motif_id),
                "control_name": "size_matched_random_motif",
                "statistic_name": "self_neighbor_fraction",
                "observed_value": observed_self_neighbor,
                "null_mean": float(np.nanmean(null_self_neighbor)),
                "null_std": float(np.nanstd(null_self_neighbor, ddof=0)),
                "empirical_pvalue": float((np.sum(null_self_neighbor >= observed_self_neighbor - 1.0e-12) + 1.0) / (n_iterations + 1.0)),
                "n_iterations": int(n_iterations),
            }
        )
    return pd.DataFrame(rows)


def compute_effect_vector(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    condition_a: str,
    condition_b: str,
) -> np.ndarray:
    labels = np.asarray(labels, dtype=object)
    values = np.asarray(values, dtype=np.float64)
    return values[labels == condition_b].mean(axis=0) - values[labels == condition_a].mean(axis=0)


def compute_effect_from_frame(
    frame: pd.DataFrame,
    *,
    condition_a: str,
    condition_b: str,
) -> float:
    values_a = frame.loc[frame["condition"].astype(str) == condition_a, "motif_fraction"].to_numpy(dtype=np.float64)
    values_b = frame.loc[frame["condition"].astype(str) == condition_b, "motif_fraction"].to_numpy(dtype=np.float64)
    if values_a.size == 0 or values_b.size == 0:
        return float("nan")
    return float(values_b.mean() - values_a.mean())


def pairwise_squared_distance(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_sq = np.sum(np.square(left), axis=1, keepdims=True)
    right_sq = np.sum(np.square(right), axis=1, keepdims=True).T
    return np.maximum(left_sq + right_sq - 2.0 * left.dot(right.T), 0.0)


def sign_match(left: float, right: float, *, tol: float = 1.0e-8) -> float:
    if not np.isfinite(left) or not np.isfinite(right):
        return np.nan
    if abs(right) <= tol:
        return float(abs(left) <= tol)
    return float(np.sign(left) == np.sign(right))


def assign_controlled_support_tier(row: pd.Series) -> str:
    sample_p = float(row.get("sample_permutation_pvalue_two_sided", np.nan))
    max_t_p = float(row.get("label_max_t_pvalue", np.nan))
    null_p = float(row.get("synthetic_null_effect_pvalue", np.nan))
    loso = float(row.get("loso_sign_consistency", np.nan))
    naive_p = float(row.get("naive_spot_pvalue", np.nan))
    if sample_p <= 0.20 and max_t_p <= 0.20 and null_p <= 0.20 and loso >= 0.80:
        return "heldout_supported"
    if sample_p <= 0.20 and loso >= 0.80:
        return "sample_level_supported"
    if naive_p <= 1.0e-4 and sample_p > 0.20:
        return "naive_only"
    return "not_supported"
