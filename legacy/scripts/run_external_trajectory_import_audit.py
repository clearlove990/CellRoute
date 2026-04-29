from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch
import h5py
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for candidate in (str(SRC), str(SCRIPTS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hvg_research import SCRNADataset, SCRNAInputSpec, build_default_method_registry, discover_scrna_input_specs, load_scrna_dataset, sanitize_dataset
from hvg_research.data import _read_h5ad_dataframe

import download_external_trajectory_datasets as extdata
import run_real_inputs_round1 as rr1
import run_regime_specific_route as base
import run_trajectory_route_coherence_audit as traj


DEFAULT_OUTPUT_DIR = ROOT / "artifacts_external_trajectory_import_audit"
ANCHOR_METHOD = traj.ANCHOR_METHOD
ROUTE_METHODS = traj.ROUTE_METHODS
ROUTE_FAMILY_LABELS = traj.ROUTE_FAMILY_LABELS
SINGLE_EXPERT_METHODS = (
    "variance",
    "fano",
    "mv_residual",
    "analytic_pearson_residual_hvg",
    "scanpy_cell_ranger_hvg",
    "scanpy_seurat_v3_hvg",
    "triku_hvg",
    "seurat_v3_like_hvg",
    "multinomial_deviance_hvg",
    "adaptive_stat_hvg",
    "adaptive_eb_shrinkage_hvg",
    "adaptive_invariant_residual_hvg",
    "adaptive_spectral_locality_hvg",
)
AUDIT_METHODS = tuple(dict.fromkeys((ANCHOR_METHOD, *SINGLE_EXPERT_METHODS)))
DEFAULT_PLANS: dict[str, rr1.DatasetPlan] = {
    "atlas_mouse_hspc_diff": rr1.DatasetPlan(
        dataset_name="atlas_mouse_hspc_diff",
        max_cells=None,
        max_genes=None,
        mode="full",
        rationale="Small trajectory dataset; safe to audit at full size.",
    ),
    "atlas_mouse_primitive_streak": rr1.DatasetPlan(
        dataset_name="atlas_mouse_primitive_streak",
        max_cells=8000,
        max_genes=6000,
        mode="sampled",
        rationale="Larger trajectory dataset; use a conservative first audit budget.",
    ),
    "atlas_human_hpsc_eht": rr1.DatasetPlan(
        dataset_name="atlas_human_hpsc_eht",
        max_cells=8000,
        max_genes=6000,
        mode="sampled",
        rationale="Dense h5ad matrix; use a memory-safe first audit budget.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an analysis-only import audit on external trajectory-like datasets.")
    parser.add_argument("--external-data-root", type=str, default="data/external_inputs")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--datasets", type=str, default=",".join(extdata.SHORTLIST_SLUGS))
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=2)
    parser.add_argument("--refine-epochs", type=int, default=6)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "device_info.json", resolve_device_info())

    selected_specs = load_selected_specs(
        external_data_root=(ROOT / args.external_data_root).resolve(),
        dataset_selector=args.datasets,
    )
    raw_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []

    for spec in selected_specs:
        plan = DEFAULT_PLANS.get(
            spec.dataset_name,
            rr1.DatasetPlan(
                dataset_name=spec.dataset_name,
                max_cells=None,
                max_genes=None,
                mode="full",
                rationale="No explicit cap registered; audit full dataset.",
            ),
        )
        dataset = load_external_dataset(spec=spec, plan=plan, random_state=args.seed)
        dataset_raw_rows = rr1.run_round1_dataset_benchmark(
            dataset=dataset,
            dataset_id=spec.dataset_id,
            spec=spec,
            method_names=AUDIT_METHODS,
            gate_model_path=None,
            refine_epochs=args.refine_epochs,
            top_k=args.top_k,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        raw_rows.extend(dataset_raw_rows)
        dataset_scored = rr1.add_run_level_scores(pd.DataFrame(dataset_raw_rows))
        dataset_method_summary = rr1.summarize_by_keys(dataset_scored, keys=["dataset", "dataset_id", "method"])
        dataset_method_summary = rr1.rank_within_group(
            dataset_method_summary,
            group_cols=["dataset", "dataset_id"],
            rank_col="dataset_rank",
        )
        summary_rows.append(
            summarize_dataset(
                spec=spec,
                dataset=dataset,
                top_k=args.top_k,
                seed=args.seed,
                refine_epochs=args.refine_epochs,
                method_summary=dataset_method_summary,
            )
        )

    raw_df = pd.DataFrame(raw_rows)
    raw_scored_df = rr1.add_run_level_scores(raw_df)
    raw_scored_df.to_csv(output_dir / "external_benchmark_raw_results.csv", index=False)

    method_summary = rr1.summarize_by_keys(raw_scored_df, keys=["dataset", "dataset_id", "method"])
    method_summary = rr1.rank_within_group(method_summary, group_cols=["dataset", "dataset_id"], rank_col="dataset_rank")
    method_summary.to_csv(output_dir / "external_benchmark_method_summary.csv", index=False)

    audit_df = pd.DataFrame(summary_rows).sort_values(["route_evidence", "dataset_name"]).reset_index(drop=True)
    audit_df.to_csv(output_dir / "external_analysis_audit.csv", index=False)
    (output_dir / "external_analysis_audit.md").write_text(render_audit_markdown(audit_df), encoding="utf-8")
    (output_dir / "route_update_decision.md").write_text(render_route_update(audit_df), encoding="utf-8")


def load_selected_specs(*, external_data_root: Path, dataset_selector: str) -> list[object]:
    selected = extdata.resolve_requested_specs(dataset_selector)
    discovered_by_name = load_external_spec_map(external_data_root)
    missing = [spec.slug for spec in selected if spec.slug not in discovered_by_name]
    if missing:
        raise FileNotFoundError(
            "Missing external dataset inputs. Run "
            f"`python scripts/download_external_trajectory_datasets.py --datasets {','.join(missing)}` first."
        )
    return [discovered_by_name[spec.slug] for spec in selected]


def load_external_spec_map(external_data_root: Path) -> dict[str, SCRNAInputSpec]:
    registry_path = external_data_root / "registry.csv"
    if registry_path.exists():
        registry_df = pd.read_csv(registry_path).fillna("")
        spec_map: dict[str, SCRNAInputSpec] = {}
        for row in registry_df.to_dict(orient="records"):
            input_path = Path(str(row["input_path"]))
            if not input_path.is_absolute():
                input_path = (ROOT / input_path).resolve()
            spec_map[str(row["dataset_name"])] = SCRNAInputSpec(
                dataset_id=str(row["dataset_id"]),
                dataset_name=str(row["dataset_name"]),
                input_path=str(input_path),
                file_format=str(row["file_format"]),
                transpose=str(row.get("transpose", "")).strip().lower() == "true",
                obs_path=None if not str(row.get("obs_path", "")).strip() else str((ROOT / str(row["obs_path"])).resolve()),
                var_path=None,
                genes_path=None if not str(row.get("genes_path", "")).strip() else str((ROOT / str(row["genes_path"])).resolve()),
                cells_path=None if not str(row.get("cells_path", "")).strip() else str((ROOT / str(row["cells_path"])).resolve()),
                labels_col=None if not str(row.get("labels_col", "")).strip() else str(row["labels_col"]),
                batches_col=None if not str(row.get("batches_col", "")).strip() else str(row["batches_col"]),
            )
        return spec_map

    discovered = discover_scrna_input_specs(external_data_root)
    return {spec.dataset_name: spec for spec in discovered}


def load_external_dataset(*, spec, plan: rr1.DatasetPlan, random_state: int) -> SCRNADataset:
    if spec.file_format == "h5ad" and (plan.max_cells is not None or plan.max_genes is not None):
        return load_h5ad_dataset_budgeted(
            path=Path(spec.input_path),
            dataset_name=spec.dataset_name,
            labels_col=spec.labels_col,
            batches_col=spec.batches_col,
            max_cells=plan.max_cells,
            max_genes=plan.max_genes,
            random_state=random_state,
        )
    return load_scrna_dataset(
        data_path=spec.input_path,
        file_format=spec.file_format,
        transpose=spec.transpose,
        obs_path=spec.obs_path,
        var_path=spec.var_path,
        genes_path=spec.genes_path,
        cells_path=spec.cells_path,
        labels_col=spec.labels_col,
        batches_col=spec.batches_col,
        dataset_name=spec.dataset_name,
        max_cells=plan.max_cells,
        max_genes=plan.max_genes,
        random_state=random_state,
    )


def load_h5ad_dataset_budgeted(
    *,
    path: Path,
    dataset_name: str,
    labels_col: str | None,
    batches_col: str | None,
    max_cells: int | None,
    max_genes: int | None,
    random_state: int,
) -> SCRNADataset:
    with h5py.File(path, "r") as handle:
        obs_df = _read_h5ad_dataframe(handle["obs"])
        var_df = _read_h5ad_dataframe(handle["var"])
        cell_names_all = obs_df.index.astype(str).to_numpy()
        gene_names_all = var_df.index.astype(str).to_numpy()
        labels_all = obs_df[labels_col].astype(str).to_numpy() if labels_col and labels_col in obs_df.columns else None
        batches_all = obs_df[batches_col].astype(str).to_numpy() if batches_col and batches_col in obs_df.columns else None

        cell_idx = select_cell_indices(
            n_cells=len(cell_names_all),
            labels=labels_all if labels_all is not None else batches_all,
            max_cells=max_cells,
            random_state=random_state,
        )
        gene_idx = select_gene_indices(
            var_df=var_df,
            n_genes=len(gene_names_all),
            max_genes=max_genes,
        )
        counts = read_h5ad_x_subset(handle["X"], cell_idx=cell_idx, gene_idx=gene_idx)

    dataset = SCRNADataset(
        counts=counts,
        gene_names=gene_names_all[gene_idx],
        cell_names=cell_names_all[cell_idx],
        labels=None if labels_all is None else labels_all[cell_idx],
        batches=None if batches_all is None else batches_all[cell_idx],
        name=dataset_name,
    )
    return sanitize_dataset(dataset)


def select_cell_indices(
    *,
    n_cells: int,
    labels: np.ndarray | None,
    max_cells: int | None,
    random_state: int,
) -> np.ndarray:
    if max_cells is None or n_cells <= max_cells:
        return np.arange(n_cells, dtype=np.int64)
    rng = np.random.default_rng(random_state)
    if labels is None:
        return np.sort(rng.choice(n_cells, size=max_cells, replace=False).astype(np.int64))

    selected: list[np.ndarray] = []
    labels_arr = np.asarray(labels, dtype=object)
    unique_labels, counts = np.unique(labels_arr, return_counts=True)
    remaining = int(max_cells)
    for label, count in sorted(zip(unique_labels, counts, strict=False), key=lambda item: str(item[0])):
        label_idx = np.where(labels_arr == label)[0]
        quota = int(round(max_cells * (int(count) / max(n_cells, 1))))
        quota = max(1, min(quota, int(label_idx.size), remaining))
        chosen = rng.choice(label_idx, size=quota, replace=False)
        selected.append(chosen.astype(np.int64))
        remaining -= quota
        if remaining <= 0:
            break
    combined = np.unique(np.concatenate(selected))
    if combined.size < max_cells:
        missing = np.setdiff1d(np.arange(n_cells, dtype=np.int64), combined, assume_unique=False)
        extra = rng.choice(missing, size=max_cells - combined.size, replace=False)
        combined = np.concatenate([combined, extra.astype(np.int64)])
    if combined.size > max_cells:
        combined = rng.choice(combined, size=max_cells, replace=False)
    return np.sort(combined.astype(np.int64))


def select_gene_indices(*, var_df: pd.DataFrame, n_genes: int, max_genes: int | None) -> np.ndarray:
    if max_genes is None or n_genes <= max_genes:
        return np.arange(n_genes, dtype=np.int64)
    score = np.zeros(n_genes, dtype=np.float64)
    for column, weight in (
        ("n_cells_by_counts", 1.0),
        ("total_counts", 0.20),
        ("mean_counts", 0.15),
    ):
        if column in var_df.columns:
            values = pd.to_numeric(var_df[column], errors="coerce").fillna(0.0).to_numpy(dtype=np.float64)
            finite = np.isfinite(values)
            if int(finite.sum()) > 0 and float(values[finite].max()) > float(values[finite].min()):
                values = (values - values[finite].min()) / max(float(values[finite].max() - values[finite].min()), 1e-12)
            score += weight * values
    if np.allclose(score, 0.0):
        selected = np.linspace(0, n_genes - 1, num=max_genes, dtype=np.int64)
    else:
        selected = np.argsort(score)[-max_genes:]
    return np.sort(selected.astype(np.int64))


def read_h5ad_x_subset(node, *, cell_idx: np.ndarray, gene_idx: np.ndarray) -> np.ndarray:
    if isinstance(node, h5py.Dataset):
        return read_dense_h5ad_subset(node, cell_idx=cell_idx, gene_idx=gene_idx)

    encoding_type = decode_attr(node.attrs.get("encoding-type", ""))
    if encoding_type in {"csr_matrix", "csc_matrix"}:
        data = node["data"][()]
        indices = node["indices"][()]
        indptr = node["indptr"][()]
        shape = tuple(int(x) for x in (node["shape"][()] if "shape" in node else node.attrs["shape"]))
        matrix = sparse.csr_matrix((data, indices, indptr), shape=shape) if encoding_type == "csr_matrix" else sparse.csc_matrix((data, indices, indptr), shape=shape)
        return np.asarray(matrix.tocsr()[cell_idx][:, gene_idx].toarray(), dtype=np.float32)
    raise ValueError(f"Unsupported h5ad X encoding for budgeted read: {encoding_type}")


def read_dense_h5ad_subset(node: h5py.Dataset, *, cell_idx: np.ndarray, gene_idx: np.ndarray) -> np.ndarray:
    output = np.empty((len(cell_idx), len(gene_idx)), dtype=np.float32)
    row_chunk = max(64, min(512, len(cell_idx)))
    for start in range(0, len(cell_idx), row_chunk):
        stop = min(start + row_chunk, len(cell_idx))
        row_idx = cell_idx[start:stop]
        block = np.asarray(node[row_idx, :], dtype=np.float32)
        output[start:stop, :] = block[:, gene_idx]
    return output


def decode_attr(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def summarize_dataset(
    *,
    spec,
    dataset,
    top_k: int,
    seed: int,
    refine_epochs: int,
    method_summary: pd.DataFrame,
) -> dict[str, object]:
    registry = build_default_method_registry(
        top_k=top_k,
        refine_epochs=refine_epochs,
        random_state=seed,
        gate_model_path=None,
    )
    current_top_k = min(int(top_k), int(dataset.counts.shape[1]))
    score_cache: dict[str, np.ndarray] = {}
    topk_cache: dict[str, np.ndarray] = {}
    for method_name in AUDIT_METHODS:
        scores = np.asarray(registry[method_name](dataset.counts, dataset.batches, current_top_k), dtype=np.float64)
        score_cache[method_name] = scores
        topk_cache[method_name] = base.topk_indices(scores, current_top_k)

    summary_by_method = method_summary.set_index("method")

    single_df = method_summary[method_summary["method"].isin(SINGLE_EXPERT_METHODS)].copy()
    best_single_row = single_df.sort_values(
        ["dataset_rank", "overall_rank", "overall_score", "runtime_sec", "method"],
        ascending=[True, True, False, True, True],
    ).iloc[0]
    route_df = method_summary[method_summary["method"].isin(ROUTE_METHODS)].copy()
    best_route_row = route_df.sort_values(
        ["dataset_rank", "overall_rank", "overall_score", "runtime_sec", "method"],
        ascending=[True, True, False, True, True],
    ).iloc[0]
    anchor_row = summary_by_method.loc[ANCHOR_METHOD]

    best_single_method = str(best_single_row["method"])
    best_route_method = str(best_route_row["method"])

    anchor_scores = score_cache[ANCHOR_METHOD]
    anchor_topk = topk_cache[ANCHOR_METHOD]
    best_single_scores = score_cache[best_single_method]
    best_single_topk = topk_cache[best_single_method]
    best_route_scores = score_cache[best_route_method]
    best_route_topk = topk_cache[best_route_method]

    anchor_overlap_to_best = base.jaccard(anchor_topk, best_single_topk)
    route_overlap_to_best = base.jaccard(best_route_topk, best_single_topk)
    anchor_corr_to_best = base.spearman_correlation(anchor_scores, best_single_scores)
    route_corr_to_best = base.spearman_correlation(best_route_scores, best_single_scores)
    route_shift_vs_anchor = 1.0 - base.jaccard(best_route_topk, anchor_topk)
    headroom_vs_best_single = float(best_single_row["overall_score"] - anchor_row["overall_score"])
    route_score_delta_vs_anchor = float(best_route_row["overall_score"] - anchor_row["overall_score"])
    route_overlap_pull = float(route_overlap_to_best - anchor_overlap_to_best)
    route_corr_pull = float(route_corr_to_best - anchor_corr_to_best)

    if best_single_method in ROUTE_METHODS or (
        route_score_delta_vs_anchor >= 0.0 and route_overlap_pull > 0.0 and route_corr_pull >= 0.0
    ):
        route_evidence = "route_supporting_evidence"
        consistency = "in_family_positive"
    elif route_score_delta_vs_anchor >= 0.0 or route_overlap_pull >= 0.0:
        route_evidence = "mixed_or_weak"
        consistency = "weak_or_partial"
    else:
        route_evidence = "route_breaking_counterexample"
        consistency = "outside_family_or_negative"

    return {
        "dataset_name": spec.dataset_name,
        "dataset_id": spec.dataset_id,
        "input_path": spec.input_path,
        "labels_col": spec.labels_col or "",
        "batches_col": spec.batches_col or "",
        "cells_loaded": int(dataset.counts.shape[0]),
        "genes_loaded": int(dataset.counts.shape[1]),
        "best_single_method": best_single_method,
        "best_single_score": float(best_single_row["overall_score"]),
        "best_single_rank": int(best_single_row["dataset_rank"]),
        "best_single_is_route_family": int(best_single_method in ROUTE_METHODS),
        "best_route_method": best_route_method,
        "best_route_score": float(best_route_row["overall_score"]),
        "best_route_rank": int(best_route_row["dataset_rank"]),
        "best_route_family_label": ROUTE_FAMILY_LABELS.get(best_route_method, ""),
        "anchor_method": ANCHOR_METHOD,
        "anchor_score": float(anchor_row["overall_score"]),
        "headroom_vs_best_single": headroom_vs_best_single,
        "route_score_delta_vs_anchor": route_score_delta_vs_anchor,
        "route_overlap_pull_vs_anchor": route_overlap_pull,
        "route_corr_pull_vs_anchor": route_corr_pull,
        "route_shift_vs_anchor": route_shift_vs_anchor,
        "route_evidence": route_evidence,
        "route_family_consistency_judgment": consistency,
    }


def render_audit_markdown(audit_df: pd.DataFrame) -> str:
    lines = [
        "# External Analysis Audit",
        "",
        "## Scope",
        f"- Anchor: `{ANCHOR_METHOD}`",
        f"- Route methods: {', '.join(f'`{name}`' for name in ROUTE_METHODS)}",
        f"- Best-single search space: {', '.join(f'`{name}`' for name in SINGLE_EXPERT_METHODS)}",
        "",
    ]
    for row in audit_df.itertuples(index=False):
        lines.extend(
            [
                f"## {row.dataset_name}",
                f"- Best single expert: `{row.best_single_method}` (score={row.best_single_score:.4f})",
                f"- Best route-family method: `{row.best_route_method}` (score={row.best_route_score:.4f})",
                f"- Anchor score: `{row.anchor_score:.4f}`",
                f"- Headroom vs best single: `{row.headroom_vs_best_single:.4f}`",
                f"- Route overlap pull vs anchor: `{row.route_overlap_pull_vs_anchor:.4f}`",
                f"- Route corr pull vs anchor: `{row.route_corr_pull_vs_anchor:.4f}`",
                f"- Evidence label: `{row.route_evidence}`",
                f"- Consistency judgment: `{row.route_family_consistency_judgment}`",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def render_route_update(audit_df: pd.DataFrame) -> str:
    support_count = int((audit_df["route_evidence"] == "route_supporting_evidence").sum())
    counterexample_count = int((audit_df["route_evidence"] == "route_breaking_counterexample").sum())
    if not audit_df.empty and support_count >= 2 and counterexample_count == 0 and float(audit_df["route_overlap_pull_vs_anchor"].min()) > 0.0:
        decision = "supports_unlock_potential"
        reason = "Multiple external datasets show positive in-family signal without any imported counterexample."
    elif counterexample_count >= max(1, support_count):
        decision = "weakens_route"
        reason = "Imported external evidence is dominated by outside-family or negative route behavior."
    else:
        decision = "supports_hold"
        reason = "External evidence is mixed or still too weak to justify unlocking model design."

    lines = [
        "# Route Update Decision",
        "",
        f"## Decision",
        f"- `{decision}`",
        "",
        "## Reason",
        f"- {reason}",
        "- This is still an analysis-only readout; it does not reopen model design by itself.",
        "",
        "## Dataset Counts",
        f"- route_supporting_evidence={support_count}",
        f"- route_breaking_counterexample={counterexample_count}",
        f"- mixed_or_weak={int((audit_df['route_evidence'] == 'mixed_or_weak').sum())}",
        "",
    ]
    return "\n".join(lines).rstrip() + "\n"


def resolve_device_info() -> dict[str, object]:
    cuda_available = torch.cuda.is_available()
    info: dict[str, object] = {
        "device": "cuda" if cuda_available else "cpu",
        "cuda_available": cuda_available,
        "cuda_count": int(torch.cuda.device_count()) if cuda_available else 0,
    }
    if cuda_available:
        info["cuda_devices"] = [torch.cuda.get_device_name(idx) for idx in range(torch.cuda.device_count())]
    return info


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
