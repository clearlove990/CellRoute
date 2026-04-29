from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for candidate in (str(SRC), str(SCRIPTS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hvg_research import discover_scrna_input_specs, load_scrna_dataset
from hvg_research.methods import FRONTIER_LITE_METHOD, canonicalize_method_names

import postprocess_selector_bank_benchmark as pp
import run_real_inputs_round1 as rr1


DEFAULT_SOURCE_DIR = ROOT / "artifacts_recomb_ismb_benchmark"
DEFAULT_OUTPUT_DIR = ROOT / "artifacts_topconf_selector_round2"
DEFAULT_METHODS = (
    "variance",
    "fano",
    "mv_residual",
    "analytic_pearson_residual_hvg",
    "scanpy_cell_ranger_hvg",
    "scanpy_seurat_v3_hvg",
    "adaptive_stat_hvg",
    "adaptive_hybrid_hvg",
    FRONTIER_LITE_METHOD,
    "seurat_v3_like_hvg",
    "multinomial_deviance_hvg",
    "triku_hvg",
    "seurat_r_vst_hvg",
    "scran_model_gene_var_hvg",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build the topconf round-2 benchmark by extending the latest RECOMB/ISMB artifact.")
    parser.add_argument("--source-dir", type=str, default=str(DEFAULT_SOURCE_DIR))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--real-data-root", type=str, default="data/real_inputs")
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=2)
    parser.add_argument("--refine-epochs", type=int, default=6)
    parser.add_argument("--gate-model-path", type=str, default=rr1.DEFAULT_GATE_MODEL_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "datasets").mkdir(parents=True, exist_ok=True)

    source_raw_path = source_dir / "benchmark_raw_results.csv"
    source_manifest_path = source_dir / "dataset_manifest.csv"
    if not source_raw_path.exists():
        raise FileNotFoundError(f"Missing source raw benchmark: {source_raw_path}")
    if not source_manifest_path.exists():
        raise FileNotFoundError(f"Missing source manifest: {source_manifest_path}")

    source_raw = pd.read_csv(source_raw_path)
    source_manifest = pd.read_csv(source_manifest_path)
    source_manifest = source_manifest.sort_values("dataset_name").reset_index(drop=True)
    selected_methods = canonicalize_method_names([part.strip() for part in args.methods.split(",") if part.strip()])
    dataset_names = tuple(source_manifest["dataset_name"].astype(str).tolist())

    real_data_root = (ROOT / args.real_data_root).resolve()
    specs = discover_scrna_input_specs(real_data_root)
    spec_map = {spec.dataset_name: spec for spec in specs if spec.dataset_name in dataset_names}
    missing_specs = sorted(set(dataset_names) - set(spec_map))
    if missing_specs:
        raise FileNotFoundError(f"Missing dataset specs for round2 benchmark: {missing_specs}")

    carryover = source_raw[
        source_raw["dataset"].isin(dataset_names)
        & source_raw["method"].isin(selected_methods)
        & (source_raw["seed"] == int(args.seed))
        & (source_raw["top_k"] == int(args.top_k))
    ].copy()
    if not carryover.empty:
        carryover["result_origin"] = "carryover_recomb_ismb"
        carryover["result_source_dir"] = str(source_dir)

    covered_pairs = {
        (str(row["dataset"]), str(row["method"]))
        for _, row in carryover[["dataset", "method"]].drop_duplicates().iterrows()
    }
    new_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []

    for _, manifest_row in source_manifest.iterrows():
        dataset_name = str(manifest_row["dataset_name"])
        dataset_id = str(manifest_row["dataset_id"])
        missing_methods = [method for method in selected_methods if (dataset_name, method) not in covered_pairs]
        if not missing_methods:
            status_rows.append(
                {
                    "dataset_name": dataset_name,
                    "status": "carryover_only",
                    "mode": manifest_row.get("benchmark_mode", ""),
                    "cells_loaded": _coerce_optional_int(manifest_row.get("benchmark_cells")),
                    "genes_loaded": _coerce_optional_int(manifest_row.get("benchmark_genes")),
                    "methods_run": 0,
                    "methods_carried": int(sum(source_raw["dataset"] == dataset_name)),
                }
            )
            copy_dataset_sidecars(source_dir=source_dir, output_dir=output_dir, dataset_name=dataset_name)
            continue

        spec = spec_map[dataset_name]
        plan = build_plan_from_manifest_row(manifest_row)
        print(
            f"Running new round2 methods for dataset={dataset_name} "
            f"mode={plan.mode} max_cells={plan.max_cells} max_genes={plan.max_genes}: {missing_methods}"
        )
        dataset = load_scrna_dataset(
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
            random_state=args.seed,
        )
        rows = rr1.run_round1_dataset_benchmark(
            dataset=dataset,
            dataset_id=dataset_id,
            spec=spec,
            method_names=tuple(missing_methods),
            gate_model_path=args.gate_model_path,
            refine_epochs=args.refine_epochs,
            top_k=args.top_k,
            seed=args.seed,
            bootstrap_samples=args.bootstrap_samples,
        )
        for row in rows:
            row["result_origin"] = "round2_incremental"
            row["result_source_dir"] = str(output_dir)
        new_rows.extend(rows)
        write_dataset_info(output_dir=output_dir, dataset_name=dataset_name, payload=build_dataset_info(dataset, spec, plan))
        status_rows.append(
            {
                "dataset_name": dataset_name,
                "status": "extended",
                "mode": plan.mode,
                "cells_loaded": int(dataset.counts.shape[0]),
                "genes_loaded": int(dataset.counts.shape[1]),
                "methods_run": len(missing_methods),
                "methods_carried": int(sum(source_raw["dataset"] == dataset_name)),
            }
        )

    combined = pd.concat([carryover, pd.DataFrame(new_rows)], ignore_index=True, sort=False)
    if combined.empty:
        raise RuntimeError("Round2 benchmark assembled no rows.")

    combined = rr1.add_run_level_scores(combined)
    combined.to_csv(output_dir / "benchmark_raw_results.csv", index=False)
    pd.DataFrame(status_rows).sort_values("dataset_name").to_csv(output_dir / "dataset_status.csv", index=False)
    source_manifest.to_csv(output_dir / "dataset_manifest.csv", index=False)
    (output_dir / "benchmark_plan.json").write_text(
        json.dumps(source_manifest.to_dict(orient="records"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    rr1.save_json(output_dir / "device_info.json", rr1.resolve_device_info())

    write_dataset_outputs(output_dir=output_dir, combined=combined, source_dir=source_dir)
    run_postprocess(output_dir=output_dir, combined=combined)
    print(f"Round2 benchmark written to {output_dir}")


def build_plan_from_manifest_row(row: pd.Series) -> rr1.DatasetPlan:
    return rr1.DatasetPlan(
        dataset_name=str(row["dataset_name"]),
        max_cells=_coerce_optional_int(row.get("benchmark_max_cells")),
        max_genes=_coerce_optional_int(row.get("benchmark_max_genes")),
        mode=str(row.get("benchmark_mode", "carryover")),
        rationale=str(row.get("rationale", "Carried from the latest recomb benchmark manifest.")),
    )


def _coerce_optional_int(value) -> int | None:
    if pd.isna(value):
        return None
    return int(value)


def copy_dataset_sidecars(*, source_dir: Path, output_dir: Path, dataset_name: str) -> None:
    source_dataset_dir = source_dir / "datasets" / dataset_name
    target_dataset_dir = output_dir / "datasets" / dataset_name
    target_dataset_dir.mkdir(parents=True, exist_ok=True)
    for filename in ("dataset_info.json", "error.json"):
        source_path = source_dataset_dir / filename
        if source_path.exists():
            shutil.copy2(source_path, target_dataset_dir / filename)


def build_dataset_info(dataset, spec, plan: rr1.DatasetPlan) -> dict[str, object]:
    return {
        "dataset_name": spec.dataset_name,
        "dataset_id": spec.dataset_id,
        "input_path": spec.input_path,
        "file_format": spec.file_format,
        "transpose": spec.transpose,
        "labels_col": spec.labels_col,
        "batches_col": spec.batches_col,
        "mode": plan.mode,
        "rationale": plan.rationale,
        "cells_loaded": int(dataset.counts.shape[0]),
        "genes_loaded": int(dataset.counts.shape[1]),
        "label_classes_loaded": None if dataset.labels is None else int(len(set(dataset.labels.tolist()))),
        "batch_classes_loaded": None if dataset.batches is None else int(len(set(dataset.batches.tolist()))),
    }


def write_dataset_info(*, output_dir: Path, dataset_name: str, payload: dict[str, object]) -> None:
    dataset_dir = output_dir / "datasets" / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)
    rr1.save_json(dataset_dir / "dataset_info.json", payload)


def write_dataset_outputs(*, output_dir: Path, combined: pd.DataFrame, source_dir: Path) -> None:
    for dataset_name, group in combined.groupby("dataset", sort=False):
        dataset_dir = output_dir / "datasets" / str(dataset_name)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        group = group.sort_values(["method", "seed", "top_k"]).reset_index(drop=True)
        group.to_csv(dataset_dir / "raw_results.csv", index=False)
        method_summary = rr1.summarize_by_keys(group, keys=["dataset", "dataset_id", "method"])
        method_summary = rr1.rank_within_group(method_summary, group_cols=["dataset", "dataset_id"], rank_col="dataset_rank")
        method_summary.to_csv(dataset_dir / "method_summary.csv", index=False)
        if not (dataset_dir / "dataset_info.json").exists():
            source_info = source_dir / "datasets" / str(dataset_name) / "dataset_info.json"
            if source_info.exists():
                shutil.copy2(source_info, dataset_dir / "dataset_info.json")


def run_postprocess(*, output_dir: Path, combined: pd.DataFrame) -> None:
    dataset_summary = pp.aggregate_results(combined, keys=["dataset", "dataset_id", "method"])
    dataset_summary = pp.rank_within_group(dataset_summary, group_cols=["dataset", "dataset_id"], rank_col="dataset_rank")
    dataset_summary.to_csv(output_dir / "benchmark_dataset_summary.csv", index=False)

    global_summary = pp.aggregate_results(combined, keys=["method"])
    global_summary = pp.rank_within_group(global_summary, group_cols=[], rank_col="global_rank")
    global_summary["display_name"] = global_summary["method"].map(pp.DISPLAY_NAME).fillna(global_summary["method"])
    global_summary.to_csv(output_dir / "benchmark_global_summary.csv", index=False)

    winners = (
        dataset_summary.sort_values(["dataset", "dataset_rank", "overall_score"], ascending=[True, True, False])
        .groupby(["dataset", "dataset_id"], sort=False, as_index=False)
        .head(1)
        .reset_index(drop=True)
    )
    winners.to_csv(output_dir / "benchmark_dataset_winners.csv", index=False)

    pairwise_summary, pairwise_dataset = pp.build_pairwise_outputs(dataset_summary)
    pairwise_summary.to_csv(output_dir / "pairwise_win_tie_loss_summary.csv", index=False)
    pairwise_dataset.to_csv(output_dir / "pairwise_dataset_deltas.csv", index=False)

    runtime_tradeoff = pp.build_runtime_tradeoff(global_summary)
    runtime_tradeoff.to_csv(output_dir / "runtime_score_tradeoff_summary.csv", index=False)

    taxonomy_df = pp.build_failure_taxonomy(combined, dataset_summary)
    taxonomy_df.to_csv(output_dir / "failure_taxonomy.csv", index=False)

    route_summary = pp.build_route_to_regime_summary(taxonomy_df)
    route_summary.to_csv(output_dir / "route_to_regime_summary.csv", index=False)

    (output_dir / "benchmark_readout.md").write_text(
        pp.build_selector_readout(
            dataset_summary=dataset_summary,
            global_summary=global_summary,
            winners=winners,
            pairwise_summary=pairwise_summary,
            runtime_tradeoff=runtime_tradeoff,
            taxonomy_df=taxonomy_df,
            route_summary=route_summary,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
