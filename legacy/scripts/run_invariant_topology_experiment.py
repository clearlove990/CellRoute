from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for candidate in (str(SRC), str(SCRIPTS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hvg_research import SCRNAInputSpec, load_scrna_dataset
from hvg_research.methods import canonicalize_method_names

import run_real_inputs_round1 as rr1


DEFAULT_OUTPUT_DIR = ROOT / "artifacts_invariant_topology_round1"
TARGET_DATASETS = (
    "GBM_sd",
    "cellxgene_human_kidney_nonpt",
    "paul15",
    "cellxgene_mouse_kidney_aging_10x",
    "cellxgene_immune_five_donors",
)
CONTROL_DATASETS = (
    "FBM_cite",
    "E-MTAB-4388",
)
DEFAULT_METHODS = (
    "adaptive_hybrid_hvg",
    "adaptive_stat_hvg",
    "adaptive_spectral_locality_hvg",
    "adaptive_invariant_residual_hvg",
    "invariant_topology_hvg_v0",
    "invariant_topology_hvg_v1",
    "invariant_topology_hvg_v2",
    "variance",
    "multinomial_deviance_hvg",
)
ROLE_BY_DATASET = {
    **{name: "target" for name in TARGET_DATASETS},
    **{name: "control" for name in CONTROL_DATASETS},
}
ANCHOR_METHOD = "adaptive_hybrid_hvg"
EXPERIMENT_PLAN_OVERRIDES = {
    "cellxgene_human_kidney_nonpt": {
        "max_cells": 4000,
        "max_genes": 12000,
        "mode": "budgeted_h5ad",
        "rationale": "Use a budgeted h5ad read to avoid dense full-matrix materialization during the first focused round.",
    },
    "cellxgene_immune_five_donors": {
        "max_cells": 6000,
        "max_genes": 10000,
        "mode": "budgeted_h5ad",
        "rationale": "Keep donor diversity while limiting dense evaluation cost for the initial focused experiment.",
    },
    "cellxgene_mouse_kidney_aging_10x": {
        "max_cells": 8000,
        "max_genes": 8000,
        "mode": "budgeted_h5ad",
        "rationale": "Apply an aggressive first-pass h5ad budget before attempting a wider mouse-kidney run.",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the first focused benchmark for invariant-topology HVG variants.")
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--real-data-root", type=str, default=str(ROOT / "data" / "real_inputs"))
    parser.add_argument("--datasets", type=str, default=",".join((*TARGET_DATASETS, *CONTROL_DATASETS)))
    parser.add_argument("--methods", type=str, default=",".join(DEFAULT_METHODS))
    parser.add_argument("--top-ks", type=str, default="50,200")
    parser.add_argument("--plan-profile", type=str, default="round1", choices=sorted(rr1.PLAN_PROFILES))
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--bootstrap-samples", type=int, default=1)
    parser.add_argument("--refine-epochs", type=int, default=6)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def parse_csv_list(raw: str) -> tuple[str, ...]:
    return tuple(part.strip() for part in raw.split(",") if part.strip())


def parse_int_csv(raw: str) -> tuple[int, ...]:
    return tuple(int(part.strip()) for part in raw.split(",") if part.strip())


def mean_or_nan(values: pd.Series) -> float:
    if values.empty:
        return float("nan")
    return float(values.mean())


def fmt_float(value: object, digits: int = 4) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def save_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def build_manifest(
    *,
    specs: list,
    plan_by_name: dict[str, rr1.DatasetPlan],
    top_ks: tuple[int, ...],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for spec in specs:
        plan = resolve_experiment_plan(spec.dataset_name, plan_by_name[spec.dataset_name])
        rows.append(
            {
                "dataset_name": spec.dataset_name,
                "dataset_id": spec.dataset_id,
                "role": ROLE_BY_DATASET.get(spec.dataset_name, "unassigned"),
                "input_path": spec.input_path,
                "file_format": spec.file_format,
                "labels_col": spec.labels_col or "",
                "batches_col": spec.batches_col or "",
                "plan_mode": plan.mode,
                "plan_max_cells": plan.max_cells,
                "plan_max_genes": plan.max_genes,
                "top_ks": ",".join(str(value) for value in top_ks),
            }
        )
    return pd.DataFrame(rows).sort_values(["role", "dataset_name"]).reset_index(drop=True)


def resolve_experiment_plan(dataset_name: str, base_plan: rr1.DatasetPlan) -> rr1.DatasetPlan:
    override = EXPERIMENT_PLAN_OVERRIDES.get(dataset_name)
    if override is None:
        return base_plan
    return rr1.DatasetPlan(
        dataset_name=base_plan.dataset_name,
        max_cells=override["max_cells"],
        max_genes=override["max_genes"],
        mode=str(override["mode"]),
        rationale=str(override["rationale"]),
    )


def load_specs_from_registry(
    *,
    real_data_root: Path,
    selected_dataset_names: tuple[str, ...],
) -> list[SCRNAInputSpec]:
    registry_path = real_data_root / "registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(f"Registry not found: {registry_path}")
    registry_df = pd.read_csv(registry_path)
    registry_by_name = {str(row["dataset_name"]): row for _, row in registry_df.iterrows()}

    specs: list[SCRNAInputSpec] = []
    missing: list[str] = []
    for dataset_name in sorted(selected_dataset_names):
        def _resolve_optional(path_value: object) -> str | None:
            if path_value is None or pd.isna(path_value) or str(path_value).strip() == "":
                return None
            return str((ROOT / str(path_value)).resolve())

        if dataset_name in registry_by_name:
            row = registry_by_name[dataset_name]
            specs.append(
                SCRNAInputSpec(
                    dataset_id=str(dataset_name),
                    dataset_name=str(dataset_name),
                    input_path=str((ROOT / str(row["input_path"])).resolve()),
                    file_format=str(row["file_format"]),
                    transpose=bool(row["transpose"]),
                    obs_path=_resolve_optional(row.get("obs_path")),
                    genes_path=_resolve_optional(row.get("genes_path")),
                    cells_path=_resolve_optional(row.get("cells_path")),
                    labels_col=None if pd.isna(row.get("labels_col")) else str(row.get("labels_col")),
                    batches_col=None if pd.isna(row.get("batches_col")) else str(row.get("batches_col")),
                )
            )
            continue

        info_path = ROOT / "artifacts_recomb_ismb_benchmark" / "datasets" / dataset_name / "dataset_info.json"
        if info_path.exists():
            info = json.loads(info_path.read_text(encoding="utf-8"))
            specs.append(
                SCRNAInputSpec(
                    dataset_id=str(info.get("dataset_id", dataset_name)),
                    dataset_name=dataset_name,
                    input_path=str(Path(str(info["input_path"])).resolve()),
                    file_format=str(info.get("file_format", "h5ad")),
                    transpose=bool(info.get("transpose", False)),
                    labels_col=None if not info.get("labels_col") else str(info.get("labels_col")),
                    batches_col=None if not info.get("batches_col") else str(info.get("batches_col")),
                )
            )
            continue

        missing.append(dataset_name)

    if missing:
        raise FileNotFoundError(f"Missing registry or artifact rows for: {sorted(missing)}")
    return specs


def build_focus_summary(dataset_summary: pd.DataFrame) -> pd.DataFrame:
    if dataset_summary.empty:
        return pd.DataFrame()

    anchor_lookup = (
        dataset_summary[dataset_summary["method"] == ANCHOR_METHOD][
            ["dataset", "top_k", "overall_score", "stability", "neighbor_preservation"]
        ]
        .rename(
            columns={
                "overall_score": "anchor_overall_score",
                "stability": "anchor_stability",
                "neighbor_preservation": "anchor_neighbor_preservation",
            }
        )
        .copy()
    )
    merged = dataset_summary.merge(anchor_lookup, on=["dataset", "top_k"], how="left")
    merged["dataset_role"] = merged["dataset"].map(ROLE_BY_DATASET).fillna("unassigned")
    merged["delta_vs_anchor"] = merged["overall_score"] - merged["anchor_overall_score"]
    merged["stability_delta_vs_anchor"] = merged["stability"] - merged["anchor_stability"]
    merged["neighbor_delta_vs_anchor"] = (
        merged["neighbor_preservation"] - merged["anchor_neighbor_preservation"]
    )

    rows: list[dict[str, object]] = []
    for (method_name, top_k), group in merged.groupby(["method", "top_k"], sort=True):
        target_group = group[group["dataset_role"] == "target"]
        control_group = group[group["dataset_role"] == "control"]
        rows.append(
            {
                "method": method_name,
                "top_k": int(top_k),
                "target_dataset_count": int(len(target_group)),
                "control_dataset_count": int(len(control_group)),
                "target_mean_overall_score": mean_or_nan(target_group["overall_score"]),
                "control_mean_overall_score": mean_or_nan(control_group["overall_score"]),
                "target_mean_delta_vs_anchor": mean_or_nan(target_group["delta_vs_anchor"]),
                "control_mean_delta_vs_anchor": mean_or_nan(control_group["delta_vs_anchor"]),
                "target_mean_stability_delta": mean_or_nan(target_group["stability_delta_vs_anchor"]),
                "control_mean_stability_delta": mean_or_nan(control_group["stability_delta_vs_anchor"]),
                "target_mean_neighbor_delta": mean_or_nan(target_group["neighbor_delta_vs_anchor"]),
                "control_mean_neighbor_delta": mean_or_nan(control_group["neighbor_delta_vs_anchor"]),
                "target_win_count": int((target_group["dataset_rank"] == 1).sum()),
                "control_win_count": int((control_group["dataset_rank"] == 1).sum()),
            }
        )

    summary = pd.DataFrame(rows)
    if summary.empty:
        return summary
    summary["go_score"] = (
        2.0 * summary["target_mean_delta_vs_anchor"].fillna(0.0)
        - 1.2 * np.maximum(-summary["control_mean_delta_vs_anchor"].fillna(0.0), 0.0)
        + 0.5 * summary["target_mean_stability_delta"].fillna(0.0)
        - 0.2 * np.maximum(-summary["control_mean_stability_delta"].fillna(0.0), 0.0)
    )
    return summary.sort_values(["top_k", "go_score", "method"], ascending=[True, False, True]).reset_index(drop=True)


def render_readout(
    *,
    focus_summary: pd.DataFrame,
    dataset_summary: pd.DataFrame,
) -> str:
    lines = [
        "# Invariant Topology Round 1 Readout",
        "",
        "## Scope",
        f"- Targets: {', '.join(TARGET_DATASETS)}",
        f"- Controls: {', '.join(CONTROL_DATASETS)}",
        f"- Anchor: `{ANCHOR_METHOD}`",
        "",
        "## Top-Line",
    ]
    if focus_summary.empty:
        lines.append("- No successful runs were produced.")
        return "\n".join(lines) + "\n"

    for top_k in sorted(focus_summary["top_k"].unique()):
        top_rows = focus_summary[focus_summary["top_k"] == top_k].copy()
        best_row = top_rows.iloc[0]
        lines.append(
            f"- top_k={top_k}: best focus-score method is `{best_row['method']}`; "
            f"target delta vs anchor={fmt_float(best_row['target_mean_delta_vs_anchor'])}; "
            f"control delta vs anchor={fmt_float(best_row['control_mean_delta_vs_anchor'])}; "
            f"target wins={int(best_row['target_win_count'])}/{int(best_row['target_dataset_count'])}."
        )

    lines.extend(
        [
            "",
            "## Per-Target Winners",
        ]
    )
    for top_k in sorted(dataset_summary["top_k"].unique()):
        subset = dataset_summary[(dataset_summary["top_k"] == top_k) & (dataset_summary["dataset"].isin(TARGET_DATASETS))]
        lines.append(f"- top_k={top_k}:")
        for row in subset.sort_values(["dataset", "dataset_rank"]).groupby("dataset", sort=True).head(1).itertuples(index=False):
            lines.append(
                f"  {row.dataset}: `{row.method}` score={fmt_float(row.overall_score)} rank={int(row.dataset_rank)}."
            )

    lines.extend(
        [
            "",
            "## Control Safety",
        ]
    )
    for top_k in sorted(dataset_summary["top_k"].unique()):
        subset = dataset_summary[(dataset_summary["top_k"] == top_k) & (dataset_summary["dataset"].isin(CONTROL_DATASETS))]
        anchor_subset = subset[subset["method"] == ANCHOR_METHOD].set_index("dataset")
        lines.append(f"- top_k={top_k}:")
        for row in subset[subset["method"].str.startswith("invariant_topology_hvg_")].itertuples(index=False):
            anchor_score = float(anchor_subset.loc[row.dataset, "overall_score"])
            lines.append(
                f"  {row.dataset} / `{row.method}`: delta vs anchor={fmt_float(float(row.overall_score) - anchor_score)}."
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_output_root = output_dir / "datasets"
    dataset_output_root.mkdir(parents=True, exist_ok=True)

    selected_dataset_names = parse_csv_list(args.datasets)
    selected_methods = canonicalize_method_names(parse_csv_list(args.methods))
    top_ks = parse_int_csv(args.top_ks)
    plan_by_name = rr1.PLAN_PROFILES[args.plan_profile]

    device_info = rr1.resolve_device_info()
    save_json(output_dir / "device_info.json", device_info)

    real_data_root = Path(args.real_data_root).resolve()
    specs = load_specs_from_registry(
        real_data_root=real_data_root,
        selected_dataset_names=selected_dataset_names,
    )
    specs = [spec for spec in specs if spec.dataset_name in plan_by_name]
    specs = sorted(specs, key=lambda spec: spec.dataset_name)

    manifest_df = build_manifest(specs=specs, plan_by_name=plan_by_name, top_ks=top_ks)
    manifest_df.to_csv(output_dir / "dataset_manifest.csv", index=False)
    save_json(output_dir / "benchmark_plan.json", manifest_df.to_dict(orient="records"))

    all_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    failure_rows: list[dict[str, object]] = []

    for spec in specs:
        plan = resolve_experiment_plan(spec.dataset_name, plan_by_name[spec.dataset_name])
        dataset_dir = dataset_output_root / spec.dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        try:
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
            dataset_info = {
                "dataset_name": spec.dataset_name,
                "dataset_id": spec.dataset_id,
                "role": ROLE_BY_DATASET.get(spec.dataset_name, "unassigned"),
                "mode": plan.mode,
                "rationale": plan.rationale,
                "cells_loaded": int(dataset.counts.shape[0]),
                "genes_loaded": int(dataset.counts.shape[1]),
                "label_classes_loaded": None if dataset.labels is None else int(len(np.unique(dataset.labels))),
                "batch_classes_loaded": None if dataset.batches is None else int(len(np.unique(dataset.batches))),
            }
            save_json(dataset_dir / "dataset_info.json", dataset_info)

            dataset_rows: list[dict[str, object]] = []
            for top_k in top_ks:
                rows = rr1.run_round1_dataset_benchmark(
                    dataset=dataset,
                    dataset_id=spec.dataset_id,
                    spec=spec,
                    method_names=selected_methods,
                    gate_model_path=None,
                    refine_epochs=args.refine_epochs,
                    top_k=top_k,
                    seed=args.seed,
                    bootstrap_samples=args.bootstrap_samples,
                )
                dataset_rows.extend(rows)

            dataset_df = pd.DataFrame(dataset_rows)
            dataset_df = rr1.add_run_level_scores(dataset_df)
            dataset_df.to_csv(dataset_dir / "raw_results.csv", index=False)

            method_summary = rr1.summarize_by_keys(
                dataset_df,
                keys=["dataset", "dataset_id", "method", "top_k"],
            )
            method_summary = rr1.rank_within_group(
                method_summary,
                group_cols=["dataset", "dataset_id", "top_k"],
                rank_col="dataset_rank",
            )
            method_summary.to_csv(dataset_dir / "method_summary.csv", index=False)

            best_by_topk = (
                method_summary.sort_values(["top_k", "dataset_rank", "overall_score"], ascending=[True, True, False])
                .groupby("top_k", sort=True)
                .head(1)
            )
            status_rows.append(
                {
                    "dataset_name": spec.dataset_name,
                    "role": ROLE_BY_DATASET.get(spec.dataset_name, "unassigned"),
                    "status": "success",
                    "cells_loaded": int(dataset.counts.shape[0]),
                    "genes_loaded": int(dataset.counts.shape[1]),
                    "methods_run": len(selected_methods),
                    "best_methods_by_top_k": "; ".join(
                        f"top_k={int(row.top_k)}:{row.method}" for row in best_by_topk.itertuples(index=False)
                    ),
                }
            )
            all_rows.extend(dataset_df.to_dict(orient="records"))
        except Exception as exc:
            error_payload = {
                "dataset_name": spec.dataset_name,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            save_json(dataset_dir / "error.json", error_payload)
            failure_rows.append(error_payload)
            status_rows.append(
                {
                    "dataset_name": spec.dataset_name,
                    "role": ROLE_BY_DATASET.get(spec.dataset_name, "unassigned"),
                    "status": "failed",
                    "cells_loaded": None,
                    "genes_loaded": None,
                    "methods_run": 0,
                    "best_methods_by_top_k": "",
                }
            )
            if args.fail_fast:
                raise

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(output_dir / "dataset_status.csv", index=False)

    if failure_rows:
        save_json(output_dir / "failures.json", failure_rows)

    if not all_rows:
        readout = render_readout(focus_summary=pd.DataFrame(), dataset_summary=pd.DataFrame())
        (output_dir / "initial_readout.md").write_text(readout, encoding="utf-8")
        print(readout)
        return

    all_results = pd.DataFrame(all_rows)
    all_results.to_csv(output_dir / "benchmark_raw_results.csv", index=False)

    dataset_summary = rr1.summarize_by_keys(
        all_results,
        keys=["dataset", "dataset_id", "method", "top_k"],
    )
    dataset_summary = rr1.rank_within_group(
        dataset_summary,
        group_cols=["dataset", "dataset_id", "top_k"],
        rank_col="dataset_rank",
    )
    dataset_summary.to_csv(output_dir / "benchmark_dataset_summary.csv", index=False)

    global_summary = rr1.summarize_by_keys(
        all_results,
        keys=["method", "top_k"],
    )
    global_summary = global_summary.sort_values(["top_k", "overall_score", "method"], ascending=[True, False, True]).reset_index(drop=True)
    global_summary.to_csv(output_dir / "benchmark_global_summary.csv", index=False)

    focus_summary = build_focus_summary(dataset_summary)
    focus_summary.to_csv(output_dir / "focus_summary.csv", index=False)

    readout = render_readout(
        focus_summary=focus_summary,
        dataset_summary=dataset_summary,
    )
    (output_dir / "initial_readout.md").write_text(readout, encoding="utf-8")
    print(readout)


if __name__ == "__main__":
    main()
