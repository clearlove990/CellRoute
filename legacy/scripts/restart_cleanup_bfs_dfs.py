from __future__ import annotations

import argparse
import csv
import os
import shutil
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
for candidate in (str(SRC), str(SCRIPTS)):
    if candidate not in sys.path:
        sys.path.insert(0, candidate)

from hvg_research import build_default_method_registry

import build_topconf_selector_round2_package as pkg
import run_real_inputs_round1 as rr1


OUTPUT_DIR = ROOT / "artifacts_restart_cleanup_bfs_dfs"
MAINLINE_DIR_TEMPLATE = "mainline_{idea_name}"
SAFE_ANCHOR_METHOD = "adaptive_hybrid_hvg"
DEFAULT_BFS_SHORTLIST = (
    "adaptive_core_consensus_hvg",
    "adaptive_rank_aggregate_hvg",
)
REPRESENTATIVE_POSITIVE_DATASETS = (
    "GBM_sd",
    "cellxgene_human_kidney_nonpt",
    "paul15",
    "cellxgene_immune_five_donors",
)
REPRESENTATIVE_CONTROL_DATASETS = (
    "mus_tissue",
    "homo_tissue",
)
REPRESENTATIVE_SMOKE_DATASETS = REPRESENTATIVE_POSITIVE_DATASETS + REPRESENTATIVE_CONTROL_DATASETS
TEXT_SUFFIXES = {
    ".csv",
    ".json",
    ".md",
    ".py",
    ".R",
    ".txt",
    ".yml",
    ".yaml",
}
KEEP_REFERENCE_PATHS = (
    ROOT / "README.md",
    ROOT / "scripts",
    ROOT / "src",
    ROOT / "artifacts_next_direction",
    ROOT / "artifacts_topconf_selector_round2",
    ROOT / "artifacts_codex_selector_mvp",
    ROOT / "artifacts_codex_selector_phase2",
    ROOT / "artifacts_codex_selector_phase3_official_repro",
)
MANDATORY_KEEP_PATHS = {
    ROOT / "src",
    ROOT / "data",
    ROOT / "README.md",
    ROOT / "requirements.txt",
    ROOT / "scripts",
    ROOT / "artifacts_topconf_selector_round2",
    ROOT / "artifacts_codex_selector_mvp",
    ROOT / "artifacts_codex_selector_phase2",
    ROOT / "artifacts_codex_selector_phase3_official_repro",
    ROOT / "artifacts_next_direction",
    ROOT / "artifacts_gate_learning_v_next9_multireal",
    ROOT / "artifacts_recomb_ismb_benchmark",
}


@dataclass(frozen=True)
class ManifestRow:
    path: Path
    entry_type: str
    current_size_bytes: int
    regenerable: bool
    referenced_by_kept_logic: bool
    rationale: str
    risk: str
    status: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ordered cleanup -> BFS -> smoke -> DFS for the next anchor-core experiment.")
    parser.add_argument("--phase", choices=("cleanup", "bfs_smoke", "dfs", "all"), default="all")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR))
    parser.add_argument("--real-data-root", type=str, default="data/real_inputs")
    parser.add_argument("--gate-model-path", type=str, default=rr1.DEFAULT_GATE_MODEL_PATH)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=2)
    parser.add_argument("--refine-epochs", type=int, default=6)
    parser.add_argument("--mainline-method", type=str, default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase in {"cleanup", "all"}:
        run_cleanup_phase(output_dir=output_dir)

    if args.phase in {"bfs_smoke", "all"}:
        smoke_selection = run_bfs_and_smoke_phase(
            output_dir=output_dir,
            real_data_root=(ROOT / args.real_data_root).resolve(),
            gate_model_path=str(Path(args.gate_model_path).resolve()),
            top_k=int(args.top_k),
            seed=int(args.seed),
            bootstrap_samples=int(args.bootstrap_samples),
            refine_epochs=int(args.refine_epochs),
        )
        if smoke_selection:
            (output_dir / "smoke_selected_method.txt").write_text(smoke_selection, encoding="utf-8")

    if args.phase in {"dfs", "all"}:
        selected_method = args.mainline_method.strip()
        if not selected_method:
            selected_path = output_dir / "smoke_selected_method.txt"
            if selected_path.exists():
                selected_method = selected_path.read_text(encoding="utf-8").strip()
        if selected_method:
            run_dfs_phase(
                output_dir=output_dir,
                real_data_root=(ROOT / args.real_data_root).resolve(),
                gate_model_path=str(Path(args.gate_model_path).resolve()),
                selected_method=selected_method,
                top_k=int(args.top_k),
                seed=int(args.seed),
                bootstrap_samples=int(args.bootstrap_samples),
                refine_epochs=int(args.refine_epochs),
            )


def run_cleanup_phase(*, output_dir: Path) -> None:
    before_sizes = measure_root_entries()
    reference_index = build_reference_index()

    keep_rows = build_keep_rows(before_sizes=before_sizes, reference_index=reference_index)
    delete_rows, pending_rows = build_delete_rows(before_sizes=before_sizes, reference_index=reference_index)

    write_manifest(
        path=output_dir / "cleanup_keep_manifest.md",
        title="Cleanup Keep Manifest",
        sections=[("Keep", keep_rows)],
    )
    write_manifest(
        path=output_dir / "cleanup_delete_manifest.md",
        title="Cleanup Delete Manifest",
        sections=[("Delete Candidates", delete_rows), ("Pending", pending_rows)],
    )

    executed_rows: list[ManifestRow] = []
    post_pending_rows = list(pending_rows)
    for row in delete_rows:
        try:
            safe_delete_path(row.path)
            executed_rows.append(
                ManifestRow(
                    path=row.path,
                    entry_type=row.entry_type,
                    current_size_bytes=row.current_size_bytes,
                    regenerable=row.regenerable,
                    referenced_by_kept_logic=row.referenced_by_kept_logic,
                    rationale=row.rationale,
                    risk=row.risk,
                    status="deleted",
                )
            )
        except Exception as exc:  # pragma: no cover - best-effort cleanup reporting
            post_pending_rows.append(
                ManifestRow(
                    path=row.path,
                    entry_type=row.entry_type,
                    current_size_bytes=row.current_size_bytes,
                    regenerable=row.regenerable,
                    referenced_by_kept_logic=row.referenced_by_kept_logic,
                    rationale=f"{row.rationale} Deletion remained blocked by workspace permissions: {type(exc).__name__}.",
                    risk="medium",
                    status="pending_permission_block",
                )
            )

    after_sizes = measure_root_entries()
    write_manifest(
        path=output_dir / "cleanup_delete_manifest.md",
        title="Cleanup Delete Manifest",
        sections=[("Deleted", executed_rows), ("Pending", post_pending_rows)],
    )
    write_disk_usage_csv(
        path=output_dir / "disk_usage_before_after.csv",
        before_sizes=before_sizes,
        after_sizes=after_sizes,
        tracked_paths=[row.path for row in keep_rows + executed_rows + pending_rows],
    )

    freed_bytes = sum(max(0, before_sizes.get(row.path, 0) - after_sizes.get(row.path, 0)) for row in executed_rows)
    notes = [
        "# Cleanup Notes",
        "",
        "## Audit First",
        "- Cleanup manifests were written before the deletion pass. Only the `Delete Candidates` section was executed.",
        "- `official_hvg_ak36oj8t` stayed in pending because directory listing is denied in the current workspace ACL state, so it could not be safely audited.",
        "- `.tmp_pip` and `.tmp_pip_py311` stayed in pending because nested pip temp directories are permission-blocked in the current workspace.",
        "",
        "## Why `.official_hvg_cache` Was Safe To Remove",
        "- The cache root is hard-coded in `src/hvg_research/official_baselines.py`.",
        "- Cache directories are keyed by raw array memory address plus shape, so they are runtime materializations rather than durable scientific assets.",
        "- The cache stores `counts.npy`, `counts.mtx`, batch sidecars, and worker outputs that can be regenerated by rerunning official baseline workers.",
        "",
        "## Keep Boundary",
        "- Preserved core code/data/docs, round2/phase2/phase3 decision artifacts, `artifacts_next_direction`, and the current anchor escape checkpoint in `artifacts_gate_learning_v_next9_multireal`.",
        "- Left small or ambiguous directories outside the priority delete list in place to avoid over-cleaning a non-git workspace.",
        "",
        "## Released Space",
        f"- Total space released in cleanup phase: {format_gib(freed_bytes)} ({freed_bytes} bytes).",
    ]
    (output_dir / "cleanup_notes.md").write_text("\n".join(notes) + "\n", encoding="utf-8")


def run_bfs_and_smoke_phase(
    *,
    output_dir: Path,
    real_data_root: Path,
    gate_model_path: str,
    top_k: int,
    seed: int,
    bootstrap_samples: int,
    refine_epochs: int,
) -> str:
    evidence = load_evidence_tables()
    write_idea_bfs(output_dir=output_dir, evidence=evidence)

    resources = load_dataset_resources(real_data_root=real_data_root)
    dataset_cache = pkg.DatasetCache(resources)

    analysis_rows = run_analysis_level_smoke(
        dataset_cache=dataset_cache,
        dataset_names=REPRESENTATIVE_SMOKE_DATASETS,
        candidate_methods=DEFAULT_BFS_SHORTLIST,
        gate_model_path=gate_model_path,
        top_k=top_k,
        seed=seed,
        refine_epochs=refine_epochs,
    )

    analysis_df = pd.DataFrame(analysis_rows).sort_values(["method", "dataset"]).reset_index(drop=True)
    analysis_summary = summarize_analysis_smoke(analysis_df)

    benchmark_candidates = analysis_summary[analysis_summary["analysis_pass"] == True]["method"].astype(str).tolist()  # noqa: E712
    raw_smoke_df = pd.DataFrame()
    smoke_summary = pd.DataFrame()
    selected_method = ""

    if benchmark_candidates:
        benchmark_methods = tuple([SAFE_ANCHOR_METHOD, *benchmark_candidates])
        raw_smoke_df = run_dataset_benchmark(
            dataset_cache=dataset_cache,
            dataset_names=REPRESENTATIVE_SMOKE_DATASETS,
            method_names=benchmark_methods,
            gate_model_path=gate_model_path,
            top_k=top_k,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            refine_epochs=refine_epochs,
        )
        biology_smoke_df = run_biology_proxy(
            dataset_cache=dataset_cache,
            dataset_names=REPRESENTATIVE_SMOKE_DATASETS,
            method_names=benchmark_methods,
            gate_model_path=gate_model_path,
            top_k=top_k,
            seed=seed,
            refine_epochs=refine_epochs,
        )
        raw_smoke_df = raw_smoke_df.merge(
            biology_smoke_df[["dataset", "method", "weighted_marker_recall_at_50"]],
            on=["dataset", "method"],
            how="left",
        )
        smoke_summary = summarize_benchmark_smoke(raw_smoke_df)
        passing = smoke_summary[smoke_summary["smoke_pass"] == True].copy()  # noqa: E712
        if not passing.empty:
            selected_method = str(
                passing.sort_values(
                    ["positive_headroom_mean_delta", "overall_mean_delta", "mean_runtime_ratio_vs_anchor"],
                    ascending=[False, False, True],
                ).iloc[0]["method"]
            )

    write_smoke_outputs(
        output_dir=output_dir,
        analysis_df=analysis_df,
        analysis_summary=analysis_summary,
        raw_smoke_df=raw_smoke_df,
        smoke_summary=smoke_summary,
        selected_method=selected_method,
    )
    return selected_method


def run_dfs_phase(
    *,
    output_dir: Path,
    real_data_root: Path,
    gate_model_path: str,
    selected_method: str,
    top_k: int,
    seed: int,
    bootstrap_samples: int,
    refine_epochs: int,
) -> None:
    resources = load_dataset_resources(real_data_root=real_data_root)
    dataset_cache = pkg.DatasetCache(resources)
    evidence = load_evidence_tables()
    all_datasets = tuple(evidence["manifest_df"]["dataset_name"].astype(str).tolist())

    if selected_method == "adaptive_core_consensus_hvg":
        ablation_method = "adaptive_core_consensus_no_agreement_hvg"
    elif selected_method == "adaptive_rank_aggregate_hvg":
        ablation_method = "adaptive_rank_aggregate_no_agreement_hvg"
    else:
        raise ValueError(f"Unsupported DFS mainline method: {selected_method}")

    method_names = (SAFE_ANCHOR_METHOD, selected_method, ablation_method)
    raw_df = run_dataset_benchmark(
        dataset_cache=dataset_cache,
        dataset_names=all_datasets,
        method_names=method_names,
        gate_model_path=gate_model_path,
        top_k=top_k,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
        refine_epochs=refine_epochs,
    )
    biology_df = run_biology_proxy(
        dataset_cache=dataset_cache,
        dataset_names=all_datasets,
        method_names=method_names,
        gate_model_path=gate_model_path,
        top_k=top_k,
        seed=seed,
        refine_epochs=refine_epochs,
    )
    raw_df = raw_df.merge(
        biology_df[["dataset", "method", "weighted_marker_recall_at_50"]],
        on=["dataset", "method"],
        how="left",
    )

    failure_df = evidence["failure_df"].set_index("dataset")
    headroom_df = evidence["headroom_df"].set_index("dataset")
    positive_headroom_datasets = sorted(headroom_df[headroom_df["headroom_vs_best_single"] > 0].index.tolist())
    atlas_datasets = sorted(failure_df[failure_df["regime"] == "atlas-like / large homogeneous panel"].index.tolist())

    selected_rows = build_delta_rows(
        raw_df=raw_df,
        method=selected_method,
        anchor_method=SAFE_ANCHOR_METHOD,
        failure_df=failure_df,
        headroom_df=headroom_df,
    )
    ablation_rows = build_delta_rows(
        raw_df=raw_df,
        method=ablation_method,
        anchor_method=SAFE_ANCHOR_METHOD,
        failure_df=failure_df,
        headroom_df=headroom_df,
    )

    mainline_dir = output_dir / MAINLINE_DIR_TEMPLATE.format(idea_name=selected_method)
    mainline_dir.mkdir(parents=True, exist_ok=True)
    selected_rows.to_csv(mainline_dir / "mainline_holdout_results.csv", index=False)

    summary_rows = [
        summarize_delta_table(
            delta_df=selected_rows,
            method=selected_method,
            positive_headroom_datasets=positive_headroom_datasets,
            atlas_datasets=atlas_datasets,
        ),
        summarize_delta_table(
            delta_df=ablation_rows,
            method=ablation_method,
            positive_headroom_datasets=positive_headroom_datasets,
            atlas_datasets=atlas_datasets,
        ),
    ]
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(mainline_dir / "mainline_holdout_summary.csv", index=False)

    regime_summary = (
        selected_rows.groupby("regime", as_index=False)
        .agg(
            dataset_count=("dataset", "count"),
            mean_delta_vs_anchor=("overall_delta_vs_anchor", "mean"),
            mean_cluster_delta=("cluster_silhouette_delta_vs_anchor", "mean"),
            mean_stability_delta=("stability_delta_vs_anchor", "mean"),
            mean_biology_delta=("biology_delta_vs_anchor", "mean"),
            mean_runtime_ratio=("runtime_ratio_vs_anchor", "mean"),
        )
        .sort_values("mean_delta_vs_anchor", ascending=False)
        .reset_index(drop=True)
    )
    regime_summary.to_csv(mainline_dir / "mainline_regime_summary.csv", index=False)

    ablation_summary = pd.DataFrame(
        [
            {
                "main_method": selected_method,
                "ablation_method": ablation_method,
                "mean_delta_vs_anchor_gap": float(
                    summary_df.loc[summary_df["method"] == selected_method, "overall_mean_delta"].iloc[0]
                    - summary_df.loc[summary_df["method"] == ablation_method, "overall_mean_delta"].iloc[0]
                ),
                "positive_headroom_gap": float(
                    summary_df.loc[summary_df["method"] == selected_method, "positive_headroom_mean_delta"].iloc[0]
                    - summary_df.loc[summary_df["method"] == ablation_method, "positive_headroom_mean_delta"].iloc[0]
                ),
                "mean_biology_gap": float(
                    summary_df.loc[summary_df["method"] == selected_method, "mean_biology_delta"].iloc[0]
                    - summary_df.loc[summary_df["method"] == ablation_method, "mean_biology_delta"].iloc[0]
                ),
                "mean_runtime_ratio_gap": float(
                    summary_df.loc[summary_df["method"] == selected_method, "mean_runtime_ratio_vs_anchor"].iloc[0]
                    - summary_df.loc[summary_df["method"] == ablation_method, "mean_runtime_ratio_vs_anchor"].iloc[0]
                ),
            }
        ]
    )
    ablation_summary.to_csv(mainline_dir / "mainline_ablation_summary.csv", index=False)

    selected_summary = summary_df[summary_df["method"] == selected_method].iloc[0]
    go = (
        float(selected_summary["overall_mean_delta"]) > 0.0
        and float(selected_summary["positive_headroom_mean_delta"]) > 0.15
        and float(selected_summary["atlas_like_mean_delta"]) >= -0.05
        and float(selected_summary["mean_biology_delta"]) >= -0.02
    )
    lines = [
        "# Mainline Go / No-Go",
        "",
        f"- Mainline method: `{selected_method}`.",
        f"- Single-pass overall mean delta vs `{SAFE_ANCHOR_METHOD}`: {float(selected_summary['overall_mean_delta']):.4f}.",
        f"- Positive-headroom subset mean delta: {float(selected_summary['positive_headroom_mean_delta']):.4f}.",
        f"- Atlas-like controls mean delta: {float(selected_summary['atlas_like_mean_delta']):.4f}.",
        f"- Mean biology proxy delta: {float(selected_summary['mean_biology_delta']):.4f}.",
        f"- Mean runtime ratio vs anchor: {float(selected_summary['mean_runtime_ratio_vs_anchor']):.4f}.",
        "",
        "## Decision",
        f"- {'GO' if go else 'NO-GO'} according to the requested DFS thresholds.",
        "",
        "## Minimal Ablation",
        f"- Compared against `{ablation_method}` only; no sweep and no second branch was run.",
        f"- Mainline minus ablation overall mean delta gap: {float(ablation_summary.iloc[0]['mean_delta_vs_anchor_gap']):.4f}.",
    ]
    (mainline_dir / "mainline_go_no_go.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def measure_root_entries() -> dict[Path, int]:
    entries = [path for path in ROOT.iterdir() if path.name != OUTPUT_DIR.name]
    return {path: measure_path_bytes(path) for path in entries}


def measure_path_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    if path.is_file():
        return int(path.stat().st_size)
    total = 0
    try:
        for child in path.rglob("*"):
            try:
                if child.is_file():
                    total += int(child.stat().st_size)
            except OSError:
                continue
    except OSError:
        return 0
    return total


def build_reference_index() -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for base in KEEP_REFERENCE_PATHS:
        paths = [base] if base.is_file() else list(base.rglob("*"))
        for path in paths:
            if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            for token in (
                ".official_hvg_cache",
                ".py311_vendor",
                ".tmp_pip",
                ".tmp_pip_py311",
                "tmp_real_input",
                "official_hvg_ak36oj8t",
                "artifacts_gate_learning",
                "artifacts_gate_learning_v_next9_multireal",
                "artifacts_selector_bank_benchmark",
                "artifacts_recomb_ismb_benchmark",
            ):
                if token in text:
                    index.setdefault(token, []).append(str(path.relative_to(ROOT)))
    return index


def build_keep_rows(*, before_sizes: dict[Path, int], reference_index: dict[str, list[str]]) -> list[ManifestRow]:
    keep_specs = [
        (ROOT / "src", "core", False, True, "Core research code and default keep asset."),
        (ROOT / "data", "core", False, False, "Primary data asset; explicitly protected."),
        (ROOT / "scripts", "core", False, True, "Runnable experiment and analysis entry points."),
        (ROOT / "README.md", "core", False, True, "Primary operator documentation."),
        (ROOT / "requirements.txt", "core", True, False, "Environment declaration."),
        (ROOT / "artifacts_topconf_selector_round2", "artifact", False, True, "Primary benchmark evidence reused for BFS."),
        (ROOT / "artifacts_codex_selector_mvp", "artifact", False, True, "Contains biology proxy and phase1 selector evidence."),
        (ROOT / "artifacts_codex_selector_phase2", "artifact", False, True, "Contains top-k/seed evidence and phase2 holdout analysis."),
        (ROOT / "artifacts_codex_selector_phase3_official_repro", "artifact", False, True, "Strict official reproduction and selector no-go proof."),
        (ROOT / "artifacts_next_direction", "artifact", False, True, "Anchor headroom analysis reused directly in BFS."),
        (ROOT / "artifacts_gate_learning_v_next9_multireal", "artifact", False, True, "Current anchor escape checkpoint path used by the existing benchmark harness."),
        (ROOT / "artifacts_recomb_ismb_benchmark", "artifact", False, True, "Default source dir for the retained round2 package builder."),
        (ROOT / ".py311_vendor", "core", False, True, "Vendored Triku dependency required by strict official baseline reproduction."),
        (ROOT / "artifacts", "smoke", True, True, "Small referenced smoke artifact retained because README still points to it."),
        (ROOT / "artifacts_adaptive_hybrid_litaware_eval", "artifact", True, False, "Small anchor-era evaluation artifact retained to avoid over-cleaning."),
        (ROOT / "artifacts_adaptive_stat_eval", "artifact", True, False, "Small anchor-era evaluation artifact retained to avoid over-cleaning."),
        (ROOT / "artifacts_adaptive_stat_eval_v2", "artifact", True, False, "Small anchor-era evaluation artifact retained to avoid over-cleaning."),
        (ROOT / "artifacts_adaptive_stat_eval_v3", "artifact", True, False, "Small anchor-era evaluation artifact retained to avoid over-cleaning."),
        (ROOT / "artifacts_real_csv", "artifact", True, False, "Data-adjacent conversion artifact retained because provenance is ambiguous in a non-git workspace."),
        (ROOT / "artifacts_real_h5ad", "artifact", True, False, "Data-adjacent conversion artifact retained because provenance is ambiguous in a non-git workspace."),
        (ROOT / "artifacts_real_mtx", "artifact", True, False, "Data-adjacent conversion artifact retained because provenance is ambiguous in a non-git workspace."),
        (ROOT / "artifacts_real_inputs_round1", "artifact", True, False, "Small prior input-round artifact retained because it is not a priority delete target."),
        (ROOT / "artifacts_real_inputs_round2", "artifact", True, False, "Small prior input-round artifact retained because it is not a priority delete target."),
    ]
    rows: list[ManifestRow] = []
    for path, entry_type, regenerable, referenced, rationale in keep_specs:
        token = path.name
        rows.append(
            ManifestRow(
                path=path,
                entry_type=entry_type,
                current_size_bytes=before_sizes.get(path, measure_path_bytes(path)),
                regenerable=regenerable,
                referenced_by_kept_logic=referenced or bool(reference_index.get(token)),
                rationale=rationale,
                risk="high" if path in MANDATORY_KEEP_PATHS else "medium",
                status="kept",
            )
        )
    return sorted(rows, key=lambda row: row.path.as_posix())


def build_delete_rows(*, before_sizes: dict[Path, int], reference_index: dict[str, list[str]]) -> tuple[list[ManifestRow], list[ManifestRow]]:
    delete_paths = [
        ROOT / ".official_hvg_cache",
        ROOT / ".tmp_pip",
        ROOT / ".tmp_pip_py311",
        ROOT / "tmp_real_input",
        ROOT / "src" / "hvg_research" / "__pycache__",
        ROOT / "scripts" / "__pycache__",
    ]
    delete_paths.extend(
        sorted(
            path
            for path in ROOT.iterdir()
            if path.name.startswith("artifacts_gate_learning") and path.name != "artifacts_gate_learning_v_next9_multireal"
        )
    )
    delete_paths.extend(sorted(path for path in ROOT.iterdir() if path.name.startswith("artifacts_selector_bank_benchmark")))

    delete_rows: list[ManifestRow] = []
    for path in delete_paths:
        token = path.name if path.exists() else path.as_posix()
        rationale = describe_delete_rationale(path)
        delete_rows.append(
            ManifestRow(
                path=path,
                entry_type=classify_delete_type(path),
                current_size_bytes=before_sizes.get(path, measure_path_bytes(path)),
                regenerable=True,
                referenced_by_kept_logic=bool(reference_index.get(token)),
                rationale=rationale,
                risk="low" if not bool(reference_index.get(token)) else "medium",
                status="planned_delete",
            )
        )

    pending_rows = [
        ManifestRow(
            path=ROOT / "official_hvg_ak36oj8t",
            entry_type="temp",
            current_size_bytes=before_sizes.get(ROOT / "official_hvg_ak36oj8t", 0),
            regenerable=True,
            referenced_by_kept_logic=bool(reference_index.get("official_hvg_ak36oj8t")),
            rationale="Access denied during listing, so the directory could not be safely audited before deletion.",
            risk="high",
            status="pending",
        )
    ]
    return delete_rows, pending_rows


def describe_delete_rationale(path: Path) -> str:
    name = path.name
    if name == ".official_hvg_cache":
        return "Runtime materialization cache for official baseline workers; keyed by memory-address-derived names and fully regenerable."
    if name in {".tmp_pip", ".tmp_pip_py311"}:
        return "Temporary pip staging directory with no retained scientific value."
    if name == "tmp_real_input":
        return "Legacy gate-learning temporary input directory only referenced by the retired gate experiment script."
    if name == "__pycache__":
        return "Python bytecode cache; always regenerable."
    if name.startswith("artifacts_selector_bank_benchmark"):
        return "Old selector-bank benchmark/smoke artifact from a retired line of work."
    if name.startswith("artifacts_gate_learning"):
        return "Old gate-learning artifact from a retired selector/routing line; not needed for the current anchor-core mainline."
    return "Regenerable cleanup target."


def classify_delete_type(path: Path) -> str:
    name = path.name
    if "cache" in name or name == "__pycache__":
        return "cache"
    if "tmp" in name:
        return "temp"
    if "smoke" in name:
        return "smoke"
    if "debug" in name:
        return "debug"
    return "obsolete"


def write_manifest(*, path: Path, title: str, sections: list[tuple[str, list[ManifestRow]]]) -> None:
    lines = [f"# {title}", ""]
    for section_name, rows in sections:
        lines.append(f"## {section_name}")
        lines.append("")
        lines.append("| path | type | current_size | regenerable | referenced | rationale | risk | status |")
        lines.append("| --- | --- | ---: | --- | --- | --- | --- | --- |")
        for row in rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row.path),
                        row.entry_type,
                        format_gib(row.current_size_bytes),
                        "yes" if row.regenerable else "no",
                        "yes" if row.referenced_by_kept_logic else "no",
                        row.rationale.replace("\n", " "),
                        row.risk,
                        row.status,
                    ]
                )
                + " |"
            )
        lines.append("")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_disk_usage_csv(*, path: Path, before_sizes: dict[Path, int], after_sizes: dict[Path, int], tracked_paths: list[Path]) -> None:
    unique_paths = sorted(set(tracked_paths), key=lambda item: str(item))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "before_bytes", "after_bytes", "freed_bytes"])
        for item in unique_paths:
            before = before_sizes.get(item, 0)
            after = after_sizes.get(item, 0)
            writer.writerow([str(item), before, after, max(0, before - after)])
        writer.writerow(["__repo_total__", sum(before_sizes.values()), sum(after_sizes.values()), max(0, sum(before_sizes.values()) - sum(after_sizes.values()))])


def safe_delete_path(path: Path) -> None:
    resolved = path.resolve(strict=False)
    root_resolved = ROOT.resolve(strict=True)
    if not str(resolved).startswith(str(root_resolved)):
        raise ValueError(f"Refusing to delete outside workspace: {path}")
    if not path.exists():
        return
    if path.is_file():
        os.chmod(path, stat.S_IWRITE)
        path.unlink()
        return

    def onerror(func, target, exc_info):  # pragma: no cover - Windows-only permission handler
        del exc_info
        os.chmod(target, stat.S_IWRITE)
        func(target)

    shutil.rmtree(path, onerror=onerror)


def load_evidence_tables() -> dict[str, pd.DataFrame]:
    headroom_df = pd.read_csv(ROOT / "artifacts_next_direction" / "anchor_headroom_tables.csv")
    headroom_df = headroom_df[headroom_df["row_type"] == "dataset"].copy().reset_index(drop=True)
    failure_df = pd.read_csv(ROOT / "artifacts_topconf_selector_round2" / "failure_taxonomy.csv").copy()
    benchmark_df = pd.read_csv(ROOT / "artifacts_topconf_selector_round2" / "benchmark_dataset_summary.csv").copy()
    biology_df = pd.read_csv(ROOT / "artifacts_codex_selector_mvp" / "biology_proxy_raw.csv").copy()
    topk_df = pd.read_csv(ROOT / "artifacts_codex_selector_phase2" / "_topk_seed_method_raw.csv").copy()
    manifest_df = pd.read_csv(ROOT / "artifacts_topconf_selector_round2" / "dataset_manifest.csv").copy()
    return {
        "headroom_df": headroom_df,
        "failure_df": failure_df,
        "benchmark_df": benchmark_df,
        "biology_df": biology_df,
        "topk_df": topk_df,
        "manifest_df": manifest_df,
    }


def write_idea_bfs(*, output_dir: Path, evidence: dict[str, pd.DataFrame]) -> None:
    headroom_df = evidence["headroom_df"]
    positive_df = headroom_df[headroom_df["headroom_vs_best_single"] > 0].copy()
    regime_summary = (
        headroom_df.groupby("regime")["headroom_vs_best_single"]
        .agg(["count", "mean", "median", "max"])
        .sort_values("mean", ascending=False)
        .reset_index()
    )
    winner_counts = headroom_df["best_single_expert"].value_counts()

    ideas = [
        build_idea_record(
            idea_name="adaptive_core_consensus_hvg",
            mechanism="Fixed single-scorer fusion over variance / mv_residual / fano / multinomial deviance with atlas guard and agreement bonus.",
            evidence_support="Positive headroom clusters in non-atlas regimes; winners are dispersed, but count-model methods and clustering/stability gains recur.",
            cross_discipline="meta-analysis + robust statistics + risk parity",
            why_better="Distills multi-expert evidence into one scorer instead of asking a selector to route among weak releases.",
            minimal_smoke="Analysis-level rank shift on six representative datasets, then top_k=200 / seed=7 small benchmark against adaptive_hybrid_hvg.",
            failure_mode="May still underfit triku/scanpy-specific signals and could become too conservative if atlas guard dominates.",
            implementation_cost="low",
            new_dependencies="no",
            expected_target_regime="batch-heavy + trajectory-like non-atlas datasets",
            smoke_cost="low",
            main_risk="Improvement may be too small once runtime and atlas control constraints are applied.",
            recommendation_rank=1,
        ),
        build_idea_record(
            idea_name="adaptive_rank_aggregate_hvg",
            mechanism="Trimmed-rank aggregation over the same anchor-core bank, with fixed agreement bonus and atlas/trajectory nudges.",
            evidence_support="Winner heterogeneity suggests score-scale mismatch; rank aggregation can fuse dispersed signals without routing.",
            cross_discipline="rank aggregation + robust voting",
            why_better="Avoids threshold/routing instability and keeps the upgrade inside one deterministic scorer.",
            minimal_smoke="Same six representative datasets; compare anchor overlap shift and then run the same fixed benchmark smoke.",
            failure_mode="Can wash out useful magnitude information and collapse toward the anchor if ranks stay too correlated.",
            implementation_cost="low",
            new_dependencies="no",
            expected_target_regime="mixed non-atlas regimes with method-order disagreement",
            smoke_cost="low",
            main_risk="Could perturb rankings without translating into clustering/stability wins.",
            recommendation_rank=2,
        ),
        build_idea_record(
            idea_name="adaptive_dispersion_guard_hvg",
            mechanism="Start from adaptive_stat and add a gene-level score-dispersion penalty to prefer cross-scorer agreement.",
            evidence_support="Positive-headroom gains look like stability/clustering rather than biology wins, so agreement regularization is plausible.",
            cross_discipline="robust statistics",
            why_better="Acts directly on anchor-core noise instead of re-opening selector calibration.",
            minimal_smoke="Inject a lightweight dispersion penalty and compare top-k drift plus stability delta on the six smoke datasets.",
            failure_mode="May simply suppress useful rare-program genes and hurt positive-headroom datasets.",
            implementation_cost="low",
            new_dependencies="no",
            expected_target_regime="atlas-risk-sensitive settings",
            smoke_cost="low",
            main_risk="Too conservative to recover non-atlas headroom.",
            recommendation_rank=3,
        ),
        build_idea_record(
            idea_name="adaptive_count_bridge_hvg",
            mechanism="Blend the classical anchor core with a single multinomial-deviance bridge term using continuous profile signals.",
            evidence_support="`multinomial_deviance_hvg` wins multiple held-out datasets and positive headroom is not explained by biology alone.",
            cross_discipline="generalized linear models",
            why_better="Cheaper and cleaner than bank expansion because it adds one count-model bridge inside the anchor.",
            minimal_smoke="Track overlap with multinomial deviance on positive-headroom datasets before small benchmark validation.",
            failure_mode="Could overfit count-model-friendly datasets and give back gains on atlas controls.",
            implementation_cost="low",
            new_dependencies="no",
            expected_target_regime="count-model-friendly + batch-heavy",
            smoke_cost="low",
            main_risk="May not generalize beyond the deviance-favored subsets.",
            recommendation_rank=4,
        ),
        build_idea_record(
            idea_name="adaptive_trajectory_rescue_hvg",
            mechanism="Trajectory-aware continuous boost toward fano-like structure when dropout and first-PC dominance are jointly high.",
            evidence_support="GBM_sd and immune-like positives sit in the high-dropout / trajectory-like regime where the anchor still has headroom.",
            cross_discipline="signal processing",
            why_better="Targets a concrete failure mode inside the anchor instead of wrapping another selector around it.",
            minimal_smoke="Check whether the trajectory subsets show larger controlled drift away from the anchor than atlas controls.",
            failure_mode="Could over-boost noisy dropout genes and reduce biology proxy quality.",
            implementation_cost="low",
            new_dependencies="no",
            expected_target_regime="high-dropout trajectory-like",
            smoke_cost="low",
            main_risk="Signal may be too narrow to help enough datasets.",
            recommendation_rank=5,
        ),
        build_idea_record(
            idea_name="adaptive_batch_invariance_hvg",
            mechanism="Use batch heterogeneity signals to upweight genes that are jointly strong under mv_residual and count-model scoring.",
            evidence_support="Kidney and several batch-heavy datasets still show anchor headroom while atlas-like controls do not.",
            cross_discipline="causal invariance",
            why_better="Moves the anchor toward batch-robust genes without selectorized bank logic.",
            minimal_smoke="Measure if batch-heavy datasets move more than atlas controls and whether runtime remains near-anchor.",
            failure_mode="Could reduce cluster separation when batch labels are weak or noisy.",
            implementation_cost="medium",
            new_dependencies="no",
            expected_target_regime="batch-heavy heterogeneous",
            smoke_cost="medium",
            main_risk="Profile signals may be too weak to drive useful changes.",
            recommendation_rank=6,
        ),
        build_idea_record(
            idea_name="adaptive_stability_shrinkage_hvg",
            mechanism="Cheap bootstrap-free self-consistency shrinkage that rewards genes stable across deterministic anchor sub-scorers.",
            evidence_support="Positive headroom is concentrated in stability/clustering metrics rather than biology proxy.",
            cross_discipline="shrinkage estimation",
            why_better="Encodes stability directly inside the scorer instead of sweeping thresholds around it.",
            minimal_smoke="Compare agreement-derived stability proxies and then benchmark only if the induced rank shift is regime-aware.",
            failure_mode="Might duplicate the anchor’s current conservatism and add little net value.",
            implementation_cost="medium",
            new_dependencies="no",
            expected_target_regime="mixed non-atlas datasets",
            smoke_cost="medium",
            main_risk="Could be too weak to create measurable benchmark delta.",
            recommendation_rank=7,
        ),
        build_idea_record(
            idea_name="adaptive_spectral_residual_hvg",
            mechanism="Augment the anchor with a graph-spectral rarity prior estimated from fast PCA neighborhoods.",
            evidence_support="Headroom sometimes appears as improved cluster separation without neighbor-preservation gains.",
            cross_discipline="graph spectral methods",
            why_better="A single scorer could inject structure-aware signal without reviving routing or bank cleanup.",
            minimal_smoke="Prototype only as analysis-level perturbation and stop before benchmark unless shifts are clearly promising.",
            failure_mode="Implementation cost is higher and the added structure prior could easily overshoot on atlas controls.",
            implementation_cost="high",
            new_dependencies="no",
            expected_target_regime="rare-program heterogeneous datasets",
            smoke_cost="high",
            main_risk="Too expensive for this restart round relative to evidence strength.",
            recommendation_rank=8,
        ),
    ]

    shortlist_df = pd.DataFrame(
        [
            {
                "idea_name": record["idea_name"],
                "mechanism": record["mechanism"],
                "expected_target_regime": record["expected_target_regime"],
                "evidence_support": record["evidence_support"],
                "smoke_cost": record["smoke_cost"],
                "main_risk": record["main_risk"],
                "recommendation_rank": record["recommendation_rank"],
            }
            for record in ideas[:2]
        ]
    )
    shortlist_df.to_csv(output_dir / "idea_shortlist.csv", index=False)

    lines = [
        "# Idea BFS",
        "",
        "## Evidence Snapshot",
        f"- Mean headroom vs best non-selector single expert across all held-out datasets: {headroom_df['headroom_vs_best_single'].mean():.4f}.",
        f"- Positive-headroom dataset count: {int((headroom_df['headroom_vs_best_single'] > 0).sum())} / {len(headroom_df)}.",
        f"- Atlas-like mean headroom: {float(headroom_df.loc[headroom_df['regime'] == 'atlas-like / large homogeneous panel', 'headroom_vs_best_single'].mean()):.4f}.",
        "- Positive-headroom improvements are concentrated in clustering/stability instead of biology proxy.",
        "",
        "## Regime Summary",
    ]
    for row in regime_summary.itertuples(index=False):
        lines.append(
            f"- {row.regime}: count={int(row.count)}, mean_headroom={float(row.mean):.4f}, median={float(row.median):.4f}, max={float(row.max):.4f}."
        )
    lines.extend(
        [
            "",
            "## Winner Spread",
        ]
    )
    for method_name, count in winner_counts.items():
        lines.append(f"- {method_name}: {int(count)} dataset wins.")
    lines.extend(
        [
            "",
            "## Candidate Ideas",
        ]
    )
    for record in ideas:
        lines.extend(
            [
                f"### {record['idea_name']}",
                f"- Mechanism hypothesis: {record['mechanism']}",
                f"- Supported by existing evidence: {record['evidence_support']}",
                f"- Cross-disciplinary inspiration: {record['cross_discipline']}",
                f"- Why more rational than selector / bank / threshold tuning: {record['why_better']}",
                f"- Minimum smoke: {record['minimal_smoke']}",
                f"- Most likely failure mode: {record['failure_mode']}",
                f"- Implementation cost: {record['implementation_cost']}",
                f"- New dependency needed: {record['new_dependencies']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Shortlist",
            f"- Rank 1: `{shortlist_df.iloc[0]['idea_name']}` because it keeps the upgrade inside one fixed scorer while directly bridging the recurring non-atlas count-model headroom.",
            f"- Rank 2: `{shortlist_df.iloc[1]['idea_name']}` because winner dispersion suggests a scale-robust aggregator is the cleanest second probe.",
        ]
    )
    (output_dir / "idea_bfs.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_idea_record(**kwargs) -> dict[str, object]:
    return kwargs


def load_dataset_resources(*, real_data_root: Path) -> pkg.DatasetResources:
    manifest_df = pd.read_csv(ROOT / "artifacts_topconf_selector_round2" / "dataset_manifest.csv")
    return pkg.load_dataset_resources(real_data_root=real_data_root, manifest_df=manifest_df)


def run_analysis_level_smoke(
    *,
    dataset_cache: pkg.DatasetCache,
    dataset_names: tuple[str, ...],
    candidate_methods: tuple[str, ...],
    gate_model_path: str,
    top_k: int,
    seed: int,
    refine_epochs: int,
) -> list[dict[str, object]]:
    registry = build_default_method_registry(
        top_k=top_k,
        refine_epochs=refine_epochs,
        random_state=seed,
        gate_model_path=gate_model_path,
    )
    rows: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        dataset = dataset_cache.get(dataset_name, seed)
        current_top_k = min(top_k, dataset.counts.shape[1])
        anchor_scores = np.asarray(registry[SAFE_ANCHOR_METHOD](dataset.counts, dataset.batches, current_top_k), dtype=np.float64)
        deviance_scores = np.asarray(registry["multinomial_deviance_hvg"](dataset.counts, dataset.batches, current_top_k), dtype=np.float64)
        anchor_topk = topk_indices(anchor_scores, current_top_k)
        deviance_topk = topk_indices(deviance_scores, current_top_k)
        anchor_deviance_overlap = jaccard(anchor_topk, deviance_topk)
        group_name = "positive_headroom" if dataset_name in REPRESENTATIVE_POSITIVE_DATASETS else "atlas_control"
        for method in candidate_methods:
            method_scores = np.asarray(registry[method](dataset.counts, dataset.batches, current_top_k), dtype=np.float64)
            method_topk = topk_indices(method_scores, current_top_k)
            topk_overlap_to_anchor = jaccard(method_topk, anchor_topk)
            rows.append(
                {
                    "stage": "analysis",
                    "dataset": dataset_name,
                    "group_name": group_name,
                    "method": method,
                    "rank_corr_to_anchor": spearman_correlation(anchor_scores, method_scores),
                    "topk_overlap_to_anchor": topk_overlap_to_anchor,
                    "topk_shift_vs_anchor": 1.0 - topk_overlap_to_anchor,
                    "topk_overlap_to_deviance": jaccard(method_topk, deviance_topk),
                    "delta_overlap_to_deviance_vs_anchor": jaccard(method_topk, deviance_topk) - anchor_deviance_overlap,
                    "score_dispersion_ratio_vs_anchor": safe_ratio(np.std(method_scores), np.std(anchor_scores)),
                }
            )
    return rows


def summarize_analysis_smoke(df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    if df.empty:
        return pd.DataFrame()
    for method, group in df.groupby("method", sort=False):
        positive_group = group[group["group_name"] == "positive_headroom"]
        control_group = group[group["group_name"] == "atlas_control"]
        positive_shift = float(positive_group["topk_shift_vs_anchor"].mean()) if not positive_group.empty else 0.0
        control_shift = float(control_group["topk_shift_vs_anchor"].mean()) if not control_group.empty else 0.0
        delta_to_deviance = float(positive_group["delta_overlap_to_deviance_vs_anchor"].mean()) if not positive_group.empty else 0.0
        analysis_pass = (positive_shift - control_shift) >= 0.02 or delta_to_deviance >= 0.01
        rows.append(
            {
                "method": method,
                "positive_shift_vs_anchor": positive_shift,
                "control_shift_vs_anchor": control_shift,
                "positive_minus_control_shift": positive_shift - control_shift,
                "positive_delta_overlap_to_deviance_vs_anchor": delta_to_deviance,
                "mean_rank_corr_to_anchor": float(group["rank_corr_to_anchor"].mean()),
                "analysis_pass": bool(analysis_pass),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["analysis_pass", "positive_minus_control_shift", "positive_delta_overlap_to_deviance_vs_anchor"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def run_dataset_benchmark(
    *,
    dataset_cache: pkg.DatasetCache,
    dataset_names: tuple[str, ...],
    method_names: tuple[str, ...],
    gate_model_path: str,
    top_k: int,
    seed: int,
    bootstrap_samples: int,
    refine_epochs: int,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        dataset = dataset_cache.get(dataset_name, seed)
        spec = dataset_cache.resources.spec_map[dataset_name]
        rows.extend(
            rr1.run_round1_dataset_benchmark(
                dataset=dataset,
                dataset_id=spec.dataset_id,
                spec=spec,
                method_names=method_names,
                gate_model_path=gate_model_path,
                refine_epochs=refine_epochs,
                top_k=top_k,
                seed=seed,
                bootstrap_samples=bootstrap_samples,
            )
        )
    raw_df = pd.DataFrame(rows)
    return rr1.add_run_level_scores(raw_df)


def run_biology_proxy(
    *,
    dataset_cache: pkg.DatasetCache,
    dataset_names: tuple[str, ...],
    method_names: tuple[str, ...],
    gate_model_path: str,
    top_k: int,
    seed: int,
    refine_epochs: int,
) -> pd.DataFrame:
    registry = build_default_method_registry(
        top_k=top_k,
        refine_epochs=refine_epochs,
        random_state=seed,
        gate_model_path=gate_model_path,
    )
    rows: list[dict[str, object]] = []
    for dataset_name in dataset_names:
        dataset = dataset_cache.get(dataset_name, seed)
        if dataset.labels is None:
            continue
        labels = np.asarray(dataset.labels, dtype=object)
        if np.unique(labels).size < 2:
            continue
        markers, class_weights = pkg.compute_one_vs_rest_markers(
            counts=dataset.counts,
            labels=labels,
            top_n=50,
        )
        current_top_k = min(top_k, dataset.counts.shape[1])
        for method in method_names:
            scores = np.asarray(registry[method](dataset.counts, dataset.batches, current_top_k), dtype=np.float64)
            selected = topk_indices(scores, current_top_k)
            _, weighted_marker, rare_marker = pkg.marker_recovery(
                selected=selected,
                marker_sets=markers,
                class_weights=class_weights,
            )
            rows.append(
                {
                    "dataset": dataset_name,
                    "method": method,
                    "weighted_marker_recall_at_50": float(weighted_marker),
                    "rare_marker_recall_at_50": float(rare_marker),
                }
            )
    return pd.DataFrame(rows)


def summarize_benchmark_smoke(raw_df: pd.DataFrame) -> pd.DataFrame:
    failure_df = pd.read_csv(ROOT / "artifacts_topconf_selector_round2" / "failure_taxonomy.csv").set_index("dataset")
    headroom_df = pd.read_csv(ROOT / "artifacts_next_direction" / "anchor_headroom_tables.csv")
    headroom_df = headroom_df[headroom_df["row_type"] == "dataset"].set_index("dataset")
    rows: list[dict[str, object]] = []
    for method in sorted(set(raw_df["method"].astype(str).tolist()) - {SAFE_ANCHOR_METHOD}):
        delta_df = build_delta_rows(
            raw_df=raw_df,
            method=method,
            anchor_method=SAFE_ANCHOR_METHOD,
            failure_df=failure_df,
            headroom_df=headroom_df,
        )
        rows.append(
            {
                **summarize_delta_table(
                    delta_df=delta_df,
                    method=method,
                    positive_headroom_datasets=list(REPRESENTATIVE_POSITIVE_DATASETS),
                    atlas_datasets=list(REPRESENTATIVE_CONTROL_DATASETS),
                ),
                "smoke_pass": bool(
                    float(delta_df[delta_df["dataset"].isin(REPRESENTATIVE_POSITIVE_DATASETS)]["overall_delta_vs_anchor"].mean()) > 0.0
                    and float(delta_df[delta_df["dataset"].isin(REPRESENTATIVE_CONTROL_DATASETS)]["overall_delta_vs_anchor"].mean()) >= -0.05
                    and float(delta_df["biology_delta_vs_anchor"].mean()) >= -0.02
                    and float(delta_df["runtime_ratio_vs_anchor"].mean()) <= 1.50
                    and (
                        float(delta_df["cluster_silhouette_delta_vs_anchor"].mean())
                        + float(delta_df["stability_delta_vs_anchor"].mean())
                    ) > 0.0
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["smoke_pass", "positive_headroom_mean_delta", "overall_mean_delta"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def build_delta_rows(
    *,
    raw_df: pd.DataFrame,
    method: str,
    anchor_method: str,
    failure_df: pd.DataFrame,
    headroom_df: pd.DataFrame,
) -> pd.DataFrame:
    method_df = raw_df[raw_df["method"] == method].copy().set_index("dataset")
    anchor_df = raw_df[raw_df["method"] == anchor_method].copy().set_index("dataset")
    join_cols = [
        "overall_score",
        "cluster_silhouette",
        "stability",
        "neighbor_preservation",
        "ari",
        "nmi",
        "label_silhouette",
        "runtime_sec",
        "weighted_marker_recall_at_50",
    ]
    anchor_join = anchor_df[join_cols].add_prefix("anchor_")
    merged = method_df.join(anchor_join, how="inner")
    merged = merged.join(failure_df[["regime"]], how="left")
    merged = merged.join(headroom_df[["headroom_vs_best_single"]], how="left")
    merged = merged.reset_index()
    merged["overall_delta_vs_anchor"] = merged["overall_score"] - merged["anchor_overall_score"]
    merged["cluster_silhouette_delta_vs_anchor"] = merged["cluster_silhouette"] - merged["anchor_cluster_silhouette"]
    merged["stability_delta_vs_anchor"] = merged["stability"] - merged["anchor_stability"]
    merged["neighbor_preservation_delta_vs_anchor"] = merged["neighbor_preservation"] - merged["anchor_neighbor_preservation"]
    merged["ari_delta_vs_anchor"] = merged.get("ari", np.nan) - merged.get("anchor_ari", np.nan)
    merged["nmi_delta_vs_anchor"] = merged.get("nmi", np.nan) - merged.get("anchor_nmi", np.nan)
    merged["label_silhouette_delta_vs_anchor"] = merged.get("label_silhouette", np.nan) - merged.get("anchor_label_silhouette", np.nan)
    merged["biology_delta_vs_anchor"] = merged["weighted_marker_recall_at_50"] - merged["anchor_weighted_marker_recall_at_50"]
    merged["runtime_ratio_vs_anchor"] = merged["runtime_sec"] / np.maximum(merged["anchor_runtime_sec"], 1e-8)
    return merged


def summarize_delta_table(
    *,
    delta_df: pd.DataFrame,
    method: str,
    positive_headroom_datasets: list[str],
    atlas_datasets: list[str],
) -> dict[str, object]:
    return {
        "method": method,
        "dataset_count": int(len(delta_df)),
        "overall_mean_delta": float(delta_df["overall_delta_vs_anchor"].mean()),
        "positive_headroom_mean_delta": float(delta_df[delta_df["dataset"].isin(positive_headroom_datasets)]["overall_delta_vs_anchor"].mean()),
        "atlas_like_mean_delta": float(delta_df[delta_df["dataset"].isin(atlas_datasets)]["overall_delta_vs_anchor"].mean()),
        "mean_cluster_delta": float(delta_df["cluster_silhouette_delta_vs_anchor"].mean()),
        "mean_stability_delta": float(delta_df["stability_delta_vs_anchor"].mean()),
        "mean_neighbor_delta": float(delta_df["neighbor_preservation_delta_vs_anchor"].mean()),
        "mean_biology_delta": float(delta_df["biology_delta_vs_anchor"].mean()),
        "mean_runtime_ratio_vs_anchor": float(delta_df["runtime_ratio_vs_anchor"].mean()),
    }


def write_smoke_outputs(
    *,
    output_dir: Path,
    analysis_df: pd.DataFrame,
    analysis_summary: pd.DataFrame,
    raw_smoke_df: pd.DataFrame,
    smoke_summary: pd.DataFrame,
    selected_method: str,
) -> None:
    result_frames = []
    if not analysis_df.empty:
        result_frames.append(analysis_df.assign(row_type="analysis_dataset"))
    if not analysis_summary.empty:
        result_frames.append(analysis_summary.assign(row_type="analysis_summary"))
    if not raw_smoke_df.empty:
        result_frames.append(raw_smoke_df.assign(row_type="benchmark_dataset"))
    if not smoke_summary.empty:
        result_frames.append(smoke_summary.assign(row_type="benchmark_summary"))
    result_df = pd.concat(result_frames, ignore_index=True, sort=False) if result_frames else pd.DataFrame()
    result_df.to_csv(output_dir / "smoke_results.csv", index=False)

    lines = [
        "# Smoke Screening",
        "",
        "## Analysis-Level Smoke",
    ]
    for row in analysis_summary.itertuples(index=False):
        lines.append(
            f"- `{row.method}`: positive_shift={float(row.positive_shift_vs_anchor):.4f}, control_shift={float(row.control_shift_vs_anchor):.4f}, "
            f"positive_minus_control={float(row.positive_minus_control_shift):.4f}, "
            f"delta_overlap_to_deviance={float(row.positive_delta_overlap_to_deviance_vs_anchor):.4f}, "
            f"pass={bool(row.analysis_pass)}."
        )
    lines.extend(["", "## Benchmark Smoke"])
    if smoke_summary.empty:
        lines.append("- No candidate cleared the analysis-level perturbation screen, so no benchmark smoke was run.")
    else:
        for row in smoke_summary.itertuples(index=False):
            lines.append(
                f"- `{row.method}`: overall_mean_delta={float(row.overall_mean_delta):.4f}, positive_mean_delta={float(row.positive_headroom_mean_delta):.4f}, "
                f"atlas_mean_delta={float(row.atlas_like_mean_delta):.4f}, biology_delta={float(row.mean_biology_delta):.4f}, "
                f"cluster_delta={float(row.mean_cluster_delta):.4f}, stability_delta={float(row.mean_stability_delta):.4f}, "
                f"runtime_ratio={float(row.mean_runtime_ratio_vs_anchor):.4f}, pass={bool(row.smoke_pass)}."
            )
    lines.extend(["", "## DFS Selection"])
    if selected_method:
        lines.append(f"- Selected for DFS: `{selected_method}`.")
    else:
        lines.append("- No candidate passed the requested smoke criteria; DFS was not unlocked.")
    (output_dir / "smoke_screening.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def topk_indices(scores: np.ndarray, top_k: int) -> np.ndarray:
    return np.argsort(np.asarray(scores, dtype=np.float64))[-int(top_k) :]


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    a_set = set(np.asarray(a).tolist())
    b_set = set(np.asarray(b).tolist())
    return float(len(a_set & b_set) / max(len(a_set | b_set), 1))


def spearman_correlation(a: np.ndarray, b: np.ndarray) -> float:
    a_rank = pd.Series(np.asarray(a, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)
    b_rank = pd.Series(np.asarray(b, dtype=np.float64)).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(a_rank) < 1e-8 or np.std(b_rank) < 1e-8:
        return 0.0
    return float(np.corrcoef(a_rank, b_rank)[0, 1])


def safe_ratio(numerator: float, denominator: float) -> float:
    if abs(float(denominator)) < 1e-8:
        return 0.0
    return float(numerator / denominator)


def format_gib(value: int) -> str:
    return f"{float(value) / (1024 ** 3):.3f} GiB"


if __name__ == "__main__":
    main()
