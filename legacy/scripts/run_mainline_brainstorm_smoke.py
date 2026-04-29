from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

try:
    import torch
except ImportError:  # pragma: no cover - smoke script can run without torch
    torch = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hvg_research import build_default_method_registry, discover_scrna_input_specs, evaluate_real_selection, load_scrna_dataset
from hvg_research.eval import timed_call


ANCHOR_METHOD = "adaptive_hybrid_hvg"
BASELINE_METHODS = (
    "variance",
    "mv_residual",
    "fano",
    "multinomial_deviance_hvg",
    ANCHOR_METHOD,
)
CANDIDATE_MAINLINES = (
    "adaptive_core_consensus_hvg",
    "adaptive_rank_aggregate_hvg",
    "adaptive_eb_shrinkage_hvg",
    "adaptive_risk_parity_hvg",
    "adaptive_spectral_locality_hvg",
    "adaptive_stability_jackknife_hvg",
    "adaptive_invariant_residual_hvg",
    "sigma_safe_core_v6_hvg",
)

IDEA_CARDS: dict[str, dict[str, str]] = {
    "adaptive_core_consensus_hvg": {
        "mainline": "固定 multi-signal consensus anchor",
        "why": "既有 benchmark 显示非 atlas 数据的正向 headroom 分散在 count residual、deviance、稳定性等信号上；固定融合可避免 selector 线已经失败的 dataset-level release 风险。",
        "risk": "可能把互补专家平均掉，在 official partial route 上表现成 mixed tradeoff。",
    },
    "adaptive_rank_aggregate_hvg": {
        "mainline": "rank-level robust aggregation",
        "why": "不同 scorer 的 score scale 不可比时，trimmed rank fusion 能保留跨方法一致靠前的基因，同时减少极端单信号主导。",
        "risk": "排名扰动可能很大，但未必转化为 ARI/NMI 或邻域保持收益。",
    },
    "adaptive_eb_shrinkage_hvg": {
        "mainline": "empirical-Bayes shrinkage anchor",
        "why": "HVG scorer 的高方差基因容易被小样本/稀疏噪声放大；EB shrinkage 可把不稳定高分拉回更稳健的跨数据集先验。",
        "risk": "收缩过强会退化成保守 anchor，缺少足够新信号。",
    },
    "adaptive_risk_parity_hvg": {
        "mainline": "risk-parity multi-objective scorer",
        "why": "当前失败模式是单一指标增益伴随其他 benchmark-facing 指标回吐；risk parity 的动机是让 variance/residual/deviance/stability 没有一个信号独占风险预算。",
        "risk": "平衡约束可能牺牲最强 regime-specific expert 的尖峰优势。",
    },
    "adaptive_spectral_locality_hvg": {
        "mainline": "graph/spectral locality scorer",
        "why": "triku-like 局部结构信号曾在部分真实数据上有优势；直接把局部邻域富集作为 gene-level 信号可测试是否存在非路由结构性 headroom。",
        "risk": "图构建成本较高，且 atlas/control 数据上可能引入 batch 或局部密度伪信号。",
    },
    "adaptive_stability_jackknife_hvg": {
        "mainline": "selection-stability-first scorer",
        "why": "如果目标是 benchmark-safe anchor，而不是单数据集尖峰，jackknife 稳定基因应更可迁移，并可解释为降低 bootstrap 方差。",
        "risk": "稳定性本身不等于生物分辨率，可能偏向 housekeeping 或高表达基因。",
    },
    "adaptive_invariant_residual_hvg": {
        "mainline": "batch/pseudo-environment invariant residual scorer",
        "why": "多 donor / batch-heavy 数据要求保留跨环境一致的 cell-state signal，抑制只在单一 batch 中突出的 shortcut。",
        "risk": "环境定义弱或标签缺失时，invariance penalty 可能误伤真实状态特异基因。",
    },
    "sigma_safe_core_v6_hvg": {
        "mainline": "safe sigma core with explicit fallback",
        "why": "sigma v5 已证明 wrapper-compatible 但太接近 baseline；v6 代表更激进但带安全 fallback 的 sigma 分支，可作为 conservative-control 主线的边界测试。",
        "risk": "安全门过强会继续近似 anchor，安全门过弱则复现 v 系列 mixed/no-go。",
    },
}


@dataclass(frozen=True)
class SmokeDatasetPlan:
    dataset_name: str
    max_cells: int
    max_genes: int
    rationale: str


SMOKE_DATASETS = (
    SmokeDatasetPlan("paul15", 900, 1800, "small curated trajectory-like hematopoiesis panel"),
    SmokeDatasetPlan("GBM_sd", 900, 2200, "compact labeled tumor-region dataset with prior non-atlas headroom"),
    SmokeDatasetPlan("cellxgene_immune_five_donors", 1000, 2200, "multi-donor immune dataset for batch-aware feasibility"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Brainstorm benchmark-aligned HVG mainlines and run lightweight smoke tests.")
    parser.add_argument("--real-data-root", type=str, default=str(ROOT / "data" / "real_inputs"))
    parser.add_argument("--output-dir", type=str, default=str(ROOT / "artifacts_mainline_brainstorm_smoke"))
    parser.add_argument("--datasets", type=str, default=",".join(plan.dataset_name for plan in SMOKE_DATASETS))
    parser.add_argument("--methods", type=str, default=",".join(CANDIDATE_MAINLINES))
    parser.add_argument("--top-k", type=int, default=200)
    parser.add_argument("--bootstrap-samples", type=int, default=1)
    parser.add_argument("--seed", type=int, default=11)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_datasets = tuple(part.strip() for part in args.datasets.split(",") if part.strip())
    requested_candidates = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    all_methods = tuple(dict.fromkeys((*BASELINE_METHODS, *requested_candidates)))
    plan_by_name = {plan.dataset_name: plan for plan in SMOKE_DATASETS}

    write_compute_context(output_dir / "compute_context.json")
    write_idea_cards(output_dir / "brainstormed_mainlines.md", requested_candidates)

    specs_by_name = {spec.dataset_name: spec for spec in discover_scrna_input_specs(Path(args.real_data_root))}
    missing = sorted(set(requested_datasets) - set(specs_by_name))
    if missing:
        raise FileNotFoundError(f"Missing dataset specs: {missing}")

    registry = build_default_method_registry(gate_model_path=None, refine_epochs=1, random_state=args.seed, top_k=args.top_k)
    missing_methods = [method for method in all_methods if method not in registry]
    if missing_methods:
        raise KeyError(f"Missing methods in registry: {missing_methods}")

    rows: list[dict[str, object]] = []
    for dataset_name in requested_datasets:
        plan = plan_by_name.get(dataset_name)
        if plan is None:
            raise KeyError(f"Dataset is not in SMOKE_DATASETS plan table: {dataset_name}")
        spec = specs_by_name[dataset_name]
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
        current_top_k = min(args.top_k, dataset.counts.shape[1])
        anchor_scores = None
        anchor_top = None
        for method_name in all_methods:
            method_fn = registry[method_name]
            status = "success"
            error = ""
            metrics: dict[str, float] = {}
            runtime_sec = float("nan")
            finite_fraction = float("nan")
            score_std = float("nan")
            overlap_to_anchor = float("nan")
            rank_corr_to_anchor = float("nan")
            try:
                scores, runtime_sec = timed_call(method_fn, dataset.counts, dataset.batches, current_top_k)
                scores = np.asarray(scores, dtype=np.float64)
                finite_fraction = float(np.isfinite(scores).mean())
                if scores.shape[0] != dataset.counts.shape[1]:
                    raise ValueError(f"score length {scores.shape[0]} != gene count {dataset.counts.shape[1]}")
                if not np.isfinite(scores).all():
                    raise ValueError("scores contain non-finite values")
                score_std = float(np.std(scores))
                selected = np.argsort(scores)[-current_top_k:]
                selected_set = set(selected.tolist())
                if method_name == ANCHOR_METHOD:
                    anchor_scores = scores.copy()
                    anchor_top = selected_set
                if anchor_scores is not None and anchor_top is not None:
                    overlap_to_anchor = len(selected_set & anchor_top) / max(len(selected_set | anchor_top), 1)
                    rank_corr_to_anchor = safe_rank_corr(scores, anchor_scores)
                metrics = evaluate_real_selection(
                    dataset.counts,
                    selected,
                    scorer_fn=lambda subset_counts, subset_batches, fn=method_fn, top_k=current_top_k: fn(
                        subset_counts, subset_batches, top_k
                    ),
                    labels=dataset.labels,
                    batches=dataset.batches,
                    top_k=current_top_k,
                    random_state=args.seed,
                    n_bootstrap=args.bootstrap_samples,
                )
            except Exception as exc:  # noqa: BLE001 - smoke table should preserve failures
                status = "error"
                error = repr(exc)

            rows.append(
                {
                    "dataset": dataset_name,
                    "dataset_rationale": plan.rationale,
                    "cells": int(dataset.counts.shape[0]),
                    "genes": int(dataset.counts.shape[1]),
                    "method": method_name,
                    "is_candidate": method_name in requested_candidates,
                    "status": status,
                    "error": error,
                    "runtime_sec": runtime_sec,
                    "finite_fraction": finite_fraction,
                    "score_std": score_std,
                    "topk_overlap_to_anchor": overlap_to_anchor,
                    "rank_corr_to_anchor": rank_corr_to_anchor,
                    **metrics,
                }
            )

    result_df = pd.DataFrame(rows)
    result_df.to_csv(output_dir / "smoke_results.csv", index=False)
    summary_df = summarize_smoke(result_df=result_df, requested_candidates=requested_candidates)
    summary_df.to_csv(output_dir / "candidate_smoke_summary.csv", index=False)
    write_smoke_report(output_dir / "smoke_report.md", result_df=result_df, summary_df=summary_df)
    print(f"Wrote mainline brainstorm smoke artifacts to {output_dir}")


def write_compute_context(path: Path) -> None:
    context: dict[str, object] = {"python": sys.version, "cuda_available": False, "gpu_count": 0, "device": "cpu"}
    if torch is not None:
        cuda_available = bool(torch.cuda.is_available())
        context.update(
            {
                "torch_version": torch.__version__,
                "cuda_available": cuda_available,
                "gpu_count": int(torch.cuda.device_count()) if cuda_available else 0,
                "device": "cuda" if cuda_available else "cpu",
            }
        )
        if cuda_available:
            context["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    path.write_text(json.dumps(context, indent=2, ensure_ascii=False), encoding="utf-8")


def write_idea_cards(path: Path, candidates: tuple[str, ...]) -> None:
    lines = [
        "# Benchmark-Aligned Mainline Brainstorm",
        "",
        "## Selection Principle",
        "- Keep the official benchmark route as the forcing function.",
        "- Do not reopen selector/routing; each idea must be a single scorer that emits one gene-level rank.",
        "- Smoke feasibility means: scorer runs on real benchmark inputs, returns finite non-flat scores, and produces measurable top-k movement from the current anchor.",
        "",
        "## Candidate Mainlines",
    ]
    for method in candidates:
        card = IDEA_CARDS.get(method, {})
        lines.extend(
            [
                f"### `{method}`",
                f"- Mainline: {card.get('mainline', 'candidate scorer')}",
                f"- Why useful: {card.get('why', 'Benchmark-compatible scorer worth smoke-testing.')}",
                f"- Main risk: {card.get('risk', 'Unknown until smoke-tested.')}",
                "",
            ]
        )
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def safe_rank_corr(scores: np.ndarray, anchor_scores: np.ndarray) -> float:
    scores_rank = pd.Series(scores).rank(method="average").to_numpy(dtype=np.float64)
    anchor_rank = pd.Series(anchor_scores).rank(method="average").to_numpy(dtype=np.float64)
    if np.std(scores_rank) == 0 or np.std(anchor_rank) == 0:
        return float("nan")
    return float(np.corrcoef(scores_rank, anchor_rank)[0, 1])


def summarize_smoke(*, result_df: pd.DataFrame, requested_candidates: tuple[str, ...]) -> pd.DataFrame:
    candidate_df = result_df[result_df["method"].isin(requested_candidates)].copy()
    rows = []
    for method_name, method_df in candidate_df.groupby("method", sort=False):
        success_df = method_df[method_df["status"] == "success"].copy()
        pass_count = int(
            (
                (success_df["finite_fraction"] >= 1.0)
                & (success_df["score_std"] > 1e-8)
                & (success_df["topk_overlap_to_anchor"].fillna(1.0) < 0.98)
            ).sum()
        )
        rows.append(
            {
                "method": method_name,
                "mainline": IDEA_CARDS.get(method_name, {}).get("mainline", "candidate scorer"),
                "success_count": int((method_df["status"] == "success").sum()),
                "dataset_count": int(len(method_df)),
                "feasibility_pass_count": pass_count,
                "mean_runtime_sec": float(success_df["runtime_sec"].mean()) if len(success_df) else float("nan"),
                "mean_topk_overlap_to_anchor": float(success_df["topk_overlap_to_anchor"].mean()) if len(success_df) else float("nan"),
                "mean_rank_corr_to_anchor": float(success_df["rank_corr_to_anchor"].mean()) if len(success_df) else float("nan"),
                "mean_ari": float(success_df["ari"].mean()) if "ari" in success_df and len(success_df) else float("nan"),
                "mean_nmi": float(success_df["nmi"].mean()) if "nmi" in success_df and len(success_df) else float("nan"),
                "feasibility_verdict": "pass" if pass_count >= 2 else "weak" if pass_count == 1 else "fail",
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["feasibility_verdict", "feasibility_pass_count", "mean_topk_overlap_to_anchor"],
        ascending=[True, False, True],
    )


def write_smoke_report(path: Path, *, result_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    lines = [
        "# Mainline Brainstorm Smoke Report",
        "",
        "## Smoke Definition",
        "- `success_count`: method completed on each small real benchmark dataset.",
        "- `feasibility_pass_count`: finite non-flat scores plus top-k overlap to `adaptive_hybrid_hvg` below 0.98.",
        "- This smoke proves implementation feasibility and non-trivial ranking movement; it does not claim benchmark superiority.",
        "",
        "## Candidate Summary",
    ]
    for row in summary_df.to_dict(orient="records"):
        lines.append(
            "- `{method}`: verdict={verdict}, success={success}/{total}, feasible_datasets={passes}, "
            "mean_overlap_to_anchor={overlap:.3f}, mean_runtime_sec={runtime:.2f}.".format(
                method=row["method"],
                verdict=row["feasibility_verdict"],
                success=int(row["success_count"]),
                total=int(row["dataset_count"]),
                passes=int(row["feasibility_pass_count"]),
                overlap=float(row["mean_topk_overlap_to_anchor"]),
                runtime=float(row["mean_runtime_sec"]),
            )
        )
    lines.extend(["", "## Dataset-Level Results"])
    candidate_rows = result_df[result_df["is_candidate"] == True].copy()  # noqa: E712
    for row in candidate_rows.to_dict(orient="records"):
        if row["status"] == "success":
            lines.append(
                "- `{method}` on `{dataset}`: overlap={overlap:.3f}, rank_corr={corr:.3f}, "
                "ARI={ari:.3f}, NMI={nmi:.3f}, runtime_sec={runtime:.2f}.".format(
                    method=row["method"],
                    dataset=row["dataset"],
                    overlap=float(row["topk_overlap_to_anchor"]),
                    corr=float(row["rank_corr_to_anchor"]),
                    ari=float(row.get("ari", float("nan"))),
                    nmi=float(row.get("nmi", float("nan"))),
                    runtime=float(row["runtime_sec"]),
                )
            )
        else:
            lines.append(f"- `{row['method']}` on `{row['dataset']}`: error={row['error']}.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
