from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - torch is optional for import but expected in this repo
    torch = None

from .sample_level_testing import evaluate_sample_level_methods


@dataclass(frozen=True)
class SimulationRuntimeInfo:
    device: str
    cuda_available: bool
    cuda_count: int
    cuda_name: str
    torch_version: str


@dataclass(frozen=True)
class HierarchicalSimulationScenario:
    scenario_id: str
    n_case: int
    n_control: int
    n_motifs: int
    n_signal_motifs: int
    patches_per_sample: int
    spots_per_patch: int
    sample_random_effect_sd: float
    patch_random_effect_sd: float
    effect_size: float
    baseline_prevalence_low: float
    baseline_prevalence_high: float

    @property
    def n_samples(self) -> int:
        return int(self.n_case + self.n_control)

    @property
    def spots_per_sample(self) -> int:
        return int(self.patches_per_sample * self.spots_per_patch)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SimulatedHierarchicalReplicate:
    scenario: HierarchicalSimulationScenario
    replicate_id: int
    random_state: int
    motif_ids: np.ndarray
    sample_ids: np.ndarray
    condition_labels: np.ndarray
    is_signal: np.ndarray
    effect_direction: np.ndarray
    true_logit_effect: np.ndarray
    baseline_prevalence: np.ndarray
    sample_totals: np.ndarray
    sample_positive_counts: np.ndarray
    mean_probability_control: np.ndarray
    mean_probability_case: np.ndarray


def get_simulation_runtime_info() -> SimulationRuntimeInfo:
    if torch is None:
        return SimulationRuntimeInfo(
            device="cpu",
            cuda_available=False,
            cuda_count=0,
            cuda_name="cpu",
            torch_version="unavailable",
        )
    cuda_available = bool(torch.cuda.is_available())
    cuda_count = int(torch.cuda.device_count()) if cuda_available else 0
    cuda_name = str(torch.cuda.get_device_name(0)) if cuda_available else "cpu"
    return SimulationRuntimeInfo(
        device="cuda" if cuda_available else "cpu",
        cuda_available=cuda_available,
        cuda_count=cuda_count,
        cuda_name=cuda_name,
        torch_version=str(torch.__version__),
    )


def simulate_hierarchical_motif_replicate(
    *,
    scenario: HierarchicalSimulationScenario,
    replicate_id: int,
    random_state: int,
    runtime_info: SimulationRuntimeInfo | None = None,
) -> SimulatedHierarchicalReplicate:
    runtime = runtime_info or get_simulation_runtime_info()
    rng = np.random.default_rng(random_state)

    motif_ids = np.asarray([f"motif_{index:03d}" for index in range(scenario.n_motifs)], dtype=object)
    sample_ids = np.asarray([f"sample_{index:03d}" for index in range(scenario.n_samples)], dtype=object)
    condition_labels = np.asarray(
        ["control"] * int(scenario.n_control) + ["case"] * int(scenario.n_case),
        dtype=object,
    )
    condition_case = (condition_labels == "case").astype(np.float32, copy=False)

    baseline_prevalence = rng.uniform(
        low=float(scenario.baseline_prevalence_low),
        high=float(scenario.baseline_prevalence_high),
        size=int(scenario.n_motifs),
    ).astype(np.float32, copy=False)
    baseline_logits = _logit(np.clip(baseline_prevalence, 1.0e-4, 1.0 - 1.0e-4))

    is_signal = np.zeros(int(scenario.n_motifs), dtype=bool)
    effect_direction = np.zeros(int(scenario.n_motifs), dtype=np.int8)
    true_logit_effect = np.zeros(int(scenario.n_motifs), dtype=np.float32)
    n_signal = int(min(max(scenario.n_signal_motifs, 0), scenario.n_motifs))
    if n_signal > 0 and float(scenario.effect_size) != 0.0:
        signal_indices = np.sort(rng.choice(int(scenario.n_motifs), size=n_signal, replace=False))
        directions = np.ones(n_signal, dtype=np.int8)
        directions[n_signal // 2 :] = -1
        rng.shuffle(directions)
        is_signal[signal_indices] = True
        effect_direction[signal_indices] = directions
        true_logit_effect[signal_indices] = directions.astype(np.float32, copy=False) * float(scenario.effect_size)

    if runtime.cuda_available:
        try:
            sample_positive_counts, mean_probability_control, mean_probability_case = _simulate_counts_torch(
                baseline_logits=baseline_logits,
                true_logit_effect=true_logit_effect,
                condition_case=condition_case,
                patches_per_sample=int(scenario.patches_per_sample),
                spots_per_patch=int(scenario.spots_per_patch),
                sample_random_effect_sd=float(scenario.sample_random_effect_sd),
                patch_random_effect_sd=float(scenario.patch_random_effect_sd),
                random_state=int(random_state),
                runtime_info=runtime,
            )
        except Exception:
            sample_positive_counts, mean_probability_control, mean_probability_case = _simulate_counts_numpy(
                baseline_logits=baseline_logits,
                true_logit_effect=true_logit_effect,
                condition_case=condition_case,
                patches_per_sample=int(scenario.patches_per_sample),
                spots_per_patch=int(scenario.spots_per_patch),
                sample_random_effect_sd=float(scenario.sample_random_effect_sd),
                patch_random_effect_sd=float(scenario.patch_random_effect_sd),
                random_state=int(random_state),
            )
    else:
        sample_positive_counts, mean_probability_control, mean_probability_case = _simulate_counts_numpy(
            baseline_logits=baseline_logits,
            true_logit_effect=true_logit_effect,
            condition_case=condition_case,
            patches_per_sample=int(scenario.patches_per_sample),
            spots_per_patch=int(scenario.spots_per_patch),
            sample_random_effect_sd=float(scenario.sample_random_effect_sd),
            patch_random_effect_sd=float(scenario.patch_random_effect_sd),
            random_state=int(random_state),
        )

    sample_totals = np.full(
        int(scenario.n_samples),
        int(scenario.spots_per_sample),
        dtype=np.int64,
    )
    return SimulatedHierarchicalReplicate(
        scenario=scenario,
        replicate_id=int(replicate_id),
        random_state=int(random_state),
        motif_ids=motif_ids,
        sample_ids=sample_ids,
        condition_labels=condition_labels,
        is_signal=is_signal,
        effect_direction=effect_direction,
        true_logit_effect=true_logit_effect.astype(np.float32, copy=False),
        baseline_prevalence=baseline_prevalence.astype(np.float32, copy=False),
        sample_totals=sample_totals,
        sample_positive_counts=sample_positive_counts.astype(np.int64, copy=False),
        mean_probability_control=mean_probability_control.astype(np.float32, copy=False),
        mean_probability_case=mean_probability_case.astype(np.float32, copy=False),
    )


def evaluate_simulated_replicate(
    *,
    replicate: SimulatedHierarchicalReplicate,
    fdr_alpha: float,
    sample_permutation_max_permutations: int,
) -> pd.DataFrame:
    control_mask = replicate.condition_labels == "control"
    case_mask = replicate.condition_labels == "case"
    sample_totals = replicate.sample_totals.astype(np.float64, copy=False)
    sample_fractions = np.divide(
        replicate.sample_positive_counts.astype(np.float64, copy=False),
        np.maximum(sample_totals[:, None], 1.0),
        out=np.zeros_like(replicate.sample_positive_counts, dtype=np.float64),
        where=sample_totals[:, None] > 0,
    )

    total_control_spots = int(np.sum(replicate.sample_totals[control_mask]))
    total_case_spots = int(np.sum(replicate.sample_totals[case_mask]))
    motif_metadata_rows: list[dict[str, object]] = []
    for motif_index, motif_id in enumerate(replicate.motif_ids.tolist()):
        control_counts = replicate.sample_positive_counts[control_mask, motif_index].astype(np.int64, copy=False)
        case_counts = replicate.sample_positive_counts[case_mask, motif_index].astype(np.int64, copy=False)
        control_fractions = sample_fractions[control_mask, motif_index]
        case_fractions = sample_fractions[case_mask, motif_index]
        motif_metadata_rows.append(
            {
                "scenario_id": replicate.scenario.scenario_id,
                "replicate_id": int(replicate.replicate_id),
                "motif_id": str(motif_id),
                "is_signal": bool(replicate.is_signal[motif_index]),
                "effect_direction": int(replicate.effect_direction[motif_index]),
                "true_logit_effect": float(replicate.true_logit_effect[motif_index]),
                "baseline_prevalence": float(replicate.baseline_prevalence[motif_index]),
                "mean_probability_control": float(replicate.mean_probability_control[motif_index]),
                "mean_probability_case": float(replicate.mean_probability_case[motif_index]),
                "mean_fraction_control": float(control_fractions.mean()) if control_fractions.size else np.nan,
                "mean_fraction_case": float(case_fractions.mean()) if case_fractions.size else np.nan,
                "delta_fraction": float(case_fractions.mean() - control_fractions.mean()) if case_fractions.size and control_fractions.size else np.nan,
                "n_control_samples": int(control_fractions.size),
                "n_case_samples": int(case_fractions.size),
                "control_positive_spots": int(control_counts.sum()),
                "case_positive_spots": int(case_counts.sum()),
                "total_control_spots": int(total_control_spots),
                "total_case_spots": int(total_case_spots),
            }
        )
    motif_metadata = pd.DataFrame(motif_metadata_rows)
    if motif_metadata.empty:
        return motif_metadata

    method_results = evaluate_sample_level_methods(
        motif_ids=replicate.motif_ids,
        sample_positive_counts=replicate.sample_positive_counts,
        sample_totals=replicate.sample_totals,
        labels=replicate.condition_labels,
        condition_a="control",
        condition_b="case",
        fdr_alpha=float(fdr_alpha),
        sample_permutation_max_permutations=int(sample_permutation_max_permutations),
        random_state=int(replicate.random_state),
        include_midp=True,
    )
    if method_results.empty:
        return method_results
    return motif_metadata.merge(method_results, on="motif_id", how="inner")


def build_method_long_results(motif_results: pd.DataFrame) -> pd.DataFrame:
    if motif_results.empty:
        return pd.DataFrame()
    if "method" in motif_results.columns:
        return motif_results.copy()
    scenario_cols = [
        "scenario_id",
        "replicate_id",
        "motif_id",
        "is_signal",
        "effect_direction",
        "true_logit_effect",
        "baseline_prevalence",
        "mean_probability_control",
        "mean_probability_case",
        "mean_fraction_control",
        "mean_fraction_case",
        "delta_fraction",
    ]
    records: list[pd.DataFrame] = []
    method_specs = (
        ("naive_fisher", "naive_fisher_pvalue", "naive_fisher_qvalue", "naive_fisher_discovery", "naive_fisher_raw_call"),
        (
            "sample_permutation",
            "sample_permutation_pvalue",
            "sample_permutation_qvalue",
            "sample_permutation_discovery",
            "sample_permutation_raw_call",
        ),
    )
    for method_name, pvalue_col, qvalue_col, discovery_col, raw_call_col in method_specs:
        subset = motif_results.loc[:, scenario_cols + [pvalue_col, qvalue_col, discovery_col, raw_call_col]].copy()
        subset = subset.rename(
            columns={
                pvalue_col: "pvalue",
                qvalue_col: "qvalue",
                discovery_col: "discovery",
                raw_call_col: "raw_call",
            }
        )
        subset["method"] = method_name
        records.append(subset)
    return pd.concat(records, ignore_index=True)


def summarize_simulation_metrics(method_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if method_results.empty:
        empty = pd.DataFrame()
        return empty, empty

    replicate_rows: list[dict[str, object]] = []
    group_cols = ["scenario_id", "replicate_id", "method"]
    for keys, frame in method_results.groupby(group_cols, observed=False):
        discoveries = int(np.sum(frame["discovery"].to_numpy(dtype=bool, copy=False)))
        raw_calls = int(np.sum(frame["raw_call"].to_numpy(dtype=bool, copy=False)))
        true_signals = int(np.sum(frame["is_signal"].to_numpy(dtype=bool, copy=False)))
        null_motifs = int(frame.shape[0] - true_signals)
        true_positives = int(np.sum(frame["discovery"].to_numpy(dtype=bool, copy=False) & frame["is_signal"].to_numpy(dtype=bool, copy=False)))
        false_positives = int(discoveries - true_positives)
        raw_true_positives = int(np.sum(frame["raw_call"].to_numpy(dtype=bool, copy=False) & frame["is_signal"].to_numpy(dtype=bool, copy=False)))
        raw_false_positives = int(raw_calls - raw_true_positives)
        signal_mask = frame["is_signal"].to_numpy(dtype=bool, copy=False)
        null_mask = ~signal_mask
        replicate_rows.append(
            {
                "scenario_id": str(keys[0]),
                "replicate_id": int(keys[1]),
                "method": str(keys[2]),
                "discoveries": discoveries,
                "true_positives": true_positives,
                "false_positives": false_positives,
                "empirical_fdr": float(false_positives / max(discoveries, 1)),
                "empirical_power": float(true_positives / max(true_signals, 1)) if true_signals > 0 else np.nan,
                "null_rejection_rate": float(false_positives / max(null_motifs, 1)) if null_motifs > 0 else np.nan,
                "raw_calls": raw_calls,
                "raw_true_positives": raw_true_positives,
                "raw_false_positives": raw_false_positives,
                "raw_empirical_fdr": float(raw_false_positives / max(raw_calls, 1)),
                "raw_empirical_power": float(raw_true_positives / max(true_signals, 1)) if true_signals > 0 else np.nan,
                "true_signals": true_signals,
                "null_motifs": null_motifs,
                "signal_mean_pvalue": float(frame.loc[signal_mask, "pvalue"].mean()) if signal_mask.any() else np.nan,
                "null_mean_pvalue": float(frame.loc[null_mask, "pvalue"].mean()) if null_mask.any() else np.nan,
            }
        )
    replicate_summary = pd.DataFrame(replicate_rows)

    scenario_rows: list[dict[str, object]] = []
    for (scenario_id, method), frame in replicate_summary.groupby(["scenario_id", "method"], observed=False):
        discoveries = int(frame["discoveries"].sum())
        true_positives = int(frame["true_positives"].sum())
        false_positives = int(frame["false_positives"].sum())
        raw_calls = int(frame["raw_calls"].sum())
        raw_true_positives = int(frame["raw_true_positives"].sum())
        raw_false_positives = int(frame["raw_false_positives"].sum())
        true_signals = int(frame["true_signals"].sum())
        null_motifs = int(frame["null_motifs"].sum())
        scenario_rows.append(
            {
                "scenario_id": str(scenario_id),
                "method": str(method),
                "n_replicates": int(frame.shape[0]),
                "true_signals_per_replicate": float(frame["true_signals"].mean()),
                "null_motifs_per_replicate": float(frame["null_motifs"].mean()),
                "mean_discoveries": float(frame["discoveries"].mean()),
                "mean_true_positives": float(frame["true_positives"].mean()),
                "mean_false_positives": float(frame["false_positives"].mean()),
                "empirical_fdr": float(false_positives / max(discoveries, 1)),
                "empirical_power": float(true_positives / max(true_signals, 1)) if true_signals > 0 else np.nan,
                "null_rejection_rate": float(false_positives / max(null_motifs, 1)) if null_motifs > 0 else np.nan,
                "mean_replicate_fdr": float(frame["empirical_fdr"].mean()),
                "mean_replicate_power": float(frame["empirical_power"].mean()),
                "raw_empirical_fdr": float(raw_false_positives / max(raw_calls, 1)),
                "raw_empirical_power": float(raw_true_positives / max(true_signals, 1)) if true_signals > 0 else np.nan,
                "signal_mean_pvalue": float(frame["signal_mean_pvalue"].mean()),
                "null_mean_pvalue": float(frame["null_mean_pvalue"].mean()),
            }
        )
    scenario_summary = pd.DataFrame(scenario_rows)
    return replicate_summary, scenario_summary


def _simulate_counts_numpy(
    *,
    baseline_logits: np.ndarray,
    true_logit_effect: np.ndarray,
    condition_case: np.ndarray,
    patches_per_sample: int,
    spots_per_patch: int,
    sample_random_effect_sd: float,
    patch_random_effect_sd: float,
    random_state: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n_samples = int(condition_case.shape[0])
    n_motifs = int(baseline_logits.shape[0])
    sample_effect = rng.normal(
        loc=0.0,
        scale=sample_random_effect_sd,
        size=(n_samples, n_motifs),
    ).astype(np.float32, copy=False)
    patch_effect = rng.normal(
        loc=0.0,
        scale=patch_random_effect_sd,
        size=(n_samples, patches_per_sample, n_motifs),
    ).astype(np.float32, copy=False)
    logits = (
        baseline_logits[None, None, :]
        + true_logit_effect[None, None, :] * condition_case[:, None, None]
        + sample_effect[:, None, :]
        + patch_effect
    )
    probabilities = _sigmoid(logits)
    patch_counts = rng.binomial(int(spots_per_patch), probabilities).astype(np.int64, copy=False)
    sample_counts = patch_counts.sum(axis=1, dtype=np.int64)
    control_mask = condition_case == 0.0
    case_mask = condition_case == 1.0
    mean_probability_control = (
        probabilities[control_mask].mean(axis=(0, 1)).astype(np.float32, copy=False)
        if np.any(control_mask)
        else np.full(n_motifs, np.nan, dtype=np.float32)
    )
    mean_probability_case = (
        probabilities[case_mask].mean(axis=(0, 1)).astype(np.float32, copy=False)
        if np.any(case_mask)
        else np.full(n_motifs, np.nan, dtype=np.float32)
    )
    return sample_counts, mean_probability_control, mean_probability_case


def _simulate_counts_torch(
    *,
    baseline_logits: np.ndarray,
    true_logit_effect: np.ndarray,
    condition_case: np.ndarray,
    patches_per_sample: int,
    spots_per_patch: int,
    sample_random_effect_sd: float,
    patch_random_effect_sd: float,
    random_state: int,
    runtime_info: SimulationRuntimeInfo,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if torch is None:  # pragma: no cover - defensive fallback
        return _simulate_counts_numpy(
            baseline_logits=baseline_logits,
            true_logit_effect=true_logit_effect,
            condition_case=condition_case,
            patches_per_sample=patches_per_sample,
            spots_per_patch=spots_per_patch,
            sample_random_effect_sd=sample_random_effect_sd,
            patch_random_effect_sd=patch_random_effect_sd,
            random_state=random_state,
        )

    device = torch.device(runtime_info.device)
    generator = torch.Generator(device=runtime_info.device)
    generator.manual_seed(int(random_state))

    baseline = torch.as_tensor(baseline_logits, dtype=torch.float32, device=device)
    effect = torch.as_tensor(true_logit_effect, dtype=torch.float32, device=device)
    condition = torch.as_tensor(condition_case, dtype=torch.float32, device=device).view(-1, 1, 1)
    sample_effect = torch.randn(
        (condition.shape[0], baseline.shape[0]),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) * float(sample_random_effect_sd)
    patch_effect = torch.randn(
        (condition.shape[0], int(patches_per_sample), baseline.shape[0]),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) * float(patch_random_effect_sd)
    logits = baseline.view(1, 1, -1) + effect.view(1, 1, -1) * condition + sample_effect.unsqueeze(1) + patch_effect
    probabilities = torch.sigmoid(logits)

    patch_counts = torch.zeros_like(probabilities, dtype=torch.int32)
    remaining_spots = int(spots_per_patch)
    chunk_size = int(min(max(spots_per_patch, 1), 256))
    while remaining_spots > 0:
        current_chunk = int(min(remaining_spots, chunk_size))
        uniform = torch.rand(
            (*probabilities.shape, current_chunk),
            generator=generator,
            device=device,
            dtype=torch.float32,
        )
        patch_counts = patch_counts + (uniform < probabilities.unsqueeze(-1)).sum(dim=-1, dtype=torch.int32)
        remaining_spots -= current_chunk

    sample_counts = patch_counts.sum(dim=1, dtype=torch.int64)
    control_mask = torch.as_tensor(condition_case == 0.0, device=device)
    case_mask = torch.as_tensor(condition_case == 1.0, device=device)
    if bool(torch.any(control_mask)):
        mean_probability_control = probabilities[control_mask].mean(dim=(0, 1))
    else:
        mean_probability_control = torch.full((baseline.shape[0],), float("nan"), dtype=torch.float32, device=device)
    if bool(torch.any(case_mask)):
        mean_probability_case = probabilities[case_mask].mean(dim=(0, 1))
    else:
        mean_probability_case = torch.full((baseline.shape[0],), float("nan"), dtype=torch.float32, device=device)

    sample_counts_np = sample_counts.detach().cpu().numpy()
    mean_probability_control_np = mean_probability_control.detach().cpu().numpy()
    mean_probability_case_np = mean_probability_case.detach().cpu().numpy()
    if runtime_info.cuda_available:
        torch.cuda.empty_cache()
    return sample_counts_np, mean_probability_control_np, mean_probability_case_np


def _logit(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return np.log(values / np.clip(1.0 - values, 1.0e-6, None)).astype(np.float32, copy=False)


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (1.0 / (1.0 + np.exp(-values))).astype(np.float32, copy=False)
