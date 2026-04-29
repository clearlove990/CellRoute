from __future__ import annotations

import numpy as np
import torch


def _is_cuda_oom(exc: Exception) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    message = str(exc).lower()
    return "cuda" in message and "out of memory" in message


def _safe_empty_cuda_cache() -> None:
    if not torch.cuda.is_available():
        return
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass


def normalize_log1p(counts: np.ndarray, target_sum: float = 1e4) -> np.ndarray:
    counts_f32 = np.asarray(counts, dtype=np.float32)
    library = counts_f32.sum(axis=1, keepdims=True, dtype=np.float32)
    library = np.maximum(library, np.float32(1.0))
    normalized = counts_f32 / library * np.float32(target_sum)
    return np.log1p(normalized, dtype=np.float32)


def _zscore(x: np.ndarray) -> np.ndarray:
    std = np.std(x)
    if std < 1e-8:
        return np.zeros_like(x, dtype=np.float64)
    return (x - np.mean(x)) / std


def score_variance(counts: np.ndarray) -> np.ndarray:
    x = normalize_log1p(counts)
    return _zscore(np.var(x, axis=0))


def score_fano(counts: np.ndarray) -> np.ndarray:
    x = normalize_log1p(counts)
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)
    fano = var / np.maximum(mean, 1e-6)
    return _zscore(fano)


def score_mean_var_residual(counts: np.ndarray) -> np.ndarray:
    x = normalize_log1p(counts)
    mean = np.mean(x, axis=0)
    var = np.var(x, axis=0)

    log_mean = np.log1p(mean)
    log_var = np.log1p(var)
    coeff = np.polyfit(log_mean, log_var, deg=2)
    expected = np.polyval(coeff, log_mean)
    residual = log_var - expected
    return _zscore(residual)


def score_analytic_pearson_residual_hvg(
    counts: np.ndarray,
    *,
    theta: float = 100.0,
    clip: float | None = None,
    gene_chunk_size: int = 1024,
    ) -> np.ndarray:
    counts_f32 = np.asarray(counts, dtype=np.float32)
    n_cells, n_genes = counts_f32.shape
    if n_cells == 0 or n_genes == 0:
        return np.zeros(n_genes, dtype=np.float64)

    clip_value = float(np.sqrt(n_cells) if clip is None else clip)
    total_sum = float(np.sum(counts_f32, dtype=np.float64))
    if total_sum <= 0.0:
        return np.zeros(n_genes, dtype=np.float64)

    if torch.cuda.is_available():
        try:
            return _score_analytic_pearson_residual_torch(
                counts=counts_f32,
                theta=float(theta),
                clip_value=clip_value,
                gene_chunk_size=gene_chunk_size,
            )
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            _safe_empty_cuda_cache()
    return _score_analytic_pearson_residual_numpy(
        counts=counts_f32,
        theta=float(theta),
        clip_value=clip_value,
        gene_chunk_size=gene_chunk_size,
    )


def _score_analytic_pearson_residual_numpy(
    *,
    counts: np.ndarray,
    theta: float,
    clip_value: float,
    gene_chunk_size: int,
) -> np.ndarray:
    n_cells, n_genes = counts.shape
    row_sum = np.sum(counts, axis=1, dtype=np.float64).astype(np.float32, copy=False)
    gene_sum = np.sum(counts, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    total_sum = max(float(np.sum(gene_sum, dtype=np.float64)), 1.0)
    residual_var = np.zeros(n_genes, dtype=np.float64)

    for start in range(0, n_genes, gene_chunk_size):
        end = min(start + gene_chunk_size, n_genes)
        count_chunk = counts[:, start:end].astype(np.float32, copy=False)
        expected = np.outer(row_sum, gene_sum[start:end]).astype(np.float32, copy=False)
        expected /= np.float32(total_sum)
        denom = np.sqrt(expected + (expected * expected) / np.float32(theta)).astype(np.float32, copy=False)
        residual = (count_chunk - expected) / np.maximum(denom, np.float32(1e-6))
        residual = np.clip(residual, -clip_value, clip_value)
        residual_var[start:end] = np.var(residual, axis=0, dtype=np.float64)
    return _zscore(residual_var)


def score_seurat_v3_like_hvg(
    counts: np.ndarray,
    *,
    clip: float | None = None,
    gene_chunk_size: int = 1024,
) -> np.ndarray:
    counts_f32 = np.asarray(counts, dtype=np.float32)
    n_cells, n_genes = counts_f32.shape
    if n_cells == 0 or n_genes == 0:
        return np.zeros(n_genes, dtype=np.float64)

    clip_value = float(np.sqrt(n_cells) if clip is None else clip)
    if torch.cuda.is_available():
        try:
            return _score_seurat_v3_like_torch(
                counts=counts_f32,
                clip_value=clip_value,
                gene_chunk_size=gene_chunk_size,
            )
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            _safe_empty_cuda_cache()
    return _score_seurat_v3_like_numpy(
        counts=counts_f32,
        clip_value=clip_value,
        gene_chunk_size=gene_chunk_size,
    )


def score_multinomial_deviance_hvg(
    counts: np.ndarray,
    *,
    gene_chunk_size: int = 512,
) -> np.ndarray:
    counts_f32 = np.asarray(counts, dtype=np.float32)
    n_cells, n_genes = counts_f32.shape
    if n_cells == 0 or n_genes == 0:
        return np.zeros(n_genes, dtype=np.float64)

    total_sum = float(np.sum(counts_f32, dtype=np.float64))
    if total_sum <= 0.0:
        return np.zeros(n_genes, dtype=np.float64)

    if torch.cuda.is_available():
        try:
            return _score_multinomial_deviance_torch(
                counts=counts_f32,
                gene_chunk_size=gene_chunk_size,
            )
        except Exception as exc:
            if not _is_cuda_oom(exc):
                raise
            _safe_empty_cuda_cache()
    return _score_multinomial_deviance_numpy(
        counts=counts_f32,
        gene_chunk_size=gene_chunk_size,
    )


def score_scran_like_model_gene_var(
    counts: np.ndarray,
    *,
    smooth_window: int | None = None,
) -> np.ndarray:
    x = normalize_log1p(counts)
    if x.shape[1] < 8:
        return score_mean_var_residual(counts)

    mean = np.mean(x, axis=0, dtype=np.float64)
    var = np.var(x, axis=0, dtype=np.float64)
    log_mean = np.log1p(np.clip(mean, 1e-8, None))
    log_var = np.log1p(np.clip(var, 1e-8, None))

    order = np.argsort(log_mean, kind="mergesort")
    ordered_var = log_var[order]
    n_genes = ordered_var.shape[0]

    if smooth_window is None:
        smooth_window = max(11, min(301, int(np.ceil(n_genes / 20.0))))
    if smooth_window % 2 == 0:
        smooth_window += 1
    smooth_window = max(3, min(smooth_window, n_genes if n_genes % 2 == 1 else max(n_genes - 1, 3)))
    if smooth_window <= 3:
        return _zscore(log_var - np.polyval(np.polyfit(log_mean, log_var, deg=2), log_mean))

    pad = smooth_window // 2
    padded = np.pad(ordered_var, (pad, pad), mode="edge")
    kernel = np.full(smooth_window, 1.0 / smooth_window, dtype=np.float64)
    smoothed = np.convolve(padded, kernel, mode="valid")

    expected = np.empty_like(log_var, dtype=np.float64)
    expected[order] = smoothed[:n_genes]
    bio_component = log_var - expected
    return _zscore(bio_component)


def _score_analytic_pearson_residual_torch(
    *,
    counts: np.ndarray,
    theta: float,
    clip_value: float,
    gene_chunk_size: int,
) -> np.ndarray:
    device = torch.device("cuda")
    n_cells, n_genes = counts.shape
    row_sum_np = np.sum(counts, axis=1, dtype=np.float64).astype(np.float32, copy=False)
    gene_sum_np = np.sum(counts, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    total_sum = max(float(np.sum(gene_sum_np, dtype=np.float64)), 1.0)
    row_sum = torch.from_numpy(row_sum_np).to(device, non_blocking=True).unsqueeze(1)

    counts_bytes = counts.nbytes
    total_memory = torch.cuda.get_device_properties(device).total_memory
    use_full_matrix = counts_bytes <= int(total_memory * 0.25)

    if use_full_matrix:
        count_tensor = torch.as_tensor(counts, dtype=torch.float32, device=device)
        gene_sum = count_tensor.sum(dim=0, keepdim=True)
        expected = row_sum * gene_sum / float(total_sum)
        denom = torch.sqrt(expected + expected.square() / float(theta))
        residual = (count_tensor - expected) / denom.clamp_min(1e-6)
        residual = residual.clamp(min=-clip_value, max=clip_value)
        residual_var = residual.var(dim=0, unbiased=False)
        return _zscore(residual_var.detach().cpu().numpy())

    residual_var = np.zeros(n_genes, dtype=np.float64)
    for start in range(0, n_genes, gene_chunk_size):
        end = min(start + gene_chunk_size, n_genes)
        count_chunk = torch.as_tensor(counts[:, start:end], dtype=torch.float32, device=device)
        gene_sum_chunk = torch.from_numpy(gene_sum_np[start:end]).to(device, non_blocking=True).unsqueeze(0)
        expected = row_sum * gene_sum_chunk / float(total_sum)
        denom = torch.sqrt(expected + expected.square() / float(theta))
        residual = (count_chunk - expected) / denom.clamp_min(1e-6)
        residual = residual.clamp(min=-clip_value, max=clip_value)
        residual_var[start:end] = residual.var(dim=0, unbiased=False).detach().cpu().numpy()
    return _zscore(residual_var)


def _score_seurat_v3_like_numpy(
    *,
    counts: np.ndarray,
    clip_value: float,
    gene_chunk_size: int,
) -> np.ndarray:
    mean = np.mean(counts, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    var = np.var(counts, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    regularized_std = _fit_regularized_std_numpy(mean=mean, var=var)
    clip_limit = np.maximum(regularized_std * np.float32(clip_value), np.float32(1e-6))
    normalized_var = np.zeros(counts.shape[1], dtype=np.float64)

    for start in range(0, counts.shape[1], gene_chunk_size):
        end = min(start + gene_chunk_size, counts.shape[1])
        chunk = counts[:, start:end].astype(np.float32, copy=False)
        centered = chunk - mean[start:end]
        centered = np.clip(centered, -clip_limit[start:end], clip_limit[start:end])
        standardized = centered / np.maximum(regularized_std[start:end], np.float32(1e-6))
        normalized_var[start:end] = np.var(standardized, axis=0, dtype=np.float64)
    return _zscore(normalized_var)


def _score_seurat_v3_like_torch(
    *,
    counts: np.ndarray,
    clip_value: float,
    gene_chunk_size: int,
) -> np.ndarray:
    device = torch.device("cuda")
    counts_bytes = counts.nbytes
    total_memory = torch.cuda.get_device_properties(device).total_memory
    use_full_matrix = counts_bytes <= int(total_memory * 0.25)

    if use_full_matrix:
        count_tensor = torch.as_tensor(counts, dtype=torch.float32, device=device)
        mean = count_tensor.mean(dim=0)
        var = count_tensor.var(dim=0, unbiased=False)
        regularized_std = _fit_regularized_std_torch(mean=mean, var=var)
        clip_limit = torch.clamp(regularized_std * float(clip_value), min=1e-6)
        centered = count_tensor - mean
        centered = torch.minimum(torch.maximum(centered, -clip_limit), clip_limit)
        standardized = centered / regularized_std.clamp_min(1e-6)
        normalized_var = standardized.var(dim=0, unbiased=False)
        return _zscore(normalized_var.detach().cpu().numpy())

    mean_np = np.zeros(counts.shape[1], dtype=np.float32)
    var_np = np.zeros(counts.shape[1], dtype=np.float32)
    for start in range(0, counts.shape[1], gene_chunk_size):
        end = min(start + gene_chunk_size, counts.shape[1])
        count_chunk = torch.as_tensor(counts[:, start:end], dtype=torch.float32, device=device)
        mean_np[start:end] = count_chunk.mean(dim=0).detach().cpu().numpy()
        var_np[start:end] = count_chunk.var(dim=0, unbiased=False).detach().cpu().numpy()

    regularized_std_np = _fit_regularized_std_numpy(mean=mean_np, var=var_np)
    clip_limit_np = np.maximum(regularized_std_np * np.float32(clip_value), np.float32(1e-6))
    normalized_var = np.zeros(counts.shape[1], dtype=np.float64)
    for start in range(0, counts.shape[1], gene_chunk_size):
        end = min(start + gene_chunk_size, counts.shape[1])
        count_chunk = torch.as_tensor(counts[:, start:end], dtype=torch.float32, device=device)
        mean_chunk = torch.from_numpy(mean_np[start:end]).to(device, non_blocking=True)
        std_chunk = torch.from_numpy(regularized_std_np[start:end]).to(device, non_blocking=True)
        clip_chunk = torch.from_numpy(clip_limit_np[start:end]).to(device, non_blocking=True)
        centered = count_chunk - mean_chunk
        centered = torch.minimum(torch.maximum(centered, -clip_chunk), clip_chunk)
        standardized = centered / std_chunk.clamp_min(1e-6)
        normalized_var[start:end] = standardized.var(dim=0, unbiased=False).detach().cpu().numpy()
    return _zscore(normalized_var)


def _fit_regularized_std_numpy(*, mean: np.ndarray, var: np.ndarray) -> np.ndarray:
    mean_f64 = np.asarray(mean, dtype=np.float64)
    var_f64 = np.asarray(var, dtype=np.float64)
    log_mean = np.log10(np.clip(mean_f64, 1e-6, None))
    log_var = np.log10(np.clip(var_f64, 1e-6, None))
    valid = np.isfinite(log_mean) & np.isfinite(log_var)
    if int(valid.sum()) < 3:
        return np.sqrt(np.clip(var_f64, 1e-6, None)).astype(np.float32, copy=False)

    coeff = np.polyfit(log_mean[valid], log_var[valid], deg=2)
    fitted_log_var = np.polyval(coeff, log_mean)
    fitted_var = np.power(10.0, fitted_log_var, dtype=np.float64)
    return np.sqrt(np.clip(fitted_var, 1e-6, None)).astype(np.float32, copy=False)


def _fit_regularized_std_torch(*, mean: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    log_mean = torch.log10(mean.clamp_min(1e-6))
    log_var = torch.log10(var.clamp_min(1e-6))
    valid = torch.isfinite(log_mean) & torch.isfinite(log_var)
    if int(valid.sum().item()) < 3:
        return torch.sqrt(var.clamp_min(1e-6))

    x = log_mean[valid]
    y = log_var[valid]
    design = torch.stack((x.square(), x, torch.ones_like(x)), dim=1)
    coeff = torch.linalg.lstsq(design, y.unsqueeze(1)).solution.squeeze(1)
    full_design = torch.stack((log_mean.square(), log_mean, torch.ones_like(log_mean)), dim=1)
    fitted_log_var = full_design @ coeff
    return torch.sqrt(torch.pow(torch.tensor(10.0, device=mean.device), fitted_log_var).clamp_min(1e-6))


def _score_multinomial_deviance_numpy(
    *,
    counts: np.ndarray,
    gene_chunk_size: int,
) -> np.ndarray:
    library = np.sum(counts, axis=1, dtype=np.float64).astype(np.float32, copy=False)
    gene_sum = np.sum(counts, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    total_sum = max(float(np.sum(gene_sum, dtype=np.float64)), 1.0)
    p = np.clip(gene_sum / np.float32(total_sum), np.float32(1e-6), np.float32(1.0 - 1e-6))
    deviance = np.zeros(counts.shape[1], dtype=np.float64)

    for start in range(0, counts.shape[1], gene_chunk_size):
        end = min(start + gene_chunk_size, counts.shape[1])
        count_chunk = counts[:, start:end].astype(np.float32, copy=False)
        p_chunk = p[start:end]
        expected = library[:, None] * p_chunk[None, :]
        complement = np.maximum(library[:, None] - count_chunk, np.float32(0.0))
        expected_complement = library[:, None] * np.maximum(np.float32(1.0) - p_chunk[None, :], np.float32(1e-6))
        term1 = np.where(
            count_chunk > 0,
            count_chunk * np.log(np.maximum(count_chunk, np.float32(1e-6)) / np.maximum(expected, np.float32(1e-6))),
            np.float32(0.0),
        )
        term2 = np.where(
            complement > 0,
            complement
            * np.log(
                np.maximum(complement, np.float32(1e-6)) / np.maximum(expected_complement, np.float32(1e-6))
            ),
            np.float32(0.0),
        )
        deviance[start:end] = np.sum(2.0 * (term1 + term2), axis=0, dtype=np.float64)
    return _zscore(deviance)


def _score_multinomial_deviance_torch(
    *,
    counts: np.ndarray,
    gene_chunk_size: int,
) -> np.ndarray:
    device = torch.device("cuda")
    library_np = np.sum(counts, axis=1, dtype=np.float64).astype(np.float32, copy=False)
    gene_sum_np = np.sum(counts, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    total_sum = max(float(np.sum(gene_sum_np, dtype=np.float64)), 1.0)
    library = torch.from_numpy(library_np).to(device, non_blocking=True).unsqueeze(1)
    p = np.clip(gene_sum_np / np.float32(total_sum), np.float32(1e-6), np.float32(1.0 - 1e-6))
    deviance = np.zeros(counts.shape[1], dtype=np.float64)

    for start in range(0, counts.shape[1], gene_chunk_size):
        end = min(start + gene_chunk_size, counts.shape[1])
        count_chunk = torch.as_tensor(counts[:, start:end], dtype=torch.float32, device=device)
        p_chunk = torch.from_numpy(p[start:end]).to(device, non_blocking=True).unsqueeze(0)
        expected = library * p_chunk
        complement = (library - count_chunk).clamp_min(0.0)
        expected_complement = library * (1.0 - p_chunk).clamp_min(1e-6)
        term1 = torch.where(
            count_chunk > 0,
            count_chunk * torch.log(count_chunk.clamp_min(1e-6) / expected.clamp_min(1e-6)),
            torch.zeros_like(count_chunk),
        )
        term2 = torch.where(
            complement > 0,
            complement * torch.log(complement.clamp_min(1e-6) / expected_complement.clamp_min(1e-6)),
            torch.zeros_like(complement),
        )
        deviance[start:end] = (2.0 * (term1 + term2).sum(dim=0)).detach().cpu().numpy()
    return _zscore(deviance)
