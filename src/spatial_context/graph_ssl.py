from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ModuleNotFoundError:  # pragma: no cover - torch is expected in this repo
    torch = None
    nn = None
    F = None

from .neighborhood import RuntimeInfo


@dataclass(frozen=True)
class GraphSSLConfig:
    hidden_dim: int = 80
    embedding_dim: int = 24
    dropout: float = 0.10
    feature_dropout: float = 0.18
    edge_dropout: float = 0.12
    neighborhood_mask: float = 0.08
    noise_std: float = 0.025
    temperature: float = 0.20
    epochs: int = 48
    batch_size: int = 1024
    contrastive_samples: int = 4096
    learning_rate: float = 3.0e-3
    weight_decay: float = 1.0e-4
    grad_clip_norm: float = 5.0
    patience: int = 10
    min_delta: float = 1.0e-4
    neighbor_weight: float = 0.35
    sample_balance_weight: float = 0.18
    condition_spread_weight: float = 0.05
    random_state: int = 7

    @classmethod
    def from_dict(cls, payload: dict[str, Any] | None) -> "GraphSSLConfig":
        if payload is None:
            return cls()
        return cls(**payload)


@dataclass(frozen=True)
class GraphSSLResult:
    embedding: np.ndarray
    training_history: pd.DataFrame
    model_metadata: dict[str, Any]


if torch is not None:

    class _GraphContrastiveEncoder(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int, dropout: float) -> None:
            super().__init__()
            self.input_proj = nn.Linear(input_dim, hidden_dim)
            self.context_proj = nn.Linear(hidden_dim, hidden_dim)
            self.output_proj = nn.Linear(hidden_dim, embedding_dim)
            self.norm1 = nn.LayerNorm(hidden_dim)
            self.norm2 = nn.LayerNorm(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
            hidden = self.input_proj(self.dropout(x))
            hidden = F.gelu(hidden)
            hidden = self.norm1(hidden + torch.sparse.mm(adjacency, hidden))
            hidden = self.context_proj(self.dropout(hidden))
            hidden = F.gelu(hidden)
            hidden = self.norm2(hidden + torch.sparse.mm(adjacency, hidden))
            embedding = self.output_proj(self.dropout(hidden))
            return F.normalize(embedding, dim=1)

else:

    class _GraphContrastiveEncoder:  # pragma: no cover - only used when torch is unavailable
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            raise ModuleNotFoundError("PyTorch is required for graph SSL training.")


def train_graph_context_embedding(
    feature_frame: pd.DataFrame | np.ndarray,
    adjacency: sparse.csr_matrix,
    *,
    runtime_info: RuntimeInfo,
    sample_ids: np.ndarray,
    condition_ids: np.ndarray | None = None,
    config: GraphSSLConfig | None = None,
) -> GraphSSLResult:
    if torch is None:
        raise ModuleNotFoundError("PyTorch is required for graph SSL training.")
    config = config or GraphSSLConfig()
    np.random.seed(config.random_state)
    torch.manual_seed(config.random_state)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.random_state)

    device = "cuda" if runtime_info.cuda_available and torch.cuda.is_available() else "cpu"
    feature_array = _standardize_features(feature_frame)
    n_nodes, n_features = feature_array.shape

    coo = adjacency.tocoo()
    rows = np.asarray(coo.row, dtype=np.int64)
    cols = np.asarray(coo.col, dtype=np.int64)
    values = np.asarray(coo.data, dtype=np.float32)
    base_adjacency = _build_sparse_tensor(
        rows=rows,
        cols=cols,
        values=values,
        shape=adjacency.shape,
        device=device,
    )

    feature_tensor = torch.as_tensor(feature_array, dtype=torch.float32, device=device)
    sample_code_array = np.asarray(pd.Categorical(sample_ids.astype(str)).codes, dtype=np.int64).copy()
    sample_codes = torch.as_tensor(sample_code_array, dtype=torch.long, device=device)
    if condition_ids is None:
        condition_codes = torch.zeros(sample_codes.shape[0], dtype=torch.long, device=device)
    else:
        condition_code_array = np.asarray(pd.Categorical(condition_ids.astype(str)).codes, dtype=np.int64).copy()
        condition_codes = torch.as_tensor(condition_code_array, dtype=torch.long, device=device)

    model = _GraphContrastiveEncoder(
        input_dim=n_features,
        hidden_dim=config.hidden_dim,
        embedding_dim=config.embedding_dim,
        dropout=config.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    rng = np.random.default_rng(config.random_state)
    history_rows: list[dict[str, float]] = []
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, config.epochs + 1):
        model.train()
        view_adj_1 = _dropout_adjacency(
            rows=rows,
            cols=cols,
            values=values,
            shape=adjacency.shape,
            edge_dropout=config.edge_dropout,
            device=device,
            rng=rng,
        )
        view_adj_2 = _dropout_adjacency(
            rows=rows,
            cols=cols,
            values=values,
            shape=adjacency.shape,
            edge_dropout=config.edge_dropout,
            device=device,
            rng=rng,
        )
        view_x_1 = _augment_features(
            feature_tensor,
            adjacency=view_adj_1,
            feature_dropout=config.feature_dropout,
            neighborhood_mask=config.neighborhood_mask,
            noise_std=config.noise_std,
        )
        view_x_2 = _augment_features(
            feature_tensor,
            adjacency=view_adj_2,
            feature_dropout=config.feature_dropout,
            neighborhood_mask=config.neighborhood_mask,
            noise_std=config.noise_std,
        )
        embedding_1 = model(view_x_1, view_adj_1)
        embedding_2 = model(view_x_2, view_adj_2)

        contrastive_loss = _batched_info_nce(
            embedding_1,
            embedding_2,
            temperature=config.temperature,
            batch_size=config.batch_size,
            max_samples=config.contrastive_samples,
        )
        neighbor_loss = 0.5 * (
            _neighbor_consistency_loss(embedding_1, base_adjacency)
            + _neighbor_consistency_loss(embedding_2, base_adjacency)
        )
        sample_balance_loss = 0.5 * (
            _sample_balance_loss(embedding_1, sample_codes, condition_codes)
            + _sample_balance_loss(embedding_2, sample_codes, condition_codes)
        )
        condition_spread = _condition_spread_bonus((embedding_1 + embedding_2) * 0.5, condition_codes)

        total_loss = (
            contrastive_loss
            + config.neighbor_weight * neighbor_loss
            + config.sample_balance_weight * sample_balance_loss
            - config.condition_spread_weight * condition_spread
        )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip_norm)
        optimizer.step()

        total_value = float(total_loss.detach().cpu().item())
        contrastive_value = float(contrastive_loss.detach().cpu().item())
        neighbor_value = float(neighbor_loss.detach().cpu().item())
        sample_balance_value = float(sample_balance_loss.detach().cpu().item())
        condition_spread_value = float(condition_spread.detach().cpu().item())
        history_rows.append(
            {
                "epoch": float(epoch),
                "total_loss": total_value,
                "contrastive_loss": contrastive_value,
                "neighbor_loss": neighbor_value,
                "sample_balance_loss": sample_balance_value,
                "condition_spread_bonus": condition_spread_value,
            }
        )

        if total_value + config.min_delta < best_loss:
            best_loss = total_value
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= config.patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        embedding = model(feature_tensor, base_adjacency).detach().cpu().numpy().astype(np.float32, copy=False)

    if device == "cuda":
        torch.cuda.empty_cache()

    history = pd.DataFrame(history_rows)
    metadata = {
        "device": device,
        "cuda_available": bool(runtime_info.cuda_available and torch.cuda.is_available()),
        "n_nodes": int(n_nodes),
        "n_features": int(n_features),
        "best_epoch": int(best_epoch),
        "best_total_loss": float(best_loss),
        "epochs_trained": int(history.shape[0]),
        "multi_gpu_mode": "single_device_full_graph",
        "config": asdict(config),
    }
    return GraphSSLResult(
        embedding=embedding,
        training_history=history,
        model_metadata=metadata,
    )


def _standardize_features(feature_frame: pd.DataFrame | np.ndarray) -> np.ndarray:
    values = feature_frame.to_numpy(dtype=np.float32, copy=True) if isinstance(feature_frame, pd.DataFrame) else np.asarray(feature_frame, dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    mean = values.mean(axis=0, keepdims=True)
    std = values.std(axis=0, keepdims=True)
    std = np.where(std > 1.0e-6, std, 1.0)
    return ((values - mean) / std).astype(np.float32, copy=False)


def _build_sparse_tensor(
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
    device: str,
) -> torch.Tensor:
    indices = torch.as_tensor(np.vstack((rows, cols)), dtype=torch.long, device=device)
    value_tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, value_tensor, size=shape, device=device).coalesce()


def _dropout_adjacency(
    *,
    rows: np.ndarray,
    cols: np.ndarray,
    values: np.ndarray,
    shape: tuple[int, int],
    edge_dropout: float,
    device: str,
    rng: np.random.Generator,
) -> torch.Tensor:
    if edge_dropout <= 0.0:
        return _build_sparse_tensor(rows=rows, cols=cols, values=values, shape=shape, device=device)
    keep_mask = rng.random(values.shape[0]) >= edge_dropout
    kept_values = values * keep_mask.astype(np.float32, copy=False)
    row_sum = np.bincount(rows, weights=kept_values, minlength=shape[0]).astype(np.float32, copy=False)
    row_sum = np.maximum(row_sum, 1.0e-6)
    normalized_values = kept_values / row_sum[rows]
    active = normalized_values > 0
    return _build_sparse_tensor(
        rows=rows[active],
        cols=cols[active],
        values=normalized_values[active].astype(np.float32, copy=False),
        shape=shape,
        device=device,
    )


def _augment_features(
    features: torch.Tensor,
    *,
    adjacency: torch.Tensor,
    feature_dropout: float,
    neighborhood_mask: float,
    noise_std: float,
) -> torch.Tensor:
    augmented = features
    if feature_dropout > 0.0:
        keep_columns = (torch.rand(features.shape[1], device=features.device) >= feature_dropout).to(features.dtype)
        scale = torch.clamp(keep_columns.mean(), min=1.0e-3)
        augmented = augmented * keep_columns.unsqueeze(0) / scale
    if noise_std > 0.0:
        augmented = augmented + noise_std * torch.randn_like(augmented)
    if neighborhood_mask > 0.0:
        node_mask = torch.rand(features.shape[0], device=features.device) < neighborhood_mask
        if bool(node_mask.any()):
            neighborhood_context = torch.sparse.mm(adjacency, augmented)
            augmented = augmented.clone()
            augmented[node_mask] = 0.5 * augmented[node_mask] + 0.5 * neighborhood_context[node_mask]
    return augmented


def _batched_info_nce(
    embedding_1: torch.Tensor,
    embedding_2: torch.Tensor,
    *,
    temperature: float,
    batch_size: int,
    max_samples: int,
) -> torch.Tensor:
    n_nodes = embedding_1.shape[0]
    if max_samples > 0 and max_samples < n_nodes:
        sample_idx = torch.randperm(n_nodes, device=embedding_1.device)[:max_samples]
    else:
        sample_idx = torch.arange(n_nodes, device=embedding_1.device)
    total = embedding_1.new_tensor(0.0)
    n_batches = 0
    for start in range(0, sample_idx.shape[0], batch_size):
        batch_idx = sample_idx[start : start + batch_size]
        view_1 = embedding_1[batch_idx]
        view_2 = embedding_2[batch_idx]
        logits_12 = torch.matmul(view_1, view_2.T) / temperature
        logits_21 = torch.matmul(view_2, view_1.T) / temperature
        target = torch.arange(batch_idx.shape[0], device=embedding_1.device)
        total = total + 0.5 * (F.cross_entropy(logits_12, target) + F.cross_entropy(logits_21, target))
        n_batches += 1
    return total / max(n_batches, 1)


def _neighbor_consistency_loss(embedding: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    neighbor_mean = torch.sparse.mm(adjacency, embedding)
    neighbor_mean = F.normalize(neighbor_mean, dim=1)
    return 1.0 - F.cosine_similarity(embedding, neighbor_mean, dim=1).mean()


def _sample_balance_loss(
    embedding: torch.Tensor,
    sample_codes: torch.Tensor,
    condition_codes: torch.Tensor,
) -> torch.Tensor:
    unique_samples = torch.unique(sample_codes)
    if unique_samples.numel() <= 1:
        return embedding.new_tensor(0.0)
    sample_centroids: list[torch.Tensor] = []
    sample_condition_codes: list[int] = []
    for sample_code in unique_samples.tolist():
        sample_mask = sample_codes == int(sample_code)
        if int(sample_mask.sum().item()) == 0:
            continue
        sample_centroids.append(embedding[sample_mask].mean(dim=0))
        sample_condition_codes.append(int(torch.mode(condition_codes[sample_mask]).values.item()))
    if len(sample_centroids) <= 1:
        return embedding.new_tensor(0.0)
    centroid_tensor = torch.stack(sample_centroids, dim=0)
    condition_tensor = torch.as_tensor(sample_condition_codes, dtype=torch.long, device=embedding.device)
    target_centroids = torch.empty_like(centroid_tensor)
    for condition_code in torch.unique(condition_tensor).tolist():
        condition_mask = condition_tensor == int(condition_code)
        target_centroids[condition_mask] = centroid_tensor[condition_mask].mean(dim=0, keepdim=True)
    return torch.mean(torch.sum(torch.square(centroid_tensor - target_centroids), dim=1))


def _condition_spread_bonus(embedding: torch.Tensor, condition_codes: torch.Tensor) -> torch.Tensor:
    unique_conditions = torch.unique(condition_codes)
    if unique_conditions.numel() <= 1:
        return embedding.new_tensor(0.0)
    centroids = []
    for condition_code in unique_conditions.tolist():
        condition_mask = condition_codes == int(condition_code)
        if int(condition_mask.sum().item()) == 0:
            continue
        centroids.append(embedding[condition_mask].mean(dim=0))
    if len(centroids) <= 1:
        return embedding.new_tensor(0.0)
    centroid_tensor = torch.stack(centroids, dim=0)
    return torch.pdist(centroid_tensor, p=2).mean()
