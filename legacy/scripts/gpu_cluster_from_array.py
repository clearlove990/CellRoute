from __future__ import annotations

import argparse
import numpy as np
import torch
from sklearn.cluster import KMeans


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute basic clusters from a dense array using torch PCA and KMeans.")
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--n-components", type=int, default=20)
    parser.add_argument("--n-clusters", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    x_array = np.load(args.input).astype(np.float32, copy=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_tensor = torch.from_numpy(x_array).to(device, non_blocking=True)
    x_tensor = x_tensor - x_tensor.mean(dim=0, keepdim=True)
    max_components = max(2, min(args.n_components, x_tensor.shape[0] - 1, x_tensor.shape[1] - 1))
    u_mat, singular_vals, _ = torch.pca_lowrank(x_tensor, q=max_components, center=False)
    embedding = (u_mat[:, :max_components] * singular_vals[:max_components]).cpu().numpy()
    labels = KMeans(n_clusters=args.n_clusters, random_state=7, n_init=10).fit_predict(embedding)
    np.save(args.output, labels.astype(np.int64))


if __name__ == "__main__":
    main()
