from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hvg_research.spatial_preprocess import run_spatial_dataset_screening


def main() -> None:
    processed, summary_df = run_spatial_dataset_screening()
    print(f"Processed datasets: {len(processed)}")
    for item in processed:
        print(
            f"{item.dataset_id}: samples={item.n_samples} spots={item.n_spots} "
            f"genes={item.n_genes} path={item.processed_path}"
        )
    print("QC rows:", len(summary_df))


if __name__ == "__main__":
    main()
