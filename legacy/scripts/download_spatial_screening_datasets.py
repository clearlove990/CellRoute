from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hvg_research.spatial_screening import (
    ROOT as PROJECT_ROOT,
    SPATIAL_RAW_DIR,
    download_selected_datasets,
    summarize_download_records,
    write_catalog_files,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the spatial screening catalog and download the selected datasets.")
    parser.add_argument("--catalog-only", action="store_true", help="Write catalog files without downloading data.")
    parser.add_argument("--all-candidates", action="store_true", help="Download every candidate instead of only the selected datasets.")
    parser.add_argument("--force", action="store_true", help="Re-download and re-extract artifacts even if present.")
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--chunk-size-mb", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_catalog_files()
    print(f"Catalog written under: {PROJECT_ROOT / 'data_catalog'}")
    if args.catalog_only:
        return

    records = download_selected_datasets(
        force=args.force,
        selected_only=not args.all_candidates,
        timeout_sec=args.timeout_sec,
        chunk_size_mb=args.chunk_size_mb,
    )
    print(f"Spatial raw data directory: {SPATIAL_RAW_DIR}")
    print(summarize_download_records(records))


if __name__ == "__main__":
    main()
