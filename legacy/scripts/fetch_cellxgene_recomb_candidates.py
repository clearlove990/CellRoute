from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests


CELLXGENE_DATASETS = (
    {
        "slug": "cellxgene_human_kidney_nonpt",
        "dataset_id": "a221d3e2-63f5-4761-9809-d821e7d21b73",
        "asset_id": "86ae4a4d-24ee-4024-9829-c4cae11fd445",
        "label": "Mature kidney dataset: non PT parenchyma",
    },
    {
        "slug": "cellxgene_immune_five_donors",
        "dataset_id": "aa2f9200-9880-4d3f-811a-0c224e3326c6",
        "asset_id": "0cc92d05-6e33-48f8-b85c-e5120ba0ac45",
        "label": "Immune cells from five healthy donors",
    },
    {
        "slug": "cellxgene_unciliated_epithelial_five_donors",
        "dataset_id": "e87a0cf7-5c3f-4a21-afb6-c516b6f8a36a",
        "asset_id": "985c345f-4dcf-46fa-a2d4-68151b0136ef",
        "label": "Unciliated epithelial cells from five healthy donors",
    },
    {
        "slug": "cellxgene_mouse_kidney_aging_10x",
        "dataset_id": "b4adefd1-0b4f-459f-9600-4723c0783c14",
        "asset_id": "76cf7ae4-fb94-4330-b4ea-188a6a97823c",
        "label": "Kidney - A single-cell transcriptomic atlas characterizes ageing tissues in the mouse - 10x",
    },
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fetch curated CELLxGENE datasets for RECOMB/ISMB benchmarking.")
    parser.add_argument("--output-root", type=str, default="data/real_inputs")
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--datasets",
        type=str,
        default=",".join(dataset["slug"] for dataset in CELLXGENE_DATASETS),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    requested = {part.strip() for part in args.datasets.split(",") if part.strip()}

    manifest = []
    for dataset in CELLXGENE_DATASETS:
        if dataset["slug"] not in requested:
            continue
        dataset_dir = output_root / dataset["slug"]
        dataset_dir.mkdir(parents=True, exist_ok=True)
        asset_info = requests.get(
            f"https://api.cellxgene.cziscience.com/dp/v1/datasets/{dataset['dataset_id']}/asset/{dataset['asset_id']}",
            timeout=120,
        )
        asset_info.raise_for_status()
        asset_payload = asset_info.json()

        target_path = dataset_dir / "source.h5ad"
        if args.force or not target_path.exists():
            download_file(asset_payload["url"], target_path)

        metadata = {
            **dataset,
            "download_url": asset_payload["url"],
            "file_size": asset_payload.get("file_size"),
            "target_path": str(target_path),
        }
        (dataset_dir / "source_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        manifest.append(metadata)
        print(f"Fetched {dataset['slug']} -> {target_path}")

    (output_root / "cellxgene_recomb_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def download_file(url: str, target_path: Path) -> None:
    response = requests.get(url, stream=True, timeout=120)
    response.raise_for_status()
    with target_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            if not chunk:
                continue
            handle.write(chunk)


if __name__ == "__main__":
    main()
