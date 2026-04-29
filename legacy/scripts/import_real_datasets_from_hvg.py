from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImportSpec:
    name: str
    labels_col: str | None
    batches_col: str | None
    source_subdir: str
    file_format: str = "mtx"
    transpose: bool = True
    files: tuple[str, ...] = (
        "dataset_manifest.csv",
        "obs.csv",
        "matrix/matrix.mtx",
        "matrix/genes.tsv",
        "matrix/barcodes.tsv",
    )


CURATED_DATASETS: dict[str, ImportSpec] = {
    "paul15": ImportSpec(
        name="paul15",
        labels_col="paul15_clusters",
        batches_col=None,
        source_subdir="datasets_public/paul15",
    ),
    "E-MTAB-4388": ImportSpec(
        name="E-MTAB-4388",
        labels_col="Sample Characteristic[cell type]",
        batches_col=None,
        source_subdir="datasets_public/E-MTAB-4388",
    ),
    "E-MTAB-4888": ImportSpec(
        name="E-MTAB-4888",
        labels_col="Sample Characteristic[cell type]",
        batches_col="Sample Characteristic[individual]",
        source_subdir="datasets_public/E-MTAB-4888",
    ),
    "E-MTAB-5061": ImportSpec(
        name="E-MTAB-5061",
        labels_col="Factor Value[inferred cell type - ontology labels]",
        batches_col="Sample Characteristic[individual]",
        source_subdir="datasets_public/E-MTAB-5061",
    ),
    "FBM_cite": ImportSpec(
        name="FBM_cite",
        labels_col="adt_celltype_label",
        batches_col=None,
        source_subdir="datasets_public/FBM_cite",
    ),
    "GBM_sd": ImportSpec(
        name="GBM_sd",
        labels_col="gbm_region_label",
        batches_col=None,
        source_subdir="datasets_public/GBM_sd",
    ),
    "mus_tissue": ImportSpec(
        name="mus_tissue",
        labels_col="tissue_label",
        batches_col=None,
        source_subdir="datasets_public/mus_tissue",
    ),
    "homo_tissue": ImportSpec(
        name="homo_tissue",
        labels_col="tissue_label",
        batches_col=None,
        source_subdir="datasets_public/homo_tissue",
    ),
}

DEFAULT_DATASET_NAMES = (
    "paul15",
    "E-MTAB-4388",
    "E-MTAB-4888",
    "E-MTAB-5061",
    "FBM_cite",
    "GBM_sd",
    "mus_tissue",
    "homo_tissue",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Import curated real scRNA-seq datasets from the hvg workspace.")
    parser.add_argument("--source-root", type=str, default=r"D:\code_py\hvg")
    parser.add_argument("--output-root", type=str, default="data/real_inputs")
    parser.add_argument("--datasets", type=str, default=",".join(DEFAULT_DATASET_NAMES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_root = Path(args.source_root).resolve()
    output_root = (repo_root / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    selected_names = [part.strip() for part in args.datasets.split(",") if part.strip()]
    unknown = [name for name in selected_names if name not in CURATED_DATASETS]
    if unknown:
        raise KeyError(f"Unknown dataset(s): {unknown}. Available: {sorted(CURATED_DATASETS)}")

    rows: list[dict[str, object]] = []
    for name in selected_names:
        spec = CURATED_DATASETS[name]
        row = import_dataset(
            spec=spec,
            source_root=source_root,
            output_root=output_root,
            repo_root=repo_root,
        )
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False))

    write_registry(rows=rows, output_root=output_root)
    print(f"Imported {len(rows)} dataset(s) into {output_root}")


def import_dataset(
    *,
    spec: ImportSpec,
    source_root: Path,
    output_root: Path,
    repo_root: Path,
) -> dict[str, object]:
    source_dir = (source_root / spec.source_subdir).resolve()
    if not source_dir.exists():
        raise FileNotFoundError(f"Missing source dataset directory: {source_dir}")

    destination_dir = output_root / spec.name
    destination_dir.mkdir(parents=True, exist_ok=True)

    copied_bytes = 0
    for relative_file in spec.files:
        source_file = source_dir / relative_file
        if not source_file.exists():
            raise FileNotFoundError(f"Missing required file for {spec.name}: {source_file}")
        destination_file = destination_dir / relative_file
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        copied_bytes += copy_if_needed(source_file, destination_file)

    input_path = destination_dir / "matrix" / "matrix.mtx"
    obs_path = destination_dir / "obs.csv"
    genes_path = destination_dir / "matrix" / "genes.tsv"
    cells_path = destination_dir / "matrix" / "barcodes.tsv"
    return {
        "dataset_name": spec.name,
        "file_format": spec.file_format,
        "transpose": spec.transpose,
        "labels_col": spec.labels_col,
        "batches_col": spec.batches_col,
        "input_path": str(input_path.relative_to(repo_root)),
        "obs_path": str(obs_path.relative_to(repo_root)),
        "genes_path": str(genes_path.relative_to(repo_root)),
        "cells_path": str(cells_path.relative_to(repo_root)),
        "source_dir": str(source_dir),
        "destination_dir": str(destination_dir.relative_to(repo_root)),
        "copied_gb": round(copied_bytes / (1024**3), 4),
        "matrix_gb": round(input_path.stat().st_size / (1024**3), 4),
    }


def copy_if_needed(source_file: Path, destination_file: Path) -> int:
    if destination_file.exists():
        same_size = destination_file.stat().st_size == source_file.stat().st_size
        same_mtime = int(destination_file.stat().st_mtime) == int(source_file.stat().st_mtime)
        if same_size and same_mtime:
            return 0
    shutil.copy2(source_file, destination_file)
    return source_file.stat().st_size


def write_registry(*, rows: list[dict[str, object]], output_root: Path) -> None:
    json_path = output_root / "registry.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    csv_path = output_root / "registry.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
