from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from hvg_research import discover_scrna_input_specs


ATLAS_H5AD_BASE_URL = "https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/sc_experiments"


@dataclass(frozen=True)
class ExternalDatasetSpec:
    accession: str
    slug: str
    title: str
    organism: str
    cell_count: int
    expected_labels: str
    expected_batches: str
    rationale: str
    shortlist_rank: int | None = None

    @property
    def public_page_url(self) -> str:
        return f"https://www.ebi.ac.uk/gxa/sc/experiments/{self.accession}/results"

    @property
    def downloads_page_url(self) -> str:
        return f"https://www.ebi.ac.uk/gxa/sc/experiments/{self.accession}/downloads"

    @property
    def download_url(self) -> str:
        return f"{ATLAS_H5AD_BASE_URL}/{self.accession}/{self.accession}.project.h5ad"


LONGLIST_SPECS: tuple[ExternalDatasetSpec, ...] = (
    ExternalDatasetSpec(
        accession="E-GEOD-81682",
        slug="atlas_mouse_hspc_diff",
        title="Single-cell RNA-Seq of mouse hematopoietic stem and progenitor cells",
        organism="mouse",
        cell_count=1919,
        expected_labels="lineage or progenitor-state annotations",
        expected_batches="replicate or plate annotations if present",
        rationale="Compact, classic differentiation trajectory with a strong route-family fit and minimal import risk.",
        shortlist_rank=1,
    ),
    ExternalDatasetSpec(
        accession="E-MTAB-10243",
        slug="atlas_mouse_primitive_streak",
        title="Gene expression after primitive streak and mesendoderm induction by different doses of differentiation inducers and pathways",
        organism="mouse",
        cell_count=51545,
        expected_labels="timepoint, dose, lineage, or differentiation-state annotations",
        expected_batches="induction condition or replicate annotations if present",
        rationale="Direct primitive streak and mesendoderm differentiation setting with clear stage structure and rich headroom for locality-style methods.",
        shortlist_rank=2,
    ),
    ExternalDatasetSpec(
        accession="E-MTAB-7324",
        slug="atlas_mouse_gastrulation",
        title="Single-cell transcriptome atlas of mouse gastrulation and early organogenesis",
        organism="mouse",
        cell_count=15927,
        expected_labels="cell type, embryo stage, or developmental branch annotations",
        expected_batches="embryo or replicate annotations if present",
        rationale="Strong developmental progression dataset, but broader atlas-like structure makes it a reserve import behind tighter shortlist candidates.",
    ),
    ExternalDatasetSpec(
        accession="E-MTAB-8205",
        slug="atlas_human_hpsc_eht",
        title="Single-cell RNA sequencing of human pluripotent stem cell differentiation towards endothelial and haematopoietic cells",
        organism="human",
        cell_count=18829,
        expected_labels="day, lineage, endothelial-to-haematopoietic transition state",
        expected_batches="replicate or condition annotations if present",
        rationale="Human endothelial-to-haematopoietic transition is a clean external trajectory-like setting and a strong shortlist candidate.",
        shortlist_rank=3,
    ),
    ExternalDatasetSpec(
        accession="E-MTAB-10945",
        slug="atlas_mouse_eht_wt_ncx1",
        title="Single cell RNA sequencing of endothelial cells and intra-aortic clusters from E9.5 wild type and Ncx1-/- embryos",
        organism="mouse",
        cell_count=17745,
        expected_labels="endothelial or intra-aortic cluster state annotations",
        expected_batches="genotype or embryo annotations",
        rationale="Biologically aligned endothelial-to-haematopoietic transition dataset, but genotype effects make it slightly less clean than the human HPSC transition shortlist option.",
    ),
    ExternalDatasetSpec(
        accession="E-MTAB-4079",
        slug="atlas_mouse_mesoderm_diversification",
        title="Single-cell RNA-seq of mesoderm diversification through embryonic development",
        organism="mouse",
        cell_count=1205,
        expected_labels="branch or lineage annotations",
        expected_batches="embryo or timepoint annotations if present",
        rationale="Clear developmental branching signal, but cell count is relatively small for a first external import audit.",
    ),
    ExternalDatasetSpec(
        accession="E-MTAB-7094",
        slug="atlas_mouse_frc_trajectory",
        title="Single cell RNA-sequencing reveals fibroblastic reticular cell differentiation trajectories",
        organism="mouse",
        cell_count=2475,
        expected_labels="fibroblastic reticular differentiation state annotations",
        expected_batches="replicate annotations if present",
        rationale="Explicit differentiation trajectory with manageable size, though narrower biology than the main shortlist.",
    ),
    ExternalDatasetSpec(
        accession="E-CURD-112",
        slug="atlas_human_fetal_bone_marrow_hematopoiesis",
        title="Single-cell atlas of blood and immune development in human fetal bone marrow and Down syndrome",
        organism="human",
        cell_count=34015,
        expected_labels="hematopoietic lineage or developmental-state annotations",
        expected_batches="sample, donor, or condition annotations",
        rationale="Strong hematopoietic development signal, but disease context introduces extra confounding for a first audit pass.",
    ),
    ExternalDatasetSpec(
        accession="E-GEOD-93593",
        slug="atlas_human_interneuron_differentiation",
        title="Single cell transcriptomics identifies differentiation of a human PSC-derived population functionally resembling fetal ventral midbrain",
        organism="human",
        cell_count=48450,
        expected_labels="differentiation state or neuronal subtype annotations",
        expected_batches="line, day, or replicate annotations",
        rationale="Useful PSC differentiation candidate, but less directly aligned with the current route than hematopoietic or mesendoderm transitions.",
    ),
    ExternalDatasetSpec(
        accession="E-GEOD-109979",
        slug="atlas_human_endoderm_differentiation",
        title="Long-term differentiating human embryonic stem cells into endoderm",
        organism="human",
        cell_count=329,
        expected_labels="day or endoderm-stage annotations",
        expected_batches="replicate annotations if present",
        rationale="Trajectory signal is likely clear, but the dataset is too small to prioritize over the main shortlist.",
    ),
)

SPEC_BY_SLUG: dict[str, ExternalDatasetSpec] = {spec.slug: spec for spec in LONGLIST_SPECS}
SPEC_BY_ACCESSION: dict[str, ExternalDatasetSpec] = {spec.accession: spec for spec in LONGLIST_SPECS}
SHORTLIST_SLUGS: tuple[str, ...] = tuple(spec.slug for spec in LONGLIST_SPECS if spec.shortlist_rank is not None)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download shortlisted external trajectory-like datasets from Single Cell Expression Atlas.")
    parser.add_argument("--output-root", type=str, default="data/external_inputs")
    parser.add_argument("--datasets", type=str, default=",".join(SHORTLIST_SLUGS))
    parser.add_argument("--timeout-sec", type=int, default=120)
    parser.add_argument("--chunk-size-mb", type=int, default=8)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def resolve_requested_specs(spec_text: str) -> tuple[ExternalDatasetSpec, ...]:
    requested = [part.strip() for part in spec_text.split(",") if part.strip()]
    if not requested or requested == ["shortlist"]:
        return tuple(SPEC_BY_SLUG[name] for name in SHORTLIST_SLUGS)
    if requested == ["all"]:
        return LONGLIST_SPECS

    resolved: list[ExternalDatasetSpec] = []
    for item in requested:
        if item in SPEC_BY_SLUG:
            resolved.append(SPEC_BY_SLUG[item])
        elif item in SPEC_BY_ACCESSION:
            resolved.append(SPEC_BY_ACCESSION[item])
        else:
            raise KeyError(f"Unknown dataset selector: {item}. Available slugs: {sorted(SPEC_BY_SLUG)}")
    return tuple(resolved)


def main() -> None:
    args = parse_args()
    output_root = (ROOT / args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selected_specs = resolve_requested_specs(args.datasets)
    write_candidate_manifest(output_root=output_root)

    if args.list_only:
        print(json.dumps([serialize_spec(spec) for spec in selected_specs], indent=2, ensure_ascii=False))
        return

    session = requests.Session()
    failures: list[dict[str, str]] = []
    completed: list[ExternalDatasetSpec] = []
    for spec in selected_specs:
        dataset_dir = output_root / spec.slug
        dataset_dir.mkdir(parents=True, exist_ok=True)
        target_path = dataset_dir / "source.h5ad"
        metadata_path = dataset_dir / "source_metadata.json"
        status = "skipped_existing"
        error_message = ""
        if args.force or not target_path.exists():
            try:
                download_file(
                    session=session,
                    url=spec.download_url,
                    target_path=target_path,
                    timeout_sec=args.timeout_sec,
                    chunk_size_mb=args.chunk_size_mb,
                )
                status = "downloaded"
            except requests.RequestException as exc:
                status = "failed"
                error_message = str(exc)
                failures.append({"dataset": spec.slug, "error": error_message})
        metadata = {
            **serialize_spec(spec),
            "download_status": status,
            "downloaded_at_utc": now_utc(),
            "target_path": str(target_path),
            "source_repository": "Single Cell Expression Atlas",
            "download_url": spec.download_url,
            "error": error_message,
        }
        metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
        if status != "failed":
            completed.append(spec)
            print(f"{status}: {spec.slug} -> {target_path}")
        else:
            print(f"failed: {spec.slug} -> {error_message}")

    write_registry(output_root=output_root, selected_specs=completed)
    if failures:
        raise RuntimeError(f"One or more dataset downloads failed: {json.dumps(failures, ensure_ascii=False)}")


def serialize_spec(spec: ExternalDatasetSpec) -> dict[str, object]:
    return {
        "accession": spec.accession,
        "slug": spec.slug,
        "title": spec.title,
        "organism": spec.organism,
        "cell_count": spec.cell_count,
        "expected_labels": spec.expected_labels,
        "expected_batches": spec.expected_batches,
        "rationale": spec.rationale,
        "shortlist_rank": spec.shortlist_rank,
        "public_page_url": spec.public_page_url,
        "downloads_page_url": spec.downloads_page_url,
        "download_url": spec.download_url,
    }


def write_candidate_manifest(*, output_root: Path) -> None:
    manifest = {
        "generated_at_utc": now_utc(),
        "shortlist_slugs": list(SHORTLIST_SLUGS),
        "datasets": [serialize_spec(spec) for spec in LONGLIST_SPECS],
    }
    (output_root / "candidate_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def download_file(
    *,
    session: requests.Session,
    url: str,
    target_path: Path,
    timeout_sec: int,
    chunk_size_mb: int,
) -> None:
    response = session.get(url, stream=True, timeout=timeout_sec)
    response.raise_for_status()
    tmp_path = target_path.with_suffix(".part")
    chunk_size = max(1, int(chunk_size_mb)) * 1024 * 1024
    with tmp_path.open("wb") as handle:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            handle.write(chunk)
    tmp_path.replace(target_path)


def write_registry(*, output_root: Path, selected_specs: list[ExternalDatasetSpec]) -> None:
    if not selected_specs:
        return
    discovered = discover_scrna_input_specs(output_root)
    discovered_by_name = {spec.dataset_name: spec for spec in discovered}
    rows: list[dict[str, object]] = []
    for source_spec in selected_specs:
        discovered_spec = discovered_by_name.get(source_spec.slug)
        if discovered_spec is None:
            continue
        input_path = Path(discovered_spec.input_path)
        rows.append(
            {
                "dataset_name": discovered_spec.dataset_name,
                "dataset_id": discovered_spec.dataset_id,
                "file_format": discovered_spec.file_format,
                "transpose": discovered_spec.transpose,
                "labels_col": discovered_spec.labels_col or "",
                "batches_col": discovered_spec.batches_col or "",
                "input_path": str(input_path.relative_to(ROOT)),
                "obs_path": discovered_spec.obs_path or "",
                "genes_path": discovered_spec.genes_path or "",
                "cells_path": discovered_spec.cells_path or "",
                "source_accession": source_spec.accession,
                "source_title": source_spec.title,
                "source_repository": "Single Cell Expression Atlas",
                "public_page_url": source_spec.public_page_url,
                "downloads_page_url": source_spec.downloads_page_url,
                "download_url": source_spec.download_url,
                "destination_dir": str(input_path.parent.relative_to(ROOT)),
                "matrix_gb": round(input_path.stat().st_size / (1024**3), 4),
            }
        )
    if not rows:
        return
    json_path = output_root / "registry.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = output_root / "registry.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


if __name__ == "__main__":
    main()
