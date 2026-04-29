from __future__ import annotations

import csv
import io
import json
import shutil
import tarfile
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import sleep


ROOT = Path(__file__).resolve().parents[2]
DATA_CATALOG_DIR = ROOT / "data_catalog"
SPATIAL_RAW_DIR = ROOT / "data" / "spatial_raw"
SPATIAL_PROCESSED_DIR = ROOT / "data" / "spatial_processed"
EXPERIMENT_DIR = ROOT / "experiments" / "03_spatial_dataset_screening"


CATALOG_FIELDS: tuple[str, ...] = (
    "dataset_name",
    "source",
    "organism",
    "tissue",
    "technology",
    "n_samples",
    "n_spots_or_cells",
    "condition_labels",
    "has_cell_type",
    "has_histology",
    "license",
    "download_url",
    "estimated_size",
    "pros",
    "cons",
    "selected",
)


@dataclass(frozen=True)
class DownloadArtifact:
    url: str
    filename: str
    kind: str
    extract: bool = True


@dataclass(frozen=True)
class CandidateDatasetSpec:
    dataset_id: str
    dataset_name: str
    source: str
    source_url: str
    organism: str
    tissue: str
    technology: str
    n_samples: int
    n_spots_or_cells: int
    condition_labels: str
    has_cell_type: str
    has_histology: str
    license: str
    estimated_size: str
    pros: str
    cons: str
    selected: str
    selection_role: str
    artifacts: tuple[DownloadArtifact, ...] = ()

    def to_manifest_row(self) -> dict[str, str]:
        return {
            "dataset_name": self.dataset_name,
            "source": self.source,
            "organism": self.organism,
            "tissue": self.tissue,
            "technology": self.technology,
            "n_samples": str(self.n_samples),
            "n_spots_or_cells": str(self.n_spots_or_cells),
            "condition_labels": self.condition_labels,
            "has_cell_type": self.has_cell_type,
            "has_histology": self.has_histology,
            "license": self.license,
            "download_url": " | ".join(artifact.url for artifact in self.artifacts) if self.artifacts else self.source_url,
            "estimated_size": self.estimated_size,
            "pros": self.pros,
            "cons": self.cons,
            "selected": self.selected,
        }


def candidate_dataset_specs() -> tuple[CandidateDatasetSpec, ...]:
    return (
        CandidateDatasetSpec(
            dataset_id="wu2021_breast_visium",
            dataset_name="Wu 2021 Breast Cancer Visium",
            source="Zenodo 4739739",
            source_url="https://zenodo.org/records/4739739",
            organism="Homo sapiens",
            tissue="Breast tumor",
            technology="10x Visium",
            n_samples=6,
            n_spots_or_cells=24584,
            condition_labels="Clinical subtype: HER2+, TNBC, ER+/PR+",
            has_cell_type="yes",
            has_histology="yes",
            license="CC BY 4.0",
            estimated_size="~0.90 GB",
            pros="Tumor-stroma-immune story is strong; paired metadata and pathology labels help interpretation; six sections are manageable.",
            cons="No normal control arm; subtype heterogeneity is useful but adds confounding.",
            selected="yes",
            selection_role="primary",
            artifacts=(
                DownloadArtifact(
                    url="https://zenodo.org/records/4739739/files/filtered_count_matrices.tar.gz?download=1",
                    filename="filtered_count_matrices.tar.gz",
                    kind="counts",
                ),
                DownloadArtifact(
                    url="https://zenodo.org/records/4739739/files/spatial.tar.gz?download=1",
                    filename="spatial.tar.gz",
                    kind="spatial",
                ),
                DownloadArtifact(
                    url="https://zenodo.org/records/4739739/files/metadata.tar.gz?download=1",
                    filename="metadata.tar.gz",
                    kind="metadata",
                ),
            ),
        ),
        CandidateDatasetSpec(
            dataset_id="gse220442_ad_mtg",
            dataset_name="GSE220442 AD MTG",
            source="GEO GSE220442",
            source_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE220442",
            organism="Homo sapiens",
            tissue="Middle temporal gyrus",
            technology="10x Visium",
            n_samples=6,
            n_spots_or_cells=22499,
            condition_labels="Alzheimer disease vs control",
            has_cell_type="yes",
            has_histology="yes",
            license="Not explicitly stated on GEO",
            estimated_size="~0.05 GB for counts+images package",
            pros="Direct disease/control contrast; six Visium sections; compact counts+images archive is easy to reproduce.",
            cons="Cell type labels may need deconvolution or clustering if only spot-level metadata are present.",
            selected="yes",
            selection_role="validation",
            artifacts=(
                DownloadArtifact(
                    url="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE220442&file=GSE220442_counts_and_images.tar.gz&format=file",
                    filename="GSE220442_counts_and_images.tar.gz",
                    kind="counts_and_images",
                ),
                DownloadArtifact(
                    url="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE220442&file=GSE220442_metadata.csv.gz&format=file",
                    filename="GSE220442_metadata.csv.gz",
                    kind="metadata_table",
                    extract=False,
                ),
            ),
        ),
        CandidateDatasetSpec(
            dataset_id="gse212903_aging_mouse_brain",
            dataset_name="GSE212903 Aging Mouse Brain",
            source="GEO GSE212903",
            source_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE212903",
            organism="Mus musculus",
            tissue="Brain",
            technology="10x Visium",
            n_samples=6,
            n_spots_or_cells=17368,
            condition_labels="2 months, 18 months, 24 months",
            has_cell_type="yes",
            has_histology="yes",
            license="Not explicitly stated on GEO",
            estimated_size="~1.42 GB",
            pros="Clean spatial layers and condition labels; useful reserve validation for region-aware motifs.",
            cons="Older/aging contrast is weaker than disease-control; raw archive is larger than the two primary picks.",
            selected="no",
            selection_role="reserve",
            artifacts=(
                DownloadArtifact(
                    url="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE212903&format=file",
                    filename="GSE212903_RAW.tar",
                    kind="raw_bundle",
                ),
                DownloadArtifact(
                    url="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE212903&file=GSE212903_spatial.tar.gz&format=file",
                    filename="GSE212903_spatial.tar.gz",
                    kind="spatial",
                ),
            ),
        ),
        CandidateDatasetSpec(
            dataset_id="gse189184_colitis",
            dataset_name="GSE189184 Colitis Panel",
            source="GEO GSE189184",
            source_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE189184",
            organism="Homo sapiens",
            tissue="Colon",
            technology="10x Visium",
            n_samples=16,
            n_spots_or_cells=50857,
            condition_labels="Ulcerative colitis, healthy, ICI colitis",
            has_cell_type="yes",
            has_histology="yes",
            license="Not explicitly stated on GEO",
            estimated_size="~8.73 GB",
            pros="Excellent inflammation story with healthy and disease states; immune and stromal neighborhoods are directly relevant.",
            cons="Archive is large for a first-week screen; preprocessing cost is meaningfully higher than the selected pair.",
            selected="no",
            selection_role="reserve",
            artifacts=(
                DownloadArtifact(
                    url="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE189184&format=file",
                    filename="GSE189184_RAW.tar",
                    kind="raw_bundle",
                ),
            ),
        ),
        CandidateDatasetSpec(
            dataset_id="gse152506_ad_mouse_plaque",
            dataset_name="GSE152506 AD Mouse Plaque Niches",
            source="GEO GSE152506",
            source_url="https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE152506",
            organism="Mus musculus",
            tissue="Brain",
            technology="Spatial Transcriptomics (legacy ST)",
            n_samples=20,
            n_spots_or_cells=10079,
            condition_labels="AppNL-G-F KI vs WT; 3, 6, 12, 18 months",
            has_cell_type="partial",
            has_histology="partial",
            license="Not explicitly stated on GEO",
            estimated_size="~0.26 GB",
            pros="Large number of sections and explicit genotype/time conditions make it a strong disease-model candidate.",
            cons="Legacy ST packaging is less turnkey than Visium; coordinate and image handling likely needs custom work.",
            selected="no",
            selection_role="reserve",
            artifacts=(
                DownloadArtifact(
                    url="https://www.ncbi.nlm.nih.gov/geo/download/?acc=GSE152506&file=GSE152506_raw_counts.txt.gz&format=file",
                    filename="GSE152506_raw_counts.txt.gz",
                    kind="counts_table",
                    extract=False,
                ),
            ),
        ),
    )


def ensure_output_dirs() -> None:
    DATA_CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    SPATIAL_RAW_DIR.mkdir(parents=True, exist_ok=True)
    SPATIAL_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "results").mkdir(parents=True, exist_ok=True)
    (EXPERIMENT_DIR / "figures").mkdir(parents=True, exist_ok=True)


def write_catalog_files() -> None:
    ensure_output_dirs()
    specs = candidate_dataset_specs()
    rows = [spec.to_manifest_row() for spec in specs]
    manifest_path = DATA_CATALOG_DIR / "download_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CATALOG_FIELDS))
        writer.writeheader()
        writer.writerows(rows)

    candidate_md = DATA_CATALOG_DIR / "candidate_datasets.md"
    candidate_md.write_text(render_candidate_markdown(specs), encoding="utf-8")

    selected_md = DATA_CATALOG_DIR / "selected_datasets.md"
    selected_md.write_text(render_selected_markdown(specs), encoding="utf-8")

    registry_json = DATA_CATALOG_DIR / "download_manifest.json"
    registry_json.write_text(
        json.dumps(
            {
                "generated_at_utc": now_utc(),
                "datasets": [spec.to_manifest_row() | {"dataset_id": spec.dataset_id, "selection_role": spec.selection_role} for spec in specs],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def render_candidate_markdown(specs: tuple[CandidateDatasetSpec, ...]) -> str:
    lines = [
        "# Candidate Spatial Datasets",
        "",
        "Screening target: prioritize multi-sample spatial cohorts with condition labels, coordinates, interpretable biology, and download paths that are reproducible inside this repository.",
        "",
        "## Shortlist Table",
        "",
        "| Dataset | Direction | Samples | Spots/Cells | Labels | Histology | Cell type | Selected |",
        "| --- | --- | ---: | ---: | --- | --- | --- | --- |",
    ]
    direction_by_role = {
        "primary": "Tumor microenvironment",
        "validation": "Brain disease/control",
        "reserve": "Reserve",
    }
    for spec in specs:
        direction = direction_by_role.get(spec.selection_role, "Reserve")
        lines.append(
            f"| {spec.dataset_name} | {direction} | {spec.n_samples} | {spec.n_spots_or_cells} | {spec.condition_labels} | {spec.has_histology} | {spec.has_cell_type} | {spec.selected} |"
        )

    lines.extend(
        [
            "",
            "## Notes By Dataset",
            "",
        ]
    )
    for spec in specs:
        lines.extend(
            [
                f"### {spec.dataset_name}",
                f"- Source: [{spec.source}]({spec.source_url})",
                f"- Tissue/technology: {spec.tissue}; {spec.technology}",
                f"- Conditions: {spec.condition_labels}",
                f"- Estimated size: {spec.estimated_size}",
                f"- Pros: {spec.pros}",
                f"- Cons: {spec.cons}",
                f"- Download artifacts: {', '.join(artifact.filename for artifact in spec.artifacts) if spec.artifacts else 'see source page'}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def render_selected_markdown(specs: tuple[CandidateDatasetSpec, ...]) -> str:
    selected_specs = [spec for spec in specs if spec.selected == "yes"]
    reserve_specs = [spec for spec in specs if spec.selection_role == "reserve" and spec.selected != "yes"]
    lines = [
        "# Selected Spatial Datasets",
        "",
        "## Active Picks",
        "",
    ]
    for spec in selected_specs:
        lines.extend(
            [
                f"### {spec.selection_role.title()}: {spec.dataset_name}",
                f"- Source: [{spec.source}]({spec.source_url})",
                f"- Why now: {spec.pros}",
                f"- Main trade-off: {spec.cons}",
                f"- Planned role: `{spec.selection_role}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Reserve Candidate",
            "",
        ]
    )
    for spec in reserve_specs[:1]:
        lines.extend(
            [
                f"### {spec.dataset_name}",
                f"- Source: [{spec.source}]({spec.source_url})",
                f"- Why reserve: {spec.pros}",
                f"- Why not active this week: {spec.cons}",
                "",
            ]
        )
    return "\n".join(lines).strip() + "\n"


def download_selected_datasets(
    *,
    force: bool = False,
    selected_only: bool = True,
    timeout_sec: int = 120,
    chunk_size_mb: int = 8,
) -> list[dict[str, str]]:
    ensure_output_dirs()
    specs = candidate_dataset_specs()
    if selected_only:
        specs = tuple(spec for spec in specs if spec.selected == "yes")
    import requests

    session = requests.Session()
    records: list[dict[str, str]] = []
    for spec in specs:
        dataset_dir = SPATIAL_RAW_DIR / spec.dataset_id
        downloads_dir = dataset_dir / "downloads"
        extracted_dir = dataset_dir / "extracted"
        downloads_dir.mkdir(parents=True, exist_ok=True)
        extracted_dir.mkdir(parents=True, exist_ok=True)
        for artifact in spec.artifacts:
            artifact_extract_dir = extracted_dir / artifact.kind
            artifact_extract_dir.mkdir(parents=True, exist_ok=True)
            target_path = downloads_dir / artifact.filename
            status = "skipped_existing"
            if force or not target_path.exists():
                download_file(
                    session=session,
                    url=artifact.url,
                    target_path=target_path,
                    timeout_sec=timeout_sec,
                    chunk_size_mb=chunk_size_mb,
                )
                status = "downloaded"
            if artifact.extract:
                marker_path = artifact_extract_dir / ".extract.done.json"
                if force or not marker_path.exists():
                    extracted_members = extract_archive(target_path=target_path, destination_dir=artifact_extract_dir)
                    marker_payload = {
                        "source_file": str(target_path),
                        "artifact_kind": artifact.kind,
                        "member_count": len(extracted_members),
                        "extracted_at_utc": now_utc(),
                    }
                    marker_path.write_text(json.dumps(marker_payload, indent=2), encoding="utf-8")
                else:
                    extracted_members = []
            else:
                extracted_members = []
            records.append(
                {
                    "dataset_id": spec.dataset_id,
                    "dataset_name": spec.dataset_name,
                    "artifact_kind": artifact.kind,
                    "status": status,
                    "download_url": artifact.url,
                    "download_path": str(target_path),
                    "extracted_dir": str(artifact_extract_dir),
                    "extracted_members": str(len(extracted_members)),
                }
            )
        metadata_path = dataset_dir / "download_record.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "dataset_id": spec.dataset_id,
                    "dataset_name": spec.dataset_name,
                    "selection_role": spec.selection_role,
                    "downloaded_at_utc": now_utc(),
                    "artifacts": records_for_dataset(records, spec.dataset_id),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
    return records


def records_for_dataset(records: list[dict[str, str]], dataset_id: str) -> list[dict[str, str]]:
    return [record for record in records if record["dataset_id"] == dataset_id]


def download_file(
    *,
    session,
    url: str,
    target_path: Path,
    timeout_sec: int,
    chunk_size_mb: int,
    max_retries: int = 4,
) -> None:
    import requests

    tmp_path = target_path.with_suffix(target_path.suffix + ".part")
    chunk_size = max(1, int(chunk_size_mb)) * 1024 * 1024
    attempt = 0
    while attempt < max_retries:
        existing_bytes = tmp_path.stat().st_size if tmp_path.exists() else 0
        headers: dict[str, str] = {}
        if existing_bytes > 0:
            headers["Range"] = f"bytes={existing_bytes}-"
        try:
            response = session.get(url, stream=True, timeout=timeout_sec, headers=headers)
            response.raise_for_status()
            mode = "ab" if existing_bytes > 0 and response.status_code == 206 else "wb"
            if mode == "wb" and tmp_path.exists():
                tmp_path.unlink()
                existing_bytes = 0
            with tmp_path.open(mode) as handle:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if not chunk:
                        continue
                    handle.write(chunk)
            tmp_path.replace(target_path)
            return
        except requests.RequestException:
            attempt += 1
            if attempt >= max_retries:
                raise
            sleep(min(5 * attempt, 20))


def extract_archive(*, target_path: Path, destination_dir: Path) -> list[str]:
    lower_name = target_path.name.lower()
    if lower_name.endswith((".tar", ".tar.gz", ".tgz")):
        return extract_tar_archive(target_path=target_path, destination_dir=destination_dir)
    if lower_name.endswith(".zip"):
        return extract_zip_archive(target_path=target_path, destination_dir=destination_dir)
    raise ValueError(f"Unsupported archive type: {target_path}")


def extract_tar_archive(*, target_path: Path, destination_dir: Path) -> list[str]:
    extracted: list[str] = []
    with tarfile.open(target_path, mode="r:*") as handle:
        for member in handle.getmembers():
            if member.isdir():
                member_path = destination_dir / member.name
                if not is_safe_relative_path(destination_dir, member_path):
                    raise ValueError(f"Unsafe tar member path: {member.name}")
                member_path.mkdir(parents=True, exist_ok=True)
                extracted.append(member.name)
                continue
            if not member.isfile():
                continue
            member_path = destination_dir / member.name
            if not is_safe_relative_path(destination_dir, member_path):
                raise ValueError(f"Unsafe tar member path: {member.name}")
            member_path.parent.mkdir(parents=True, exist_ok=True)
            with handle.extractfile(member) as source, member_path.open("wb") as target:
                if source is None:
                    continue
                shutil.copyfileobj(source, target)
            extracted.append(member.name)
    return extracted


def extract_zip_archive(*, target_path: Path, destination_dir: Path) -> list[str]:
    extracted: list[str] = []
    with zipfile.ZipFile(target_path) as handle:
        for member in handle.infolist():
            member_path = destination_dir / member.filename
            if not is_safe_relative_path(destination_dir, member_path):
                raise ValueError(f"Unsafe zip member path: {member.filename}")
            handle.extract(member, path=destination_dir)
            extracted.append(member.filename)
    return extracted


def is_safe_relative_path(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def summarize_download_records(records: list[dict[str, str]]) -> str:
    csv_buffer = io.StringIO()
    fieldnames = [
        "dataset_id",
        "dataset_name",
        "artifact_kind",
        "status",
        "download_url",
        "download_path",
        "extracted_dir",
        "extracted_members",
    ]
    writer = csv.DictWriter(csv_buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(records)
    return csv_buffer.getvalue()


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
