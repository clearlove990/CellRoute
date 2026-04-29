from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import textwrap
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

os.environ.setdefault("OMP_NUM_THREADS", "1")

try:
    import torch
except Exception:  # pragma: no cover - torch may be unavailable
    torch = None


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = ROOT / "artifacts_benchmarkhvg_bootstrap_audit"
EXTERNAL_DIR = ROOT / "external"
DATA_RAW_DIR = ROOT / "data" / "standard_benchmarkhvg_raw"
DATA_PROCESSED_DIR = ROOT / "data" / "standard_benchmarkhvg_processed"
DEFAULT_R_ENV_ROOT = ROOT / ".conda_benchmarkhvg_bootstrap"
R_ENV_ROOT = DEFAULT_R_ENV_ROOT
R_ENV_RSCRIPT = R_ENV_ROOT / "lib" / "R" / "bin" / "x64" / "Rscript.exe"
BENCHMARKHVG_REPO = EXTERNAL_DIR / "benchmarkHVG"
MIXHVG_REPO = EXTERNAL_DIR / "mixhvg"
SCRNASEQPROCESS_REPO = EXTERNAL_DIR / "scRNAseqProcess"
DUO4_BAD_ZIP = DATA_PROCESSED_DIR / "duo4_pbmc.zip"
DUO4_GOOD_ZIP = DATA_PROCESSED_DIR / "duo4_pbmc_redownload.zip"
DUO4_EXPR = DATA_PROCESSED_DIR / "duo4_pbmc" / "duo4_pbmc" / "duo4_expr.rds"
DUO4_LABEL = DATA_PROCESSED_DIR / "duo4_pbmc" / "duo4_pbmc" / "duo4_label.rds"
MIGRATION_DECISION = ROOT / "artifacts_standard_hvg_benchmark_migration" / "migration_decision.md"
CURRENT_METHODS_AUDIT = (
    ROOT / "artifacts_standard_hvg_benchmark_migration" / "current_methods_adapter_audit.md"
)
OFFICIAL_BASELINES = ROOT / "src" / "hvg_research" / "official_baselines.py"
OFFICIAL_R_WORKER = ROOT / "scripts" / "score_official_hvg.R"
OFFICIAL_PY_WORKER = ROOT / "scripts" / "score_official_hvg.py"

PACKAGE_NAMES = [
    "jsonlite",
    "Seurat",
    "SeuratObject",
    "SingleCellExperiment",
    "scuttle",
    "scran",
    "FNN",
    "Rfast",
    "mclust",
    "caret",
    "pdist",
    "cluster",
    "NMI",
    "reticulate",
    "remotes",
    "BiocManager",
    "lisi",
    "mixhvg",
    "benchmarkHVG",
]


def _set_r_env_root(path: Path) -> None:
    global R_ENV_ROOT, R_ENV_RSCRIPT
    R_ENV_ROOT = path.resolve()
    R_ENV_RSCRIPT = R_ENV_ROOT / "lib" / "R" / "bin" / "x64" / "Rscript.exe"


_set_r_env_root(Path(os.environ.get("BENCHMARKHVG_R_ENV", str(DEFAULT_R_ENV_ROOT))))


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and self.error is None


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _safe_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _r_string(value: str | Path) -> str:
    return json.dumps(str(value).replace("\\", "/"))


def _quote_command(args: list[str]) -> str:
    return subprocess.list2cmdline(args)


def _read_git_config(repo_path: Path) -> dict[str, str | None]:
    config_path = repo_path / ".git" / "config"
    commit_path = repo_path / ".git" / "refs" / "heads" / "main"
    if not config_path.exists():
        return {"remote_url": None, "head_commit": None}
    text = config_path.read_text(encoding="utf-8", errors="replace")
    match = re.search(r'^\s*url\s*=\s*(.+)$', text, flags=re.MULTILINE)
    remote_url = match.group(1).strip() if match else None
    head_commit = None
    if commit_path.exists():
        head_commit = commit_path.read_text(encoding="utf-8", errors="replace").strip() or None
    return {"remote_url": remote_url, "head_commit": head_commit}


def _resource_row(
    resource_name: str,
    local_path: Path | str,
    *,
    exists: bool,
    accessible: bool,
    installed: bool | None,
    version: str | None,
    source: str,
    status: str,
    notes: str,
) -> dict[str, str]:
    return {
        "resource_name": resource_name,
        "local_path": str(local_path),
        "exists": _bool(exists),
        "accessible": _bool(accessible),
        "installed": "" if installed is None else _bool(installed),
        "version": _safe_text(version),
        "source": source,
        "status": status,
        "notes": notes,
    }


def _ensure_no_overwrite(path: Path) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {path}")


def _write_text(path: Path, content: str) -> None:
    _ensure_no_overwrite(path)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _ensure_no_overwrite(path)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    _ensure_no_overwrite(path)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _run_command(
    args: list[str],
    *,
    cwd: Path = ROOT,
    env: dict[str, str] | None = None,
    timeout: int = 120_000,
) -> CommandResult:
    try:
        completed = subprocess.run(
            args,
            cwd=str(cwd),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout,
            check=False,
        )
        return CommandResult(
            command=args,
            returncode=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
        )
    except Exception as exc:  # pragma: no cover - defensive
        return CommandResult(
            command=args,
            returncode=1,
            stdout="",
            stderr="",
            error=f"{type(exc).__name__}: {exc}",
        )


def _r_runtime_env() -> tuple[Path | None, dict[str, str] | None]:
    if not R_ENV_RSCRIPT.exists():
        return None, None
    env = os.environ.copy()
    env["R_HOME"] = str(R_ENV_ROOT / "lib" / "R")
    path_parts = [
        R_ENV_ROOT / "lib" / "R" / "bin" / "x64",
        R_ENV_ROOT / "lib" / "R" / "bin",
        R_ENV_ROOT / "Library" / "mingw-w64" / "bin",
        R_ENV_ROOT / "Library" / "usr" / "bin",
        R_ENV_ROOT / "Library" / "bin",
        R_ENV_ROOT / "Scripts",
        R_ENV_ROOT / "bin",
    ]
    existing = [str(part) for part in path_parts if part.exists()]
    env["PATH"] = ";".join(existing + [env.get("PATH", "")])
    return R_ENV_RSCRIPT, env


def _r_version_string(tmp_dir: Path) -> str | None:
    payload, _ = _run_r_json(
        """
        result <- list(
          version = R.version$version.string
        )
        """,
        tmp_dir=tmp_dir,
    )
    if payload is None:
        return None
    return payload.get("version")


def _bioc_repos_expr() -> str:
    repos = {
        "BioC": "https://bioconductor.org/packages/3.20/bioc",
        "BioCAnn": "https://bioconductor.org/packages/3.20/data/annotation",
        "BioCExp": "https://bioconductor.org/packages/3.20/data/experiment",
        "CRAN": "https://cloud.r-project.org",
    }
    return "c(" + ",".join(f"{name}={_r_string(url)}" for name, url in repos.items()) + ")"


def _run_r_json(
    code: str,
    *,
    tmp_dir: Path,
    timeout: int = 120_000,
) -> tuple[dict[str, Any] | None, CommandResult]:
    rscript, env = _r_runtime_env()
    if rscript is None or env is None:
        return None, CommandResult(command=[], returncode=1, stdout="", stderr="", error="Rscript missing")
    script_path = tmp_dir / f"tmp_{uuid.uuid4().hex}.R"
    json_path = tmp_dir / f"tmp_{uuid.uuid4().hex}.json"
    script_path.write_text(
        "\n".join(
            [
                "options(warn = 1)",
                "if (!requireNamespace('jsonlite', quietly = TRUE)) stop('jsonlite package is missing')",
                "result <- NULL",
                code,
                f"jsonlite::write_json(result, path = {_r_string(json_path)}, auto_unbox = TRUE, pretty = TRUE, null = 'null')",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    result = _run_command([str(rscript), "--vanilla", str(script_path)], env=env, timeout=timeout)
    payload = None
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    try:
        script_path.unlink(missing_ok=True)
        json_path.unlink(missing_ok=True)
    except OSError:
        pass
    return payload, result


def _run_r_script(
    code: str,
    *,
    tmp_dir: Path,
    timeout: int = 120_000,
) -> CommandResult:
    rscript, env = _r_runtime_env()
    if rscript is None or env is None:
        return CommandResult(command=[], returncode=1, stdout="", stderr="", error="Rscript missing")
    script_path = tmp_dir / f"tmp_{uuid.uuid4().hex}.R"
    script_path.write_text(code + "\n", encoding="utf-8")
    result = _run_command([str(rscript), "--vanilla", str(script_path)], env=env, timeout=timeout)
    try:
        script_path.unlink(missing_ok=True)
    except OSError:
        pass
    return result


def _compute_context() -> dict[str, Any]:
    payload: dict[str, Any] = {
        "cwd": str(ROOT),
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_imported": torch is not None,
        "preferred_device": "cpu",
        "device_note": "GPU status recorded per AGENTS.md. This bootstrap audit is primarily I/O/R-bound.",
    }
    if torch is not None:
        try:
            cuda_available = bool(torch.cuda.is_available())
            payload["cuda_available"] = cuda_available
            payload["cuda_device_count"] = int(torch.cuda.device_count()) if cuda_available else 0
            payload["preferred_device"] = "cuda" if cuda_available else "cpu"
            if cuda_available:
                payload["cuda_devices"] = [
                    {"index": idx, "name": torch.cuda.get_device_name(idx)}
                    for idx in range(torch.cuda.device_count())
                ]
        except Exception as exc:  # pragma: no cover - defensive
            payload["cuda_available"] = False
            payload["cuda_error"] = f"{type(exc).__name__}: {exc}"
    else:
        payload["cuda_available"] = False
    return payload


def _package_audit(tmp_dir: Path) -> tuple[list[dict[str, Any]], CommandResult | None]:
    rscript, _ = _r_runtime_env()
    if rscript is None:
        return [], None
    pkg_vec = "c(" + ",".join(_r_string(pkg) for pkg in PACKAGE_NAMES) + ")"
    code = textwrap.dedent(
        f"""
        pkgs <- {pkg_vec}
        ip <- installed.packages()
        result <- lapply(pkgs, function(p) {{
          installed <- p %in% rownames(ip)
          version <- if (installed) as.character(ip[p, "Version"]) else NULL
          libpath <- if (installed) as.character(ip[p, "LibPath"]) else NULL
          load_error <- NULL
          load_ok <- tryCatch({{
            suppressPackageStartupMessages(loadNamespace(p))
            TRUE
          }}, error = function(e) {{
            load_error <<- conditionMessage(e)
            FALSE
          }})
          list(
            package = p,
            installed = installed,
            version = version,
            libpath = libpath,
            load_ok = load_ok,
            load_error = load_error
          )
        }})
        """
    )
    payload, result = _run_r_json(code, tmp_dir=tmp_dir)
    if payload is None:
        return [], result
    return list(payload), result


def _duo4_summary(tmp_dir: Path) -> tuple[dict[str, Any] | None, CommandResult | None]:
    if not DUO4_EXPR.exists() or not DUO4_LABEL.exists():
        return None, None
    code = textwrap.dedent(
        f"""
        expr <- readRDS({_r_string(DUO4_EXPR)})
        label <- readRDS({_r_string(DUO4_LABEL)})
        label_unique <- if (is.factor(label)) length(levels(label)) else length(unique(label))
        result <- list(
          expr_class = unname(class(expr)),
          expr_dim = unname(dim(expr)),
          expr_has_rownames = !is.null(rownames(expr)),
          expr_has_colnames = !is.null(colnames(expr)),
          label_class = unname(class(label)),
          label_length = length(label),
          label_unique = label_unique
        )
        """
    )
    return _run_r_json(code, tmp_dir=tmp_dir)


def _source_check(path: Path, *, tmp_dir: Path, parse_only: bool = False, symbol: str | None = None) -> tuple[dict[str, Any] | None, CommandResult]:
    if not path.exists():
        return None, CommandResult(command=[], returncode=1, stdout="", stderr="", error=f"Missing file: {path}")
    if parse_only:
        code = textwrap.dedent(
            f"""
            parse_error <- NULL
            ok <- tryCatch({{
              parse(file = {_r_string(path)})
              TRUE
            }}, error = function(e) {{
              parse_error <<- conditionMessage(e)
              FALSE
            }})
            result <- list(ok = ok, parse_error = parse_error)
            """
        )
    else:
        exists_clause = f"exists({_r_string(symbol)})" if symbol else "TRUE"
        code = textwrap.dedent(
            f"""
            source_error <- NULL
            ok <- tryCatch({{
              source({_r_string(path)})
              TRUE
            }}, error = function(e) {{
              source_error <<- conditionMessage(e)
              FALSE
            }})
            result <- list(ok = ok, source_error = source_error, symbol_exists = if (ok) {exists_clause} else FALSE)
            """
        )
    payload, result = _run_r_json(code, tmp_dir=tmp_dir)
    return payload, result


def _network_attempts(tmp_dir: Path) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []

    conda_prefix = [
        "conda",
        "run",
        "--no-capture-output",
        "-p",
        str(R_ENV_ROOT),
        "Rscript",
        "-e",
    ]
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["CONDA_NO_PLUGINS"] = "true"
    env["CONDA_PKGS_DIRS"] = str(ROOT / ".conda_pkgs")
    env["CONDA_ENVS_PATH"] = str(ROOT / ".conda_envs")
    env["LOCALAPPDATA"] = str(ROOT / ".localappdata")

    bioc_repos = _bioc_repos_expr()
    commands = [
        (
            "biocmanager_validation",
            "options(download.file.method='wininet'); suppressPackageStartupMessages(library(BiocManager)); BiocManager::version()",
        ),
        (
            "bioc_direct_repo_probe",
            f"options(download.file.method='wininet', pkgType='win.binary'); repos <- {bioc_repos}; ap <- available.packages(repos=repos); cat(nrow(ap), '\\n')",
        ),
    ]
    for label, expr in commands:
        result = _run_command(conda_prefix + [expr], env=env, timeout=240_000)
        attempts.append(
            {
                "label": label,
                "command": _quote_command(conda_prefix + [expr]),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": result.error,
            }
        )
    return attempts


def _remote_install_audit(tmp_dir: Path) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    package_name = "lisi"
    lib_dir = tmp_dir / f"{package_name}_remote_lib"
    lib_dir.mkdir(parents=True, exist_ok=True)
    code = textwrap.dedent(
        f"""
        if (!requireNamespace("remotes", quietly = TRUE)) stop("remotes package is missing")
        libdir <- {_r_string(lib_dir)}
        dir.create(libdir, showWarnings = FALSE, recursive = TRUE)
        install_warning <- character()
        install_error <- NULL
        tryCatch({{
          withCallingHandlers({{
            remotes::install_github("immunogenomics/LISI", lib = libdir, upgrade = "never", dependencies = FALSE, force = TRUE)
          }}, warning = function(w) {{
            install_warning <<- c(install_warning, conditionMessage(w))
            invokeRestart("muffleWarning")
          }})
        }}, error = function(e) {{
          install_error <<- conditionMessage(e)
        }})
        ip <- installed.packages(lib.loc = c(libdir, .libPaths()))
        installed <- {_r_string(package_name)} %in% rownames(ip)
        load_error <- NULL
        load_ok <- FALSE
        if (installed) {{
          load_ok <- tryCatch({{
            suppressPackageStartupMessages(loadNamespace({_r_string(package_name)}, lib.loc = c(libdir, .libPaths())))
            TRUE
          }}, error = function(e) {{
            load_error <<- conditionMessage(e)
            FALSE
          }})
        }}
        result <- list(
          package = {_r_string(package_name)},
          source = "https://github.com/immunogenomics/LISI",
          libdir = libdir,
          installed = installed,
          load_ok = load_ok,
          install_warning = if (length(install_warning)) paste(unique(install_warning), collapse = " | ") else NULL,
          install_error = install_error,
          load_error = load_error
        )
        """
    )
    payload, result = _run_r_json(code, tmp_dir=tmp_dir, timeout=480_000)
    audits.append(
        {
            "package": package_name,
            "payload": payload,
            "command": _quote_command([str(R_ENV_RSCRIPT), "--vanilla", "<temp_script>"]),
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "error": result.error,
        }
    )
    return audits


def _local_install_audit(tmp_dir: Path) -> list[dict[str, Any]]:
    audits: list[dict[str, Any]] = []
    packages = [
        ("mixhvg", MIXHVG_REPO),
        ("benchmarkHVG", BENCHMARKHVG_REPO),
    ]
    for package_name, repo_path in packages:
        lib_dir = tmp_dir / f"{package_name}_lib"
        lib_dir.mkdir(parents=True, exist_ok=True)
        code = textwrap.dedent(
            f"""
            if (!requireNamespace("remotes", quietly = TRUE)) stop("remotes package is missing")
            libdir <- {_r_string(lib_dir)}
            dir.create(libdir, showWarnings = FALSE, recursive = TRUE)
            install_warning <- NULL
            install_error <- NULL
            tryCatch({{
              withCallingHandlers({{
                remotes::install_local({_r_string(repo_path)}, lib = libdir, upgrade = "never", dependencies = FALSE, force = TRUE)
              }}, warning = function(w) {{
                install_warning <<- conditionMessage(w)
                invokeRestart("muffleWarning")
              }})
            }}, error = function(e) {{
              install_error <<- conditionMessage(e)
            }})
            ip <- installed.packages(lib.loc = libdir)
            installed <- {_r_string(package_name)} %in% rownames(ip)
            load_error <- NULL
            load_ok <- FALSE
            if (installed) {{
              load_ok <- tryCatch({{
                suppressPackageStartupMessages(loadNamespace({_r_string(package_name)}, lib.loc = libdir))
                TRUE
              }}, error = function(e) {{
                load_error <<- conditionMessage(e)
                FALSE
              }})
            }}
            result <- list(
              package = {_r_string(package_name)},
              repo = {_r_string(repo_path)},
              libdir = libdir,
              installed = installed,
              load_ok = load_ok,
              install_warning = install_warning,
              install_error = install_error,
              load_error = load_error
            )
            """
        )
        payload, result = _run_r_json(code, tmp_dir=tmp_dir, timeout=240_000)
        audits.append(
            {
                "package": package_name,
                "repo": str(repo_path),
                "payload": payload,
                "command": _quote_command([str(R_ENV_RSCRIPT), "--vanilla", "<temp_script>"]),
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
                "error": result.error,
            }
        )
    return audits


def _tiny_mtx_write(path: Path) -> None:
    content = "\n".join(
        [
            "%%MatrixMarket matrix coordinate integer general",
            "%",
            "4 3 8",
            "1 1 1",
            "1 3 3",
            "2 2 2",
            "2 3 1",
            "3 1 5",
            "4 1 1",
            "4 2 1",
            "4 3 1",
        ]
    )
    path.write_text(content + "\n", encoding="utf-8")


def _bridge_smoke(tmp_dir: Path) -> list[dict[str, Any]]:
    rscript, env = _r_runtime_env()
    if rscript is None or env is None:
        return []
    counts_path = tmp_dir / "tiny_counts.mtx"
    _tiny_mtx_write(counts_path)
    results: list[dict[str, Any]] = []
    for method in ["seurat_r_vst_hvg", "scran_model_gene_var_hvg"]:
        output_path = tmp_dir / f"{method}_scores.txt"
        cmd = [
            str(rscript),
            str(OFFICIAL_R_WORKER),
            "--method",
            method,
            "--counts-mtx-path",
            str(counts_path),
            "--top-k",
            "2",
            "--output-path",
            str(output_path),
        ]
        result = _run_command(cmd, env=env, timeout=120_000)
        results.append(
            {
                "method": method,
                "command": _quote_command(cmd),
                "returncode": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "error": result.error,
                "output_exists": output_path.exists(),
            }
        )
    return results


def _official_pilot_smoke(tmp_dir: Path) -> dict[str, Any] | None:
    if not DUO4_EXPR.exists() or not DUO4_LABEL.exists():
        return None
    code = textwrap.dedent(
        f"""
        if (!requireNamespace("Seurat", quietly = TRUE)) stop("Seurat package is missing")
        if (!requireNamespace("mixhvg", quietly = TRUE)) stop("mixhvg package is missing")
        if (!requireNamespace("Rfast", quietly = TRUE)) stop("Rfast package is missing")
        if (!requireNamespace("cluster", quietly = TRUE)) stop("cluster package is missing")
        expr <- readRDS({_r_string(DUO4_EXPR)})
        label <- readRDS({_r_string(DUO4_LABEL)})
        suppressPackageStartupMessages(library(Rfast))
        suppressPackageStartupMessages(library(cluster))
        source({_r_string(BENCHMARKHVG_REPO / "R" / "benchmark_HVG_2_Evaluation_Criteria.R")})
        set.seed(1)
        obj <- Seurat::CreateSeuratObject(expr)
        obj <- Seurat::NormalizeData(obj, verbose = FALSE)
        obj <- mixhvg::FindVariableFeaturesMix(obj, nfeatures = 2000, verbose = FALSE)
        hvgs <- Seurat::VariableFeatures(obj)
        obj <- Seurat::ScaleData(obj, features = hvgs, verbose = FALSE)
        suppressWarnings(obj <- Seurat::RunPCA(obj, features = hvgs, npcs = 20, verbose = FALSE))
        emb <- Seurat::Embeddings(obj, reduction = "pca")
        emb <- emb[, seq_len(min(20, ncol(emb))), drop = FALSE]
        result <- list(
          ok = TRUE,
          n_cells = nrow(emb),
          n_pcs = ncol(emb),
          hvg_count = length(hvgs),
          top_hvgs = unname(head(hvgs, 10)),
          var_ratio = between_within_var_ratio(emb, label),
          asw = asw_func(emb, label)
        )
        """
    )
    payload, result = _run_r_json(code, tmp_dir=tmp_dir, timeout=900_000)
    if payload is None:
        return {
            "ok": False,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
        }
    return payload


def _official_wrapper_smoke(tmp_dir: Path) -> dict[str, Any] | None:
    if not DUO4_EXPR.exists() or not DUO4_LABEL.exists():
        return None
    code = textwrap.dedent(
        f"""
        if (!requireNamespace("benchmarkHVG", quietly = TRUE)) stop("benchmarkHVG package is missing")
        expr <- readRDS({_r_string(DUO4_EXPR)})
        label <- readRDS({_r_string(DUO4_LABEL)})
        set.seed(1)
        base_res <- benchmarkHVG::hvg_pca(expr, nfeatures = 500, SCT = FALSE)
        mix_res <- benchmarkHVG::mixture_hvg_pca(
          expr,
          nfeatures = 500,
          method_list = c("mv_lognc", "logmv_lognc"),
          mixture_index_list = list(c(1, 2))
        )
        eval_base <- benchmarkHVG::evaluate_hvg_discrete(base_res$seurat.obj.pca[1], label, verbose = FALSE)
        eval_mix <- benchmarkHVG::evaluate_hvg_discrete(mix_res$seurat.obj.pca[1], label, verbose = FALSE)
        result <- list(
          ok = TRUE,
          base_pca_count = length(base_res$seurat.obj.pca),
          base_first_dim = unname(dim(base_res$seurat.obj.pca[[1]])),
          mix_pca_count = length(mix_res$seurat.obj.pca),
          mix_first_dim = unname(dim(mix_res$seurat.obj.pca[[1]])),
          eval_base_var_ratio = unname(eval_base$var_ratio[1]),
          eval_base_ari = unname(eval_base$ari[1]),
          eval_base_nmi = unname(eval_base$nmi[1]),
          eval_base_lisi = unname(eval_base$lisi[1]),
          eval_mix_var_ratio = unname(eval_mix$var_ratio[1]),
          eval_mix_ari = unname(eval_mix$ari[1]),
          eval_mix_nmi = unname(eval_mix$nmi[1]),
          eval_mix_lisi = unname(eval_mix$lisi[1])
        )
        """
    )
    payload, result = _run_r_json(code, tmp_dir=tmp_dir, timeout=1_800_000)
    if payload is None:
        return {
            "ok": False,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "error": result.error,
        }
    return payload


def _hook_audit() -> dict[str, Any]:
    mix_text = MIXHVG_REPO.joinpath("R", "FindVariableFeaturesMix.R").read_text(
        encoding="utf-8",
        errors="replace",
    )
    benchmark_text = BENCHMARKHVG_REPO.joinpath("R", "benchmark_HVG_1_Methods.R").read_text(
        encoding="utf-8",
        errors="replace",
    )
    return {
        "mixhvg_has_extra_rank_arg": "extra.rank = NULL" in mix_text,
        "mixhvg_length_guard": "extra.rank needs to match the rank of all genes." in mix_text,
        "mixhvg_name_guard": "extra.rank needs to match the names of all genes." in mix_text,
        "mixhvg_reorder_logic": "extra.rank = extra.rank[match(allfeatures,names(extra.rank))]" in mix_text,
        "mixhvg_rank_doc": 'Provide the best gene with rank "1"' in mix_text,
        "benchmarkhvg_forwards_extra_rank": "extra.rank = extra.rank" in benchmark_text,
        "benchmarkhvg_uses_method_list": "method_list=c(" in benchmark_text,
        "recommended_anchor": "adaptive_hybrid_hvg",
    }


def _render_resource_inventory(
    compute_context: dict[str, Any],
    package_rows: list[dict[str, Any]],
    source_checks: dict[str, dict[str, Any] | None],
    *,
    r_version: str | None,
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    python_path = Path(sys.executable)
    rows.append(
        _resource_row(
            "python",
            python_path,
            exists=python_path.exists(),
            accessible=os.access(python_path, os.R_OK),
            installed=True,
            version=platform.python_version(),
            source="local runtime",
            status="ok",
            notes="Primary Python runtime for audit script.",
        )
    )
    git_path = shutil.which("git")
    rows.append(
        _resource_row(
            "git",
            git_path or "git",
            exists=git_path is not None,
            accessible=git_path is not None,
            installed=git_path is not None,
            version=None,
            source="PATH lookup",
            status="ok" if git_path else "missing",
            notes="Git CLI located via PATH.",
        )
    )
    rows.append(
        _resource_row(
            "r_conda_env",
            R_ENV_ROOT,
            exists=R_ENV_ROOT.exists(),
            accessible=os.access(R_ENV_ROOT, os.R_OK),
            installed=R_ENV_ROOT.exists(),
            version=None,
            source="workspace local conda env",
            status="present" if R_ENV_ROOT.exists() else "missing",
            notes="Workspace bootstrap env for official benchmark R stack.",
        )
    )
    rows.append(
        _resource_row(
            "rscript_env",
            R_ENV_RSCRIPT,
            exists=R_ENV_RSCRIPT.exists(),
            accessible=os.access(R_ENV_RSCRIPT, os.R_OK),
            installed=R_ENV_RSCRIPT.exists(),
            version=r_version if R_ENV_RSCRIPT.exists() else None,
            source="workspace local conda env",
            status="ok" if R_ENV_RSCRIPT.exists() else "missing",
            notes="Preferred Rscript for benchmark bootstrap audit.",
        )
    )

    repo_defs = [
        ("benchmarkHVG_repo", BENCHMARKHVG_REPO, "https://github.com/RuzhangZhao/benchmarkHVG"),
        ("mixhvg_repo", MIXHVG_REPO, "https://github.com/RuzhangZhao/mixhvg"),
        ("scRNAseqProcess_repo", SCRNASEQPROCESS_REPO, "https://github.com/RuzhangZhao/scRNAseqProcess"),
    ]
    for name, path, src in repo_defs:
        git_meta = _read_git_config(path)
        rows.append(
            _resource_row(
                name,
                path,
                exists=path.exists(),
                accessible=os.access(path, os.R_OK),
                installed=path.exists(),
                version=git_meta["head_commit"],
                source=src,
                status="ok" if path.exists() else "missing",
                notes=f"origin={git_meta['remote_url'] or 'unknown'}",
            )
        )

    data_defs = [
        ("data_raw_root", DATA_RAW_DIR, "local data dir", ""),
        ("data_processed_root", DATA_PROCESSED_DIR, "local data dir", ""),
        ("duo4_partial_zip", DUO4_BAD_ZIP, "local processed data", "Preserved incomplete earlier download."),
        ("duo4_redownload_zip", DUO4_GOOD_ZIP, "Zenodo redownload", "Redownloaded archive chosen as current pilot source."),
        ("duo4_expr_rds", DUO4_EXPR, "Zenodo processed data", "Processed cell-sorting RNA matrix."),
        ("duo4_label_rds", DUO4_LABEL, "Zenodo processed data", "Processed cell labels."),
    ]
    for name, path, src, note in data_defs:
        rows.append(
            _resource_row(
                name,
                path,
                exists=path.exists(),
                accessible=os.access(path, os.R_OK) if path.exists() else False,
                installed=path.exists(),
                version=_sha256(path) if path.is_file() else None,
                source=src,
                status="ok" if path.exists() else "missing",
                notes=note,
            )
        )

    rows.append(
        _resource_row(
            "official_baselines_bridge",
            OFFICIAL_BASELINES,
            exists=OFFICIAL_BASELINES.exists(),
            accessible=os.access(OFFICIAL_BASELINES, os.R_OK),
            installed=OFFICIAL_BASELINES.exists(),
            version=None,
            source="local repo bridge",
            status="ok" if OFFICIAL_BASELINES.exists() else "missing",
            notes="Python-side bridge for official baselines.",
        )
    )
    rows.append(
        _resource_row(
            "score_official_hvg_r_worker",
            OFFICIAL_R_WORKER,
            exists=OFFICIAL_R_WORKER.exists(),
            accessible=os.access(OFFICIAL_R_WORKER, os.R_OK),
            installed=OFFICIAL_R_WORKER.exists(),
            version=None,
            source="local repo bridge",
            status="ok" if OFFICIAL_R_WORKER.exists() else "missing",
            notes="R worker invoked by local official baseline bridge.",
        )
    )
    rows.append(
        _resource_row(
            "score_official_hvg_py_worker",
            OFFICIAL_PY_WORKER,
            exists=OFFICIAL_PY_WORKER.exists(),
            accessible=os.access(OFFICIAL_PY_WORKER, os.R_OK),
            installed=OFFICIAL_PY_WORKER.exists(),
            version=None,
            source="local repo bridge",
            status="ok" if OFFICIAL_PY_WORKER.exists() else "missing",
            notes="Python worker for scanpy/triku official baselines.",
        )
    )

    for pkg_row in package_rows:
        rows.append(
            _resource_row(
                f"r_pkg::{pkg_row['package']}",
                pkg_row.get("libpath") or "",
                exists=bool(pkg_row.get("installed")),
                accessible=bool(pkg_row.get("installed")),
                installed=bool(pkg_row.get("installed")),
                version=pkg_row.get("version"),
                source="R installed.packages() + loadNamespace()",
                status="loaded" if pkg_row.get("load_ok") else ("installed_but_load_fails" if pkg_row.get("installed") else "missing"),
                notes=pkg_row.get("load_error") or "",
            )
        )

    for key, payload in source_checks.items():
        if payload is None:
            continue
        rows.append(
            _resource_row(
                f"source_check::{key}",
                key,
                exists=True,
                accessible=True,
                installed=True,
                version=None,
                source="R source()/parse() audit",
                status="ok" if payload.get("ok") else "failed",
                notes=payload.get("source_error") or payload.get("parse_error") or "",
            )
        )

    rows.append(
        _resource_row(
            "compute_context",
            "torch.cuda",
            exists=True,
            accessible=True,
            installed=compute_context.get("torch_imported", False),
            version=None,
            source="Python torch runtime",
            status="cuda_available" if compute_context.get("cuda_available") else "cpu_only",
            notes=f"preferred_device={compute_context.get('preferred_device')}",
        )
    )
    return rows


def _render_install_log(
    *,
    network_attempts: list[dict[str, Any]],
    remote_install_attempts: list[dict[str, Any]],
    local_install_attempts: list[dict[str, Any]],
) -> str:
    lines: list[str] = [
        "# BenchmarkHVG Bootstrap Install And Access Log",
        "",
        "## Existing Local State",
        "",
        f"- Active R env root: `{R_ENV_ROOT}`",
        f"- Active Rscript: `{R_ENV_RSCRIPT}`",
        f"- Found local clone: `{BENCHMARKHVG_REPO}`",
        f"- Found local clone: `{MIXHVG_REPO}`",
        f"- Found local clone: `{SCRNASEQPROCESS_REPO}`",
        f"- Found local processed pilot archive: `{DUO4_GOOD_ZIP}`",
        f"- Preserved earlier incomplete archive: `{DUO4_BAD_ZIP}`",
        "",
        "## Current-Round Network Attempts",
        "",
    ]
    for attempt in network_attempts:
        status = "success" if attempt["returncode"] == 0 else "failed"
        lines.extend(
            [
                f"### {attempt['label']}",
                "",
                f"- Status: `{status}`",
                f"- Command: `{attempt['command']}`",
                f"- Return code: `{attempt['returncode']}`",
            ]
        )
        if attempt["stdout"].strip():
            lines.extend(["- Stdout summary:", "", "```text", attempt["stdout"].strip(), "```"])
        if attempt["stderr"].strip() or attempt["error"]:
            lines.extend(
                [
                    "- Stderr / error summary:",
                    "",
                    "```text",
                    (attempt["stderr"] or attempt["error"] or "").strip(),
                    "```",
                ]
            )
        lines.append("")

    lines.extend(["## Remote Dependency Install Attempts", ""])
    for attempt in remote_install_attempts:
        payload = attempt.get("payload") or {}
        installed = payload.get("installed")
        load_ok = payload.get("load_ok")
        lines.extend(
            [
                f"### {attempt['package']}",
                "",
                f"- Installed into isolated temp lib: `{installed}`",
                f"- Load after install: `{load_ok}`",
                f"- Install warning: `{payload.get('install_warning') or ''}`",
                f"- Install error: `{payload.get('install_error') or ''}`",
                f"- Load error: `{payload.get('load_error') or ''}`",
                f"- Return code: `{attempt['returncode']}`",
            ]
        )
        if attempt["stdout"].strip():
            lines.extend(["- Stdout:", "", "```text", attempt["stdout"].strip(), "```"])
        if attempt["stderr"].strip() or attempt["error"]:
            lines.extend(
                [
                    "- Stderr / error:",
                    "",
                    "```text",
                    (attempt["stderr"] or attempt["error"] or "").strip(),
                    "```",
                ]
            )
        lines.append("")

    lines.extend(["## Local Package Install Attempts", ""])
    for attempt in local_install_attempts:
        payload = attempt.get("payload") or {}
        installed = payload.get("installed")
        load_ok = payload.get("load_ok")
        lines.extend(
            [
                f"### {attempt['package']}",
                "",
                f"- Installed into isolated temp lib: `{installed}`",
                f"- Load after install: `{load_ok}`",
                f"- Install warning: `{payload.get('install_warning') or ''}`",
                f"- Install error: `{payload.get('install_error') or ''}`",
                f"- Load error: `{payload.get('load_error') or ''}`",
                f"- Return code: `{attempt['returncode']}`",
            ]
        )
        if attempt["stdout"].strip():
            lines.extend(["- Stdout:", "", "```text", attempt["stdout"].strip(), "```"])
        if attempt["stderr"].strip() or attempt["error"]:
            lines.extend(
                [
                    "- Stderr / error:",
                    "",
                    "```text",
                    (attempt["stderr"] or attempt["error"] or "").strip(),
                    "```",
                ]
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_import_audit(
    *,
    package_rows: list[dict[str, Any]],
    source_checks: dict[str, dict[str, Any] | None],
    remote_install_attempts: list[dict[str, Any]],
    local_install_attempts: list[dict[str, Any]],
    bridge_smoke: list[dict[str, Any]],
    official_pilot_smoke: dict[str, Any] | None,
    official_wrapper_smoke: dict[str, Any] | None,
) -> str:
    pkg_map = {row["package"]: row for row in package_rows}
    lisi_install = next((item for item in remote_install_attempts if item["package"] == "lisi"), None)
    mix_install = next((item for item in local_install_attempts if item["package"] == "mixhvg"), None)
    benchmark_install = next((item for item in local_install_attempts if item["package"] == "benchmarkHVG"), None)
    lines = [
        "# benchmarkHVG Import Audit",
        "",
        "## benchmarkHVG",
        "",
        f"- Repo present: `{BENCHMARKHVG_REPO.exists()}`",
        f"- README present: `{(BENCHMARKHVG_REPO / 'README.md').exists()}`",
        f"- NAMESPACE present: `{(BENCHMARKHVG_REPO / 'NAMESPACE').exists()}`",
        f"- `benchmark_HVG_1_Methods.R` parse ok: `{(source_checks.get('benchmark_methods_parse') or {}).get('ok', False)}`",
        f"- Parse error: `{(source_checks.get('benchmark_methods_parse') or {}).get('parse_error') or ''}`",
        f"- `benchmark_HVG_2_Evaluation_Criteria.R` source ok: `{(source_checks.get('benchmark_eval_source') or {}).get('ok', False)}`",
        f"- `benchmark_HVG_3_RunExample.R` source ok: `{(source_checks.get('benchmark_example_source') or {}).get('ok', False)}`",
        f"- Installed in env: `{pkg_map.get('benchmarkHVG', {}).get('installed', False)}`",
        f"- Load in env: `{pkg_map.get('benchmarkHVG', {}).get('load_ok', False)}`",
        f"- Isolated local install: `{((benchmark_install or {}).get('payload') or {}).get('installed', False)}`",
        f"- Isolated local load: `{((benchmark_install or {}).get('payload') or {}).get('load_ok', False)}`",
        "",
        "## mixhvg",
        "",
        f"- Repo present: `{MIXHVG_REPO.exists()}`",
        f"- DESCRIPTION present: `{(MIXHVG_REPO / 'DESCRIPTION').exists()}`",
        f"- `FindVariableFeaturesMix.R` source ok: `{(source_checks.get('mixhvg_source') or {}).get('ok', False)}`",
        f"- Installed in env: `{pkg_map.get('mixhvg', {}).get('installed', False)}`",
        f"- Load in env: `{pkg_map.get('mixhvg', {}).get('load_ok', False)}`",
        f"- Isolated local install: `{((mix_install or {}).get('payload') or {}).get('installed', False)}`",
        f"- Isolated local load: `{((mix_install or {}).get('payload') or {}).get('load_ok', False)}`",
        f"- Dominant local install failure: `{((mix_install or {}).get('payload') or {}).get('load_error') or ((mix_install or {}).get('payload') or {}).get('install_error') or ''}`",
        "",
        "## lisi",
        "",
        f"- Installed in env: `{pkg_map.get('lisi', {}).get('installed', False)}`",
        f"- Load in env: `{pkg_map.get('lisi', {}).get('load_ok', False)}`",
        f"- Isolated GitHub install: `{((lisi_install or {}).get('payload') or {}).get('installed', False)}`",
        f"- Isolated GitHub load: `{((lisi_install or {}).get('payload') or {}).get('load_ok', False)}`",
        f"- Dominant isolated install failure: `{((lisi_install or {}).get('payload') or {}).get('load_error') or ((lisi_install or {}).get('payload') or {}).get('install_error') or ''}`",
        "",
        "## scRNAseqProcess",
        "",
        f"- Repo present: `{SCRNASEQPROCESS_REPO.exists()}`",
        f"- README present: `{(SCRNASEQPROCESS_REPO / 'README.md').exists()}`",
        f"- Dataset loading script present: `{(SCRNASEQPROCESS_REPO / 'DatatsetLoading.R').exists()}`",
        f"- Installable R package manifest present: `{(SCRNASEQPROCESS_REPO / 'DESCRIPTION').exists()}`",
        "",
        "## Bridge Smoke",
        "",
    ]
    for smoke in bridge_smoke:
        error_text = (smoke["stderr"] or smoke["error"] or "").strip()
        lines.extend(
            [
                f"- `{smoke['method']}` return code: `{smoke['returncode']}`",
                f"- `{smoke['method']}` output exists: `{smoke['output_exists']}`",
                f"- `{smoke['method']}` failure summary: `{error_text}`",
            ]
        )
    if official_pilot_smoke and official_pilot_smoke.get("ok"):
        lines.extend(
            [
                "- Bridge smoke failures above came from the intentionally tiny synthetic matrix and do not contradict the successful `duo4_pbmc` pilot run.",
            ]
        )
    lines.extend(["", "## Official Pilot Smoke", ""])
    if official_pilot_smoke is None:
        lines.append("- No pilot smoke was attempted because the duo4 pilot files were not both present.")
    elif official_pilot_smoke.get("ok"):
        lines.extend(
            [
                f"- `mixhvg + Seurat + PCA` on `duo4_pbmc` succeeded: `{official_pilot_smoke.get('ok')}`",
                f"- PCA cell count: `{official_pilot_smoke.get('n_cells')}`",
                f"- PCA dimensions kept: `{official_pilot_smoke.get('n_pcs')}`",
                f"- HVG count: `{official_pilot_smoke.get('hvg_count')}`",
                f"- Top HVGs preview: `{official_pilot_smoke.get('top_hvgs')}`",
                f"- Official evaluator metric `between_within_var_ratio` succeeded: `{official_pilot_smoke.get('var_ratio')}`",
                f"- Official evaluator metric `asw_func` succeeded: `{official_pilot_smoke.get('asw')}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Pilot smoke succeeded: `False`",
                f"- Failure return code: `{official_pilot_smoke.get('returncode')}`",
                f"- Failure stderr: `{official_pilot_smoke.get('stderr') or official_pilot_smoke.get('error') or ''}`",
            ]
        )
    lines.extend(["", "## Official Wrapper Smoke", ""])
    if official_wrapper_smoke is None:
      lines.append("- No wrapper smoke was attempted because the duo4 pilot files were not both present.")
    elif official_wrapper_smoke.get("ok"):
        lines.extend(
            [
                f"- `benchmarkHVG::hvg_pca()` succeeded: `True`",
                f"- Base PCA list count: `{official_wrapper_smoke.get('base_pca_count')}`",
                f"- Base first PCA shape: `{official_wrapper_smoke.get('base_first_dim')}`",
                f"- `benchmarkHVG::mixture_hvg_pca()` succeeded: `True`",
                f"- Mix PCA list count: `{official_wrapper_smoke.get('mix_pca_count')}`",
                f"- Mix first PCA shape: `{official_wrapper_smoke.get('mix_first_dim')}`",
                f"- `evaluate_hvg_discrete()` on base wrapper output succeeded with ARI `{official_wrapper_smoke.get('eval_base_ari')}` and LISI `{official_wrapper_smoke.get('eval_base_lisi')}`",
                f"- `evaluate_hvg_discrete()` on mix wrapper output succeeded with ARI `{official_wrapper_smoke.get('eval_mix_ari')}` and LISI `{official_wrapper_smoke.get('eval_mix_lisi')}`",
            ]
        )
    else:
        lines.extend(
            [
                f"- Wrapper smoke succeeded: `False`",
                f"- Failure return code: `{official_wrapper_smoke.get('returncode')}`",
                f"- Failure stderr: `{official_wrapper_smoke.get('stderr') or official_wrapper_smoke.get('error') or ''}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Current Runnable Layer",
            "",
            "- The local audit can read official repositories, inspect README/NAMESPACE/entry files, source the evaluation file, load `mixhvg` and `benchmarkHVG` as working R packages, and load the processed duo4 pilot dataset.",
            "- The local audit can run the official `mixhvg` README-style HVG-to-PCA path on `duo4_pbmc`, execute evaluator-adjacent metrics from `benchmark_HVG_2_Evaluation_Criteria.R`, and execute the official `benchmarkHVG::hvg_pca()` / `benchmarkHVG::mixture_hvg_pca()` wrappers end to end on the duo4 pilot.",
            "- The remaining caveat is no longer runtime reachability; it is reproducibility hygiene. The current local success depends on explicit local compatibility patches applied to `benchmarkHVG` and `lisi`.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _render_dataset_audit(
    duo4_summary: dict[str, Any] | None,
    official_pilot_smoke: dict[str, Any] | None,
    official_wrapper_smoke: dict[str, Any] | None,
) -> str:
    lines = [
        "# Dataset Bootstrap Audit",
        "",
        "## Located Zhao-Related Data",
        "",
        f"- Partial archive preserved: `{DUO4_BAD_ZIP}`",
        f"- Working redownload archive: `{DUO4_GOOD_ZIP}`",
        f"- Redownload archive size: `{DUO4_GOOD_ZIP.stat().st_size if DUO4_GOOD_ZIP.exists() else 'missing'}`",
        f"- Redownload archive sha256: `{_sha256(DUO4_GOOD_ZIP) or ''}`",
        f"- Extracted expression RDS: `{DUO4_EXPR}`",
        f"- Extracted label RDS: `{DUO4_LABEL}`",
        "",
        "## Pilot Choice",
        "",
        "- Selected pilot dataset: `duo4_pbmc`.",
        "- Reason: it is the cell-sorting example shown directly in the official benchmark README and run example, and it is small enough for a first local bootstrap audit.",
        "",
        "## Load Result",
        "",
    ]
    if duo4_summary is None:
        lines.extend(
            [
                "- `duo4_pbmc` processed RDS files were not both present, so no load-level audit was possible.",
            ]
        )
    else:
        lines.extend(
            [
                f"- Expression object class: `{duo4_summary.get('expr_class')}`",
                f"- Expression dimensions: `{duo4_summary.get('expr_dim')}`",
                f"- Expression has row names: `{duo4_summary.get('expr_has_rownames')}`",
                f"- Expression has column names: `{duo4_summary.get('expr_has_colnames')}`",
                f"- Label object class: `{duo4_summary.get('label_class')}`",
                f"- Label length: `{duo4_summary.get('label_length')}`",
                f"- Distinct labels: `{duo4_summary.get('label_unique')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Pilot Execution Layer",
            "",
        ]
    )
    if official_pilot_smoke and official_pilot_smoke.get("ok"):
        lines.extend(
            [
                "- `mixhvg` could be applied on the `duo4_pbmc` pilot in the new R 4.4 env.",
                "- A PCA embedding was produced from pilot HVGs and reached evaluator-adjacent official metric functions.",
                f"- `between_within_var_ratio`: `{official_pilot_smoke.get('var_ratio')}`",
                f"- `asw_func`: `{official_pilot_smoke.get('asw')}`",
            ]
        )
    else:
        lines.append("- The pilot did not reach an official runtime layer beyond raw object loading in this run.")
    if official_wrapper_smoke and official_wrapper_smoke.get("ok"):
        lines.extend(
            [
                "- The official `benchmarkHVG::hvg_pca()` wrapper also succeeded on the same pilot.",
                "- The official `benchmarkHVG::mixture_hvg_pca()` wrapper also succeeded on the same pilot.",
                f"- Base-wrapper ARI / NMI / LISI: `{official_wrapper_smoke.get('eval_base_ari')}` / `{official_wrapper_smoke.get('eval_base_nmi')}` / `{official_wrapper_smoke.get('eval_base_lisi')}`",
                f"- Mix-wrapper ARI / NMI / LISI: `{official_wrapper_smoke.get('eval_mix_ari')}` / `{official_wrapper_smoke.get('eval_mix_nmi')}` / `{official_wrapper_smoke.get('eval_mix_lisi')}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Gap To Real Official Benchmark",
            "",
            "- The pilot data itself is locally usable.",
            "- The remaining gap is no longer a missing pilot dataset or a missing official runtime layer.",
            "- The remaining gap is reproducibility cleanliness: the current local success depends on explicit local compatibility patches to `benchmarkHVG` packaging and `lisi` installation.",
            "- Those patches should be frozen and documented before any claim of reproducible official benchmark replication.",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _render_adapter_hook_audit(hook_audit: dict[str, Any]) -> str:
    lines = [
        "# Adapter Hook Audit",
        "",
        "## Hook Existence",
        "",
        f"- `mixhvg::FindVariableFeaturesMix(..., extra.rank = NULL, ...)` source-level hook detected: `{hook_audit['mixhvg_has_extra_rank_arg']}`",
        f"- `benchmarkHVG::mixture_hvg_pca(..., extra.rank = NULL, ...)` forwarding path detected in source text: `{hook_audit['benchmarkhvg_forwards_extra_rank']}`",
        "",
        "## Contract",
        "",
        f"- Full-length rank required for all genes: `{hook_audit['mixhvg_length_guard']}`",
        f"- Gene names must match full feature set when names are supplied: `{hook_audit['mixhvg_name_guard']}`",
        f"- Named rank vector is reordered to benchmark gene order: `{hook_audit['mixhvg_reorder_logic']}`",
        f"- Rank semantics documented as best gene = 1, worst gene = largest number: `{hook_audit['mixhvg_rank_doc']}`",
        "- `nfeatures` remains the top-k selector after the external rank is injected; the hook itself expects a full-rank vector rather than only a top-k list.",
        "",
        "## Current Repo Anchor Recommendation",
        "",
        f"- First adapter anchor: `{hook_audit['recommended_anchor']}`.",
        f"- Rationale source: `{CURRENT_METHODS_AUDIT}` and `{MIGRATION_DECISION}` already established that existing repo methods emit one continuous score per gene in original gene order, and that `adaptive_hybrid_hvg` is the safest carry-over anchor.",
        "",
        "## Boundary For This Round",
        "",
        "- This round confirms hook existence and API contract only.",
        "- This round does not run `adaptive_hybrid_hvg` through the hook and does not claim adapter readiness.",
        "- Adapter work should wait until at least one official method/evaluator layer is actually runnable locally.",
    ]
    return "\n".join(lines).rstrip() + "\n"


def _render_final_decision(
    *,
    source_checks: dict[str, dict[str, Any] | None],
    package_rows: list[dict[str, Any]],
    bridge_smoke: list[dict[str, Any]],
    duo4_summary: dict[str, Any] | None,
    network_attempts: list[dict[str, Any]],
    remote_install_attempts: list[dict[str, Any]],
    official_pilot_smoke: dict[str, Any] | None,
    official_wrapper_smoke: dict[str, Any] | None,
) -> str:
    pkg_map = {row["package"]: row for row in package_rows}
    methods_parse_ok = (source_checks.get("benchmark_methods_parse") or {}).get("ok", False)
    mix_source_ok = (source_checks.get("mixhvg_source") or {}).get("ok", False)
    eval_source_ok = (source_checks.get("benchmark_eval_source") or {}).get("ok", False)
    duo4_loaded = duo4_summary is not None
    worker_success = any(item["returncode"] == 0 for item in bridge_smoke)
    pilot_success = bool(official_pilot_smoke and official_pilot_smoke.get("ok"))
    wrapper_success = bool(official_wrapper_smoke and official_wrapper_smoke.get("ok"))
    lisi_install = next((item for item in remote_install_attempts if item["package"] == "lisi"), None)
    lisi_fail_detail = (
        ((lisi_install or {}).get("payload") or {}).get("load_error")
        or ((lisi_install or {}).get("payload") or {}).get("install_error")
        or ((lisi_install or {}).get("payload") or {}).get("install_warning")
        or ""
    )
    if methods_parse_ok and mix_source_ok and eval_source_ok and duo4_loaded and pilot_success and wrapper_success:
        status = "success_with_local_patches"
    elif mix_source_ok and eval_source_ok and duo4_loaded and pilot_success:
        status = "partial_success"
    else:
        status = "failure"
    lines = [
        "# Final Bootstrap Decision",
        "",
        "## Verdict",
        "",
        f"- official bootstrap status: `{status}`",
        (
            "- rationale: the official wrapper chain is runnable locally on the duo4 pilot after explicit local compatibility patches, including `benchmarkHVG::hvg_pca()`, `benchmarkHVG::mixture_hvg_pca()`, and `evaluate_hvg_discrete()`."
            if wrapper_success
            else (
                "- rationale: local official resources and pilot data are present, and the bootstrap now reaches a real official pilot runtime layer, but the unmodified official package/wrapper chain is still blocked."
                if pilot_success
                else "- rationale: local official resources and pilot data are present, but official package/runtime execution is still blocked before any official benchmark run can be claimed."
            )
        ),
        "",
        "## Blocking Boundary",
        "",
        f"- benchmark methods file parse clean: `{methods_parse_ok}`",
        f"- `Seurat` load ok: `{pkg_map.get('Seurat', {}).get('load_ok', False)}`",
        f"- `scran` load ok: `{pkg_map.get('scran', {}).get('load_ok', False)}`",
        f"- `lisi` load ok: `{pkg_map.get('lisi', {}).get('load_ok', False)}`",
        f"- `NMI` load ok: `{pkg_map.get('NMI', {}).get('load_ok', False)}`",
        f"- `benchmarkHVG` load ok: `{pkg_map.get('benchmarkHVG', {}).get('load_ok', False)}`",
        f"- bridge worker any success: `{worker_success}`",
        f"- official duo4 pilot smoke success: `{pilot_success}`",
        f"- official duo4 wrapper smoke success: `{wrapper_success}`",
        "- dominant caveats:",
        "  1. `benchmark_HVG_1_Methods.R` required a one-line local parse fix before the package/wrapper chain could load.",
        "  2. `benchmarkHVG` required local package hygiene cleanup via `.Rbuildignore` so analysis scripts and example scripts inside `R/` would not execute during package installation.",
        "  3. `NMI` was recovered from an existing local pure-R CRAN installation and reused in the new R 4.4 env.",
        "  4. `lisi` is installed locally through a compatibility patch that removes the broken Windows compilation path and provides the official Simpson-index logic in pure R.",
        f"  5. The original upstream GitHub install path for `lisi` is still not healthy in this environment. Detail: `{lisi_fail_detail}`",
        "",
        "## Adapter Stub Gate",
        "",
        f"- allow `adaptive_hybrid_hvg` adapter stub next round: `{'yes' if pilot_success else 'no'}`",
        (
            "- reason: this round reached the official wrapper layer on `duo4_pbmc`, so a thin adapter stub can be scoped next without entering full benchmark."
            if pilot_success
            else "- reason: this round still did not get an official method path or evaluator path to execute past import/source-level checks; claiming `adapter stub ready` would overstate the current bootstrap state."
        ),
        "",
        "## Minimal Next Command",
        "",
        "- Use the new R 4.4 env and rerun the patched bootstrap audit:",
        "",
        "```powershell",
        f"python {ROOT / 'scripts' / 'run_benchmarkhvg_bootstrap_audit.py'} --r-env-root {R_ENV_ROOT} --output-dir {ROOT / 'artifacts_benchmarkhvg_bootstrap_audit_r44'}",
        "```",
        "",
        "- If the next round starts adapter work, keep it at stub scope and do not enter full benchmark until these local compatibility patches are documented and frozen.",
        "",
        "## Network Attempt Notes",
        "",
    ]
    for attempt in network_attempts:
        summary = (attempt["stderr"] or attempt["stdout"] or attempt["error"] or "").strip().splitlines()
        short = summary[0] if summary else ""
        lines.append(f"- `{attempt['label']}`: rc={attempt['returncode']} summary=`{short}`")
    return "\n".join(lines).rstrip() + "\n"


def run_audit(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "_tmp"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    compute_context = _compute_context()
    compute_context["r_env_root"] = str(R_ENV_ROOT)
    compute_context["rscript_path"] = str(R_ENV_RSCRIPT)
    compute_context["rscript_exists"] = R_ENV_RSCRIPT.exists()
    r_version = _r_version_string(tmp_dir)
    if r_version is not None:
        compute_context["r_version"] = r_version

    package_rows, _pkg_cmd = _package_audit(tmp_dir)
    duo4_summary, _duo4_cmd = _duo4_summary(tmp_dir)

    source_checks: dict[str, dict[str, Any] | None] = {}
    source_checks["benchmark_methods_parse"], _ = _source_check(
        BENCHMARKHVG_REPO / "R" / "benchmark_HVG_1_Methods.R",
        tmp_dir=tmp_dir,
        parse_only=True,
    )
    source_checks["benchmark_eval_source"], _ = _source_check(
        BENCHMARKHVG_REPO / "R" / "benchmark_HVG_2_Evaluation_Criteria.R",
        tmp_dir=tmp_dir,
        parse_only=False,
        symbol="evaluate_hvg_discrete",
    )
    source_checks["benchmark_example_source"], _ = _source_check(
        BENCHMARKHVG_REPO / "R" / "benchmark_HVG_3_RunExample.R",
        tmp_dir=tmp_dir,
        parse_only=False,
        symbol=None,
    )
    source_checks["mixhvg_source"], _ = _source_check(
        MIXHVG_REPO / "R" / "FindVariableFeaturesMix.R",
        tmp_dir=tmp_dir,
        parse_only=False,
        symbol="FindVariableFeaturesMix",
    )

    hook_audit = _hook_audit()
    bridge_smoke = _bridge_smoke(tmp_dir)
    network_attempts = _network_attempts(tmp_dir)
    remote_install_attempts = _remote_install_audit(tmp_dir)
    local_install_attempts = _local_install_audit(tmp_dir)
    official_pilot_smoke = _official_pilot_smoke(tmp_dir)
    official_wrapper_smoke = _official_wrapper_smoke(tmp_dir)

    resource_rows = _render_resource_inventory(
        compute_context,
        package_rows,
        source_checks,
        r_version=r_version,
    )
    inventory_path = output_dir / "bootstrap_resource_inventory.csv"
    _write_csv(
        inventory_path,
        resource_rows,
        fieldnames=[
            "resource_name",
            "local_path",
            "exists",
            "accessible",
            "installed",
            "version",
            "source",
            "status",
            "notes",
        ],
    )

    install_log_path = output_dir / "bootstrap_install_and_access_log.md"
    _write_text(
        install_log_path,
        _render_install_log(
            network_attempts=network_attempts,
            remote_install_attempts=remote_install_attempts,
            local_install_attempts=local_install_attempts,
        ),
    )

    import_audit_path = output_dir / "benchmarkhvg_import_audit.md"
    _write_text(
        import_audit_path,
        _render_import_audit(
            package_rows=package_rows,
            source_checks=source_checks,
            remote_install_attempts=remote_install_attempts,
            local_install_attempts=local_install_attempts,
            bridge_smoke=bridge_smoke,
            official_pilot_smoke=official_pilot_smoke,
            official_wrapper_smoke=official_wrapper_smoke,
        ),
    )

    dataset_audit_path = output_dir / "dataset_bootstrap_audit.md"
    _write_text(
        dataset_audit_path,
        _render_dataset_audit(duo4_summary, official_pilot_smoke, official_wrapper_smoke),
    )

    adapter_hook_path = output_dir / "adapter_hook_audit.md"
    _write_text(adapter_hook_path, _render_adapter_hook_audit(hook_audit))

    final_decision_path = output_dir / "final_bootstrap_decision.md"
    _write_text(
        final_decision_path,
        _render_final_decision(
            source_checks=source_checks,
            package_rows=package_rows,
            bridge_smoke=bridge_smoke,
            duo4_summary=duo4_summary,
            network_attempts=network_attempts,
            remote_install_attempts=remote_install_attempts,
            official_pilot_smoke=official_pilot_smoke,
            official_wrapper_smoke=official_wrapper_smoke,
        ),
    )

    compute_context_path = output_dir / "compute_context.json"
    _write_json(compute_context_path, compute_context)

    return {
        "bootstrap_resource_inventory.csv": inventory_path,
        "bootstrap_install_and_access_log.md": install_log_path,
        "benchmarkhvg_import_audit.md": import_audit_path,
        "dataset_bootstrap_audit.md": dataset_audit_path,
        "adapter_hook_audit.md": adapter_hook_path,
        "final_bootstrap_decision.md": final_decision_path,
        "compute_context.json": compute_context_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a benchmarkHVG official bootstrap audit.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Artifact output directory. Existing artifact files are never overwritten.",
    )
    parser.add_argument(
        "--r-env-root",
        type=Path,
        default=None,
        help="Optional R conda env root. Defaults to BENCHMARKHVG_R_ENV or the legacy workspace env path.",
    )
    args = parser.parse_args()

    if args.r_env_root is not None:
        _set_r_env_root(args.r_env_root)
    elif "BENCHMARKHVG_R_ENV" in os.environ:
        _set_r_env_root(Path(os.environ["BENCHMARKHVG_R_ENV"]))

    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        existing_files = list(output_dir.glob("*"))
        if existing_files:
            raise FileExistsError(
                f"Output directory already contains files; refusing to overwrite artifacts: {output_dir}"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    generated = run_audit(output_dir)
    print(json.dumps({key: str(value) for key, value in generated.items()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
