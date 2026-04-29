#!/usr/bin/env Rscript

parse_args <- function(argv) {
  args <- list()
  i <- 1L
  while (i <= length(argv)) {
    key <- argv[[i]]
    if (!startsWith(key, "--")) {
      stop(sprintf("Unexpected argument: %s", key), call. = FALSE)
    }
    if (i == length(argv)) {
      stop(sprintf("Missing value for argument: %s", key), call. = FALSE)
    }
    args[[substring(key, 3L)]] <- argv[[i + 1L]]
    i <- i + 2L
  }
  args
}


escape_json_string <- function(value) {
  value <- gsub("\\\\", "\\\\\\\\", value)
  value <- gsub("\"", "\\\\\"", value)
  value <- gsub("\r", "\\\\r", value)
  value <- gsub("\n", "\\\\n", value)
  value
}


metadata_to_json <- function(metadata) {
  keys <- names(metadata)
  if (length(keys) == 0L) {
    return("{}")
  }
  entries <- vapply(
    keys,
    function(key) {
      value <- metadata[[key]]
      value_json <- if (is.null(value) || (length(value) == 1L && is.na(value))) {
        "null"
      } else if (is.logical(value)) {
        ifelse(value, "true", "false")
      } else if (is.numeric(value)) {
        format(value, scientific = FALSE, trim = TRUE, digits = 17)
      } else {
        sprintf("\"%s\"", escape_json_string(as.character(value[[1L]])))
      }
      sprintf("  \"%s\": %s", escape_json_string(key), value_json)
    },
    character(1L)
  )
  paste0("{\n", paste(entries, collapse = ",\n"), "\n}\n")
}


write_metadata <- function(path, metadata) {
  if (is.null(path) || identical(path, "")) {
    return(invisible(NULL))
  }
  if (requireNamespace("jsonlite", quietly = TRUE)) {
    jsonlite::write_json(metadata, path = path, auto_unbox = TRUE, pretty = TRUE)
    return(invisible(NULL))
  }
  writeLines(metadata_to_json(metadata), con = path, useBytes = TRUE)
  invisible(NULL)
}


require_worker_packages <- function(required_packages) {
  missing_packages <- required_packages[!vapply(required_packages, requireNamespace, logical(1L), quietly = TRUE)]
  if (length(missing_packages) > 0L) {
    stop(
      sprintf(
        "Missing required R packages: %s",
        paste(missing_packages, collapse = ", ")
      ),
      call. = FALSE
    )
  }
}


read_batches <- function(path) {
  if (is.null(path) || identical(path, "") || !file.exists(path)) {
    return(NULL)
  }
  values <- readLines(path, warn = FALSE, encoding = "UTF-8")
  if (length(values) == 0L || length(unique(values)) < 2L) {
    return(NULL)
  }
  factor(values)
}


score_scran_model_gene_var <- function(counts_mtx_path, batches_path, top_k) {
  require_worker_packages(c("Matrix", "SingleCellExperiment", "scuttle", "scran"))

  counts_matrix <- Matrix::t(Matrix::readMM(counts_mtx_path))
  sce <- SingleCellExperiment::SingleCellExperiment(list(counts = counts_matrix))
  sce <- scuttle::logNormCounts(sce)

  batch_values <- read_batches(batches_path)
  metadata <- list(
    official_backend = "scran",
    method = "scran_model_gene_var_hvg",
    top_k = as.integer(top_k),
    score_key = "bio",
    used_batch_key = "",
    batch_fallback_used = FALSE,
    transposed_input = TRUE,
    scran_version = as.character(utils::packageVersion("scran")),
    scuttle_version = as.character(utils::packageVersion("scuttle"))
  )

  if (!is.null(batch_values)) {
    if (length(batch_values) != ncol(sce)) {
      stop(
        sprintf(
          "Batch vector length mismatch: expected %d cells, got %d entries.",
          ncol(sce),
          length(batch_values)
        ),
        call. = FALSE
      )
    }
    metadata$used_batch_key <- "benchmark_batch"
    dec <- scran::modelGeneVar(sce, block = batch_values)
  } else {
    dec <- scran::modelGeneVar(sce)
  }

  scores <- as.numeric(dec[["bio"]])
  finite_mask <- is.finite(scores)
  if (any(!finite_mask)) {
    replacement <- if (any(finite_mask)) min(scores[finite_mask]) - 1 else 0
    scores[!finite_mask] <- replacement
  }

  top_n <- min(as.integer(top_k), length(scores))
  metadata$highly_variable_count <- as.integer(top_n)
  list(scores = scores, metadata = metadata)
}


score_seurat_r_vst <- function(counts_mtx_path, top_k) {
  require_worker_packages(c("Matrix", "Seurat"))

  counts_matrix <- Matrix::t(Matrix::readMM(counts_mtx_path))
  if (is.null(rownames(counts_matrix))) {
    rownames(counts_matrix) <- sprintf("gene_%d", seq_len(nrow(counts_matrix)))
  }
  if (is.null(colnames(counts_matrix))) {
    colnames(counts_matrix) <- sprintf("cell_%d", seq_len(ncol(counts_matrix)))
  }

  seu <- Seurat::CreateSeuratObject(counts = counts_matrix, assay = "RNA", project = "hvg")
  seu <- Seurat::FindVariableFeatures(
    object = seu,
    selection.method = "vst",
    nfeatures = as.integer(top_k),
    verbose = FALSE
  )

  hvf_info <- Seurat::HVFInfo(seu)
  score_candidates <- c("variance.standardized", "vst.variance.standardized", "variances_norm")
  score_key <- score_candidates[score_candidates %in% colnames(hvf_info)][1L]
  if (is.na(score_key) || is.null(score_key) || identical(score_key, "")) {
    stop("Seurat HVFInfo did not expose a standardized variance column.", call. = FALSE)
  }

  scores <- as.numeric(hvf_info[[score_key]])
  finite_mask <- is.finite(scores)
  if (any(!finite_mask)) {
    replacement <- if (any(finite_mask)) min(scores[finite_mask]) - 1 else 0
    scores[!finite_mask] <- replacement
  }

  metadata <- list(
    official_backend = "Seurat",
    method = "seurat_r_vst_hvg",
    top_k = as.integer(top_k),
    score_key = score_key,
    used_batch_key = "",
    batch_fallback_used = FALSE,
    transposed_input = TRUE,
    seurat_version = as.character(utils::packageVersion("Seurat")),
    highly_variable_count = as.integer(min(as.integer(top_k), length(scores)))
  )
  list(scores = scores, metadata = metadata)
}


main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  required_keys <- c("method", "counts-mtx-path", "top-k", "output-path")
  missing_keys <- required_keys[!required_keys %in% names(args)]
  if (length(missing_keys) > 0L) {
    stop(sprintf("Missing required arguments: %s", paste(missing_keys, collapse = ", ")), call. = FALSE)
  }

  method <- args[["method"]]
  top_k <- as.integer(args[["top-k"]])
  if (is.na(top_k) || top_k < 1L) {
    stop("--top-k must be a positive integer.", call. = FALSE)
  }

  if (!method %in% c("scran_model_gene_var_hvg", "seurat_r_vst_hvg")) {
    stop(sprintf("Unsupported R baseline method: %s", method), call. = FALSE)
  }

  result <- if (identical(method, "scran_model_gene_var_hvg")) {
    score_scran_model_gene_var(
      counts_mtx_path = args[["counts-mtx-path"]],
      batches_path = if ("batches-path" %in% names(args)) args[["batches-path"]] else NULL,
      top_k = top_k
    )
  } else {
    score_seurat_r_vst(
      counts_mtx_path = args[["counts-mtx-path"]],
      top_k = top_k
    )
  }

  writeLines(format(result$scores, scientific = FALSE, trim = TRUE, digits = 17), con = args[["output-path"]], useBytes = TRUE)
  if ("metadata-path" %in% names(args)) {
    write_metadata(args[["metadata-path"]], result$metadata)
  }
}


main()
