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


require_packages <- function(pkgs) {
  missing <- pkgs[!vapply(pkgs, requireNamespace, logical(1L), quietly = TRUE)]
  if (length(missing) > 0L) {
    stop(sprintf("Missing required R packages: %s", paste(missing, collapse = ", ")), call. = FALSE)
  }
}


write_json <- function(path, payload) {
  require_packages("jsonlite")
  jsonlite::write_json(payload, path = path, auto_unbox = TRUE, pretty = TRUE, null = "null")
}


coerce_expr_to_sparse <- function(expr) {
  if (inherits(expr, "dgCMatrix")) {
    return(expr)
  }
  if (is.data.frame(expr)) {
    expr <- as.matrix(expr)
  }
  if (is.matrix(expr) || inherits(expr, "Matrix")) {
    return(Matrix::Matrix(expr, sparse = TRUE))
  }
  stop(sprintf("Unsupported expression container class for export: %s", paste(class(expr), collapse = "|")), call. = FALSE)
}


export_dataset <- function(expr_rds, annotation_rds, annotation_mode, counts_mtx, gene_names_out, cell_names_out, metadata_out, annotation_out = NULL) {
  require_packages(c("Matrix", "jsonlite"))

  expr <- readRDS(expr_rds)
  annotation <- readRDS(annotation_rds)

  if (is.null(dim(expr)) || length(dim(expr)) != 2L) {
    stop("Expression object must be a 2D matrix-like object.", call. = FALSE)
  }
  if (is.null(rownames(expr)) || is.null(colnames(expr))) {
    stop("Expression matrix must have both row names and column names.", call. = FALSE)
  }
  if (!annotation_mode %in% c("discrete", "continuous")) {
    stop(sprintf("Unsupported annotation mode: %s", annotation_mode), call. = FALSE)
  }

  expr_sparse <- coerce_expr_to_sparse(expr)
  Matrix::writeMM(obj = expr_sparse, file = counts_mtx)
  writeLines(rownames(expr_sparse), con = gene_names_out, useBytes = TRUE)
  writeLines(colnames(expr_sparse), con = cell_names_out, useBytes = TRUE)
  annotation_matches_cells <- if (identical(annotation_mode, "discrete")) {
    length(annotation) == ncol(expr_sparse)
  } else {
    !is.null(dim(annotation)) && ncol(annotation) == ncol(expr_sparse)
  }
  if (!is.null(annotation_out) && identical(annotation_mode, "discrete")) {
    writeLines(as.character(annotation), con = annotation_out, useBytes = TRUE)
  }

  payload <- list(
    ok = TRUE,
    expr_rds = normalizePath(expr_rds, winslash = "/", mustWork = TRUE),
    annotation_rds = normalizePath(annotation_rds, winslash = "/", mustWork = TRUE),
    annotation_mode = annotation_mode,
    counts_mtx = normalizePath(counts_mtx, winslash = "/", mustWork = TRUE),
    gene_names = normalizePath(gene_names_out, winslash = "/", mustWork = TRUE),
    cell_names = normalizePath(cell_names_out, winslash = "/", mustWork = TRUE),
    expr_class = unname(class(expr)),
    expr_dim = unname(dim(expr)),
    annotation_class = unname(class(annotation)),
    annotation_dim = if (is.null(dim(annotation))) NULL else unname(dim(annotation)),
    annotation_length = length(annotation),
    annotation_matches_cells = annotation_matches_cells,
    annotation_path = if (is.null(annotation_out) || !file.exists(annotation_out)) NULL else normalizePath(annotation_out, winslash = "/", mustWork = TRUE),
    distinct_labels = if (identical(annotation_mode, "discrete")) {
      if (is.factor(annotation)) length(levels(annotation)) else length(unique(annotation))
    } else {
      NULL
    }
  )
  write_json(metadata_out, payload)
}


read_rank_vector <- function(path, expected_genes) {
  rank_df <- utils::read.delim(path, sep = "\t", header = TRUE, stringsAsFactors = FALSE, check.names = FALSE)
  required_cols <- c("gene", "rank")
  missing_cols <- required_cols[!required_cols %in% colnames(rank_df)]
  if (length(missing_cols) > 0L) {
    stop(sprintf("Rank file is missing required columns: %s", paste(missing_cols, collapse = ", ")), call. = FALSE)
  }

  gene <- as.character(rank_df$gene)
  rank <- as.numeric(rank_df$rank)
  if (anyNA(gene) || anyNA(rank)) {
    stop("Rank file contains NA gene names or NA ranks.", call. = FALSE)
  }
  if (length(gene) != length(expected_genes)) {
    stop(
      sprintf(
        "Rank vector length mismatch: expected %d genes, got %d rows.",
        length(expected_genes),
        length(gene)
      ),
      call. = FALSE
    )
  }
  if (!setequal(gene, expected_genes)) {
    stop("Rank vector gene names do not match expression row names.", call. = FALSE)
  }

  rank_vec <- setNames(rank, gene)
  rank_vec[match(expected_genes, names(rank_vec))]
}


metric_payload_discrete <- function(eval_obj, pca_obj, hvgs) {
  list(
    hvg_count = length(hvgs),
    pca_dim = unname(dim(pca_obj)),
    var_ratio = unname(eval_obj$var_ratio[[1]]),
    ari = unname(eval_obj$ari[[1]]),
    nmi = unname(eval_obj$nmi[[1]]),
    lisi = unname(eval_obj$lisi[[1]]),
    top_hvgs_head = unname(head(hvgs, 10L))
  )
}


metric_payload_continuous <- function(eval_obj, pca_obj, hvgs) {
  list(
    hvg_count = length(hvgs),
    pca_dim = unname(dim(pca_obj)),
    var_ratio = unname(eval_obj$var_ratio[[1]]),
    dist_cor = unname(eval_obj$dist_cor[[1]]),
    knn_ratio = unname(eval_obj$knn_ratio[[1]]),
    three_nn = unname(eval_obj[["3nn"]][[1]]),
    max_ari = unname(eval_obj$max_ari[[1]]),
    max_nmi = unname(eval_obj$max_nmi[[1]]),
    top_hvgs_head = unname(head(hvgs, 10L))
  )
}


prepare_continuous_annotation <- function(annotation, continuous_input) {
  if (identical(continuous_input, "CITEseq")) {
    require_packages(c("Seurat", "SeuratObject"))
    scale_pro <- Seurat::CreateSeuratObject(counts = annotation)
    scale_pro <- Seurat::NormalizeData(scale_pro, normalization.method = "CLR", margin = 2, verbose = FALSE)
    scale_pro <- Seurat::ScaleData(scale_pro, verbose = FALSE)
    pro_scaled <- SeuratObject::LayerData(scale_pro[["RNA"]], layer = "scale.data")
    return(list(pro = pro_scaled, eval_input = "MultiomeATAC", preprocessing = "local_citeseq_clr_scale_workaround"))
  }
  list(pro = annotation, eval_input = continuous_input, preprocessing = "none")
}


run_adapter_eval <- function(expr_rds, annotation_rds, annotation_mode, continuous_input, rank_tsv, output_json, nfeatures, method_list_csv) {
  require_packages(c("benchmarkHVG", "jsonlite"))

  expr <- readRDS(expr_rds)
  annotation <- readRDS(annotation_rds)
  expected_genes <- rownames(expr)
  if (is.null(expected_genes)) {
    stop("Expression matrix must expose gene row names.", call. = FALSE)
  }
  if (!annotation_mode %in% c("discrete", "continuous")) {
    stop(sprintf("Unsupported annotation mode: %s", annotation_mode), call. = FALSE)
  }

  rank_vec <- read_rank_vector(rank_tsv, expected_genes = expected_genes)
  seurat_expected_genes <- gsub("_", "-", expected_genes, fixed = TRUE)
  if (anyDuplicated(seurat_expected_genes) > 0L) {
    stop("Gene names become duplicated after Seurat underscore-to-dash normalization.", call. = FALSE)
  }
  names(rank_vec) <- seurat_expected_genes
  method_list <- trimws(strsplit(method_list_csv, ",", fixed = TRUE)[[1L]])
  if (length(method_list) == 0L || any(method_list == "")) {
    stop("method-list-csv must contain at least one method name.", call. = FALSE)
  }
  mixture_index_list <- list(seq_along(method_list))

  set.seed(1)
  baseline <- benchmarkHVG::mixture_hvg_pca(
    expr,
    nfeatures = as.integer(nfeatures),
    method_list = method_list,
    mixture_index_list = mixture_index_list
  )
  adapter <- benchmarkHVG::mixture_hvg_pca(
    expr,
    nfeatures = as.integer(nfeatures),
    method_list = method_list,
    extra.rank = rank_vec,
    mixture_index_list = mixture_index_list
  )

  if (identical(annotation_mode, "discrete")) {
    baseline_eval <- benchmarkHVG::evaluate_hvg_discrete(baseline$seurat.obj.pca[1], annotation, verbose = FALSE)
    adapter_eval <- benchmarkHVG::evaluate_hvg_discrete(adapter$seurat.obj.pca[1], annotation, verbose = FALSE)
    baseline_metrics <- metric_payload_discrete(
      eval_obj = baseline_eval,
      pca_obj = baseline$seurat.obj.pca[[1]],
      hvgs = baseline$var.seurat.obj[[1]]
    )
    adapter_metrics <- metric_payload_discrete(
      eval_obj = adapter_eval,
      pca_obj = adapter$seurat.obj.pca[[1]],
      hvgs = adapter$var.seurat.obj[[1]]
    )
  } else {
    continuous_cfg <- prepare_continuous_annotation(annotation, continuous_input)
    baseline_eval <- benchmarkHVG::evaluate_hvg_continuous(
      pcalist = baseline$seurat.obj.pca[1],
      pro = continuous_cfg$pro,
      input = continuous_cfg$eval_input,
      verbose = FALSE
    )
    adapter_eval <- benchmarkHVG::evaluate_hvg_continuous(
      pcalist = adapter$seurat.obj.pca[1],
      pro = continuous_cfg$pro,
      input = continuous_cfg$eval_input,
      verbose = FALSE
    )
    baseline_metrics <- metric_payload_continuous(
      eval_obj = baseline_eval,
      pca_obj = baseline$seurat.obj.pca[[1]],
      hvgs = baseline$var.seurat.obj[[1]]
    )
    adapter_metrics <- metric_payload_continuous(
      eval_obj = adapter_eval,
      pca_obj = adapter$seurat.obj.pca[[1]],
      hvgs = adapter$var.seurat.obj[[1]]
    )
  }

  baseline_hvgs <- baseline$var.seurat.obj[[1]]
  adapter_hvgs <- adapter$var.seurat.obj[[1]]
  rank_top <- names(sort(rank_vec, decreasing = FALSE))[seq_len(min(as.integer(nfeatures), length(rank_vec)))]

  payload <- list(
    ok = TRUE,
    evaluation_mode = annotation_mode,
    continuous_input = if (identical(annotation_mode, "continuous")) continuous_input else NULL,
    continuous_input_effective = if (identical(annotation_mode, "continuous")) continuous_cfg$eval_input else NULL,
    annotation_preprocessing = if (identical(annotation_mode, "continuous")) continuous_cfg$preprocessing else NULL,
    nfeatures = as.integer(nfeatures),
    method_list = method_list,
    mixture_index_list = lapply(mixture_index_list, as.integer),
    rank_contract = list(
      expected_gene_count = length(expected_genes),
      provided_gene_count = length(rank_vec),
      best_rank = unname(min(rank_vec)),
      worst_rank = unname(max(rank_vec)),
      contiguous = identical(sort(unique(as.integer(rank_vec))), seq_len(length(rank_vec)))
    ),
    baseline = baseline_metrics,
    adapter = adapter_metrics,
    overlap = list(
      adapter_vs_baseline_hvg_overlap = sum(adapter_hvgs %in% baseline_hvgs),
      adapter_vs_external_top_overlap = sum(adapter_hvgs %in% rank_top),
      baseline_vs_external_top_overlap = sum(baseline_hvgs %in% rank_top),
      external_top_head = unname(head(rank_top, 10L))
    )
  )
  write_json(output_json, payload)
}


main <- function() {
  args <- parse_args(commandArgs(trailingOnly = TRUE))
  if (!"mode" %in% names(args)) {
    stop("Missing required argument: --mode", call. = FALSE)
  }

  mode <- args[["mode"]]
  if (identical(mode, "export_dataset")) {
    required <- c(
      "expr-rds",
      "annotation-rds",
      "annotation-mode",
      "counts-mtx",
      "gene-names-out",
      "cell-names-out",
      "metadata-out"
    )
    missing <- required[!required %in% names(args)]
    if (length(missing) > 0L) {
      stop(sprintf("Missing required arguments for export_dataset: %s", paste(missing, collapse = ", ")), call. = FALSE)
    }
    export_dataset(
      expr_rds = args[["expr-rds"]],
      annotation_rds = args[["annotation-rds"]],
      annotation_mode = args[["annotation-mode"]],
      counts_mtx = args[["counts-mtx"]],
      gene_names_out = args[["gene-names-out"]],
      cell_names_out = args[["cell-names-out"]],
      metadata_out = args[["metadata-out"]],
      annotation_out = if ("annotation-out" %in% names(args)) args[["annotation-out"]] else NULL
    )
    return(invisible(NULL))
  }

  if (identical(mode, "run_adapter_eval")) {
    required <- c("expr-rds", "annotation-rds", "annotation-mode", "rank-tsv", "output-json", "nfeatures", "method-list-csv")
    missing <- required[!required %in% names(args)]
    if (length(missing) > 0L) {
      stop(sprintf("Missing required arguments for run_adapter_eval: %s", paste(missing, collapse = ", ")), call. = FALSE)
    }
    run_adapter_eval(
      expr_rds = args[["expr-rds"]],
      annotation_rds = args[["annotation-rds"]],
      annotation_mode = args[["annotation-mode"]],
      continuous_input = if ("continuous-input" %in% names(args)) args[["continuous-input"]] else "MultiomeATAC",
      rank_tsv = args[["rank-tsv"]],
      output_json = args[["output-json"]],
      nfeatures = as.integer(args[["nfeatures"]]),
      method_list_csv = args[["method-list-csv"]]
    )
    return(invisible(NULL))
  }

  stop(sprintf("Unsupported mode: %s", mode), call. = FALSE)
}


main()
