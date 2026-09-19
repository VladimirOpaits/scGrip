#!/usr/bin/env Rscript
# Export barcode-addressable annotations from a published Seurat RDS.
#
# The raw count archive and its processed publication object are kept separate:
# this script exports only metadata, so Python code can join it explicitly to a
# model-panel H5AD by (sample, barcode), rather than assuming row order.

arguments <- commandArgs(trailingOnly = TRUE)

option <- function(name, default = NULL, required = FALSE) {
  position <- match(name, arguments)
  if (is.na(position)) {
    if (required) stop(sprintf("Missing required option %s", name))
    return(default)
  }
  if (position == length(arguments)) stop(sprintf("Option %s needs a value", name))
  arguments[[position + 1]]
}

input <- option("--input", required = TRUE)
output <- option("--output", required = TRUE)
sample_col <- option("--sample-col", required = TRUE)
cell_col <- option("--cell-col", required = TRUE)
label_col <- option("--label-col", required = TRUE)
label_score_col <- option("--label-score-col")
mapping_score_col <- option("--mapping-score-col")
subcluster_col <- option("--subcluster-col")
barcode_prefix_regex <- option("--barcode-prefix-regex", "^S_[0-9]+_")

suppressPackageStartupMessages(library(Seurat))
object <- readRDS(input)
metadata <- object@meta.data
needed <- c(sample_col, cell_col, label_col)
missing <- setdiff(needed, colnames(metadata))
if (length(missing) > 0) {
  stop(sprintf("Published Seurat metadata misses: %s", paste(missing, collapse = ", ")))
}

optional_value <- function(column) {
  if (is.null(column)) return(rep(NA_character_, nrow(metadata)))
  if (!(column %in% colnames(metadata))) stop(sprintf("Published Seurat metadata misses optional column %s", column))
  as.character(metadata[[column]])
}

out <- data.frame(
  source_cell_name = rownames(metadata),
  sample_id = as.character(metadata[[sample_col]]),
  cell_barcode = sub(barcode_prefix_regex, "", as.character(metadata[[cell_col]])),
  published_label = as.character(metadata[[label_col]]),
  published_label_score = optional_value(label_score_col),
  published_mapping_score = optional_value(mapping_score_col),
  published_subcluster = optional_value(subcluster_col),
  stringsAsFactors = FALSE
)
if (any(is.na(out$sample_id)) || any(out$sample_id == "") || any(is.na(out$cell_barcode)) || any(out$cell_barcode == "")) {
  stop("Published metadata contains blank sample/barcode values after normalisation.")
}
key <- paste(out$sample_id, out$cell_barcode, sep = "\t")
if (anyDuplicated(key)) stop("(sample_id, cell_barcode) is not unique in published metadata.")
dir.create(dirname(output), recursive = TRUE, showWarnings = FALSE)
write.table(out, output, sep = "\t", quote = FALSE, row.names = FALSE, na = "")
cat(sprintf("wrote %s: %d barcode-addressable published labels\n", output, nrow(out)))
