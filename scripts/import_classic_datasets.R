# Refresh the classic dataset CSV snapshots from R 4.5.2 and pinned countreg data.
# Run from the checkout root with the directory holding CodParasites.rda and
# CrabSatellites.rda as the sole argument. See data-raw/datasets/README.md.
args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1L) stop("usage: Rscript scripts/import_classic_datasets.R COUNTREG_DATA_DIR")

raw_dir <- "data-raw/datasets"
manifest_path <- file.path(raw_dir, "classic_catalog.tsv")
manifest <- read.delim(manifest_path, colClasses = "character", na.strings = NULL,
                       quote = "", comment.char = "", check.names = FALSE)
manifest$string_columns <- ""

for (i in seq_len(nrow(manifest))) {
    entry <- manifest[i, ]
    transform <- strsplit(entry$transform, ":", fixed = TRUE)[[1L]]
    if (transform[1L] == "countreg") {
        dataset_env <- new.env(parent = emptyenv())
        load(file.path(args[1L], paste0(entry$r_object, ".rda")), envir = dataset_env)
        object <- get(entry$r_object, envir = dataset_env, inherits = FALSE)
    } else {
        object <- getExportedValue("datasets", entry$r_object)
    }
    frame <- switch(transform[1L],
        frame = as.data.frame(object),
        countreg = as.data.frame(object),
        counts = as.data.frame.table(object, responseName = "count"),
        series = data.frame(time = as.numeric(time(object)), value = as.numeric(object)),
        multiseries = data.frame(time = as.numeric(time(object)), unclass(object), check.names = FALSE),
        rownames = {
            frame <- as.data.frame(object)
            identifiers <- setNames(data.frame(rownames(frame)), transform[2L])
            cbind(identifiers, frame)
        },
        named = setNames(data.frame(names(object), as.numeric(object)), transform[2:3]),
        vector = setNames(data.frame(as.numeric(object)), transform[2L]),
        stop("unknown transform: ", entry$transform)
    )
    strings <- vapply(frame, function(column) is.factor(column) || is.character(column) ||
                         is.logical(column), logical(1))
    manifest$string_columns[i] <- paste(names(frame)[strings], collapse = "|")
    write.csv(frame, file.path(raw_dir, paste0(entry$name, ".csv")), row.names = FALSE,
              na = "", fileEncoding = "UTF-8")
}
manifest <- manifest[order(manifest$name, method = "radix"), ]
write.table(manifest, manifest_path, row.names = FALSE, quote = FALSE, sep = "\t",
            fileEncoding = "UTF-8")
