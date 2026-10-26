# Native R NN example. MNIST and Fashion-MNIST: strict IDX loading and nn preprocessing.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
# CMD: just r-example idx_loader
# Port of tflow's idx_loader. Create a tiny on-disk IDX pair with known
# bytes, load uint8 NHWC images and integer labels, shuffle matched
# image/label rows, then cast and divide by 255 with nn ops. Check malformed
# magic, truncated and trailing data, dimension overflow, count mismatches and
# label ranges before allocation.
#
# Also inspect cached real test sets under $DATAMUNGE_DATA_DIR/{mnist,
# fashion_mnist}. Optional root=PATH changes that root; dir=PATH loads a
# required IDX pair directly, and split=train|t10k selects its filename prefix.
#
# nn$read_idx reads one IDX file of any element type (and gunzips ".gz"), so
# the pair rules of tflow's ParseIdx are checked here in idx_loader_load_pair.
# Pixels come back as int32 holding 0..255 (nn has no uint8 dtype).
nn_helper("checks")

idx_loader_be32 <- function(value) as.raw(c(value %/% 16777216, value %/% 65536 %% 256, value %/% 256 %% 256, value %% 256))

idx_loader_write <- function(path, bytes) writeBin(as.raw(bytes), path)

# The first four bytes of a (possibly gzip-compressed) IDX file: a uint8
# file's magic is 0x0000 08 <rank>.
idx_loader_require_ubyte <- function(path, rank) {
    magic <- if (file.exists(path)) readBin(path, "raw", n = 4L) else raw(0)
    magic <- as.integer(magic)
    if (length(magic) == 4L && magic[1] == 0x1f && magic[2] == 0x8b)
        return(invisible(TRUE))
    if (length(magic) < 4L || magic[1] != 0L || magic[2] != 0L || magic[3] != 8L || magic[4] != rank)
        stop("IDX requires uint8 rank-3 images and rank-1 labels: ", path, call. = FALSE)
    invisible(TRUE)
}

# An MNIST-style IDX pair with tflow's ParseIdx rules.
idx_loader_load_pair <- function(images_path, labels_path, classes = 10) {
    idx_loader_require_ubyte(images_path, 3L)
    idx_loader_require_ubyte(labels_path, 1L)
    images <- nn$read_idx(images_path)
    labels <- nn$read_idx(labels_path)
    if (images$dim() != 3L || labels$dim() != 1L)
        stop("IDX requires rank-3 images and rank-1 labels", call. = FALSE)
    if (any(as.numeric(images$shape()) == 0))
        stop("invalid IDX dimensions", call. = FALSE)
    if (images$size(0L) != labels$size(0L))
        stop("IDX payload size/count mismatch", call. = FALSE)
    if (nn$max(labels)$item() >= classes)
        stop("IDX label out of range", call. = FALSE)
    list(images = nn$reshape(images, c(images$size(0L), images$size(1L), images$size(2L), 1)),
        labels = nn$cast(labels, nn$DType$Int64))
}

idx_loader_fixtures <- function() {
    pixels <- c(0, 51, 102, 153, 204, 255, 255, 204, 153, 102, 51, 0)
    images <- c(idx_loader_be32(2051), idx_loader_be32(2), idx_loader_be32(2), idx_loader_be32(3), as.raw(pixels))
    labels <- c(idx_loader_be32(2049), idx_loader_be32(2), as.raw(c(7, 3)))
    directory <- nn_output_dir("fixtures/idx")
    idx_loader_write(file.path(directory, "images.idx"), images)
    idx_loader_write(file.path(directory, "labels.idx"), labels)
    dataset <- idx_loader_load_pair(file.path(directory, "images.idx"), file.path(directory, "labels.idx"))
    nn_require(identical(as.numeric(dataset$images$shape()), c(2, 2, 3, 1)), "IDX NHWC shape")
    nn_require(all(dataset$images$values() == pixels), "IDX pixel order")
    y <- dataset$labels$values()
    nn_require(y[1] == 7 && y[2] == 3, "IDX label order")
    shuffled_x <- nn$take_rows(dataset$images, c(1, 0))
    shuffled_y <- nn$take_rows(dataset$labels, c(1, 0))
    nn_require(shuffled_x$at(c(0, 0, 0, 0)) == 255 && shuffled_y$at(0) == 3, "paired row gathering")

    normalized <- nn$div(nn$cast(shuffled_x, nn$DType$Float32), 255)
    means <- nn$mean(normalized, c(1L, 2L, 3L))
    n <- normalized$values()
    sx <- shuffled_x$values()
    for (i in seq_len(12L)) nn_near(n[i], sx[i] / 255, 1e-7, "pixel normalization")
    for (m in means$values()) nn_near(m, 0.5, 1e-7, "per-image mean")

    failures <- 0L
    reject <- function(bad_x, bad_y) {
        idx_loader_write(file.path(directory, "bad_images.idx"), bad_x)
        idx_loader_write(file.path(directory, "bad_labels.idx"), bad_y)
        nn_throws(idx_loader_load_pair(file.path(directory, "bad_images.idx"), file.path(directory, "bad_labels.idx")),
            "invalid IDX pair must fail")
        failures <<- failures + 1L
    }
    edit <- function(bytes, index, value) {
        bytes[index + 1L] <- as.raw(value)
        bytes
    }
    reject(edit(images, 3, 0), labels)                       # rank-0 image magic
    reject(images, edit(labels, 3, 0))                       # rank-0 label magic
    reject(images[-length(images)], labels)                  # truncated images
    reject(c(images, as.raw(0)), labels)                     # trailing image byte
    reject(images, labels[-length(labels)])                  # truncated labels
    reject(images, c(labels, as.raw(0)))                     # trailing label byte
    reject(edit(images, 7, 0), labels)                       # zero images
    reject(edit(images, 11, 0), labels)                      # zero rows
    reject(images, edit(labels, 7, 3))                       # label count mismatch
    reject(images, edit(labels, length(labels) - 1L, 10))    # label out of range
    overflow <- images
    overflow[5:16] <- as.raw(255)                            # dimension overflow
    reject(overflow, labels)
    reject(raw(0), labels)
    reject(images, raw(0))
    cat("  2 images, shape [2,2,3,1], paired shuffle and normalization passed\n")
    cat(sprintf("  %d malformed IDX pairs rejected\n", failures))
}

idx_loader_inspect <- function(directory, split) {
    images <- file.path(directory, paste0(split, "-images-idx3-ubyte"))
    labels <- file.path(directory, paste0(split, "-labels-idx1-ubyte"))
    dataset <- idx_loader_load_pair(images, labels)
    n <- dataset$labels$size(0L)
    counts <- tabulate(dataset$labels$values() + 1L, nbins = 10L)
    pixels <- nn$cast(dataset$images, nn$DType$Float64)
    low <- as.integer(nn$min(pixels)$item())
    high <- as.integer(nn$max(pixels)$item())
    cat(sprintf("  %s/%s: %d images, shape=%s, normalized mean=%g, byte range=[%d,%d]\n", basename(directory), split,
        as.integer(n), nn$shape_str(dataset$images$shape()), nn$mean(pixels)$item() / 255, low, high))
    cat("    class counts:", paste0(" ", counts, collapse = ""), "\n", sep = "")
    if (file.exists(paste0(images, ".gz")) && file.exists(paste0(labels, ".gz"))) {
        gz <- idx_loader_load_pair(paste0(images, ".gz"), paste0(labels, ".gz"))
        nn_require(nn$all(nn$equal(gz$images, dataset$images))$item() == 1 &&
            nn$all(nn$equal(gz$labels, dataset$labels))$item() == 1, "gzip IDX matches the decompressed file")
        cat("    .gz files decode to identical tensors\n")
    }
}

main <- function(arguments = commandArgs(TRUE)) {
    tryCatch({
        root <- nn_data_root()
        required <- ""
        split <- "t10k"
        for (arg in arguments) {
            at <- regexpr("=", arg, fixed = TRUE)
            key <- if (at > 0) substring(arg, 1L, at - 1L) else arg
            value <- if (at > 0) substring(arg, at + 1L) else ""
            if (!nzchar(value))
                stop("expected key=value", call. = FALSE)
            if (key == "root") {
                root <- value
            } else if (key == "dir") {
                required <- value
            } else if (key == "split" && value %in% c("train", "t10k")) {
                split <- value
            } else {
                stop("unknown IDX option: ", arg, call. = FALSE)
            }
        }
        cat("== IDX fixture and validation checks\n")
        idx_loader_fixtures()
        cat("\n== Real IDX datasets\n")
        if (nzchar(required)) {
            idx_loader_inspect(required, split)
        } else {
            for (name in c("mnist", "fashion_mnist")) {
                if (file.exists(file.path(root, name, paste0(split, "-images-idx3-ubyte"))))
                    idx_loader_inspect(file.path(root, name), split)
                else cat(sprintf("  %s not cached; fetch with scripts/fetch_datasets.py %s\n", name, name))
            }
        }
        cat("All IDX loader checks passed.\n")
        0L
    }, error = function(e) {
        message(conditionMessage(e))
        1L
    })
}

if (sys.nframe() == 0L) quit(status = main())
