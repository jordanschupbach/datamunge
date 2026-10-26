# CIFAR-10/CIFAR-100 binary records to NHWC nn tensors (examples/nn/common/cifar.hpp).
# CIFAR-10 records have one label byte followed by 1024 red, green and blue
# values. CIFAR-100 records begin with coarse then fine labels; both are kept
# and the fine label is exposed as labels. File order determines example
# order. Incomplete/empty records and out-of-range labels are rejected before
# allocating. Pixels are int32 holding 0..255 and labels int32.
nn_helper("binary")

cifar_kinds <- c("k10", "k100")

# A dataset: list(images = int32 [N,32,32,3], labels = int32 [N],
# coarse_labels = int32 [N] for CIFAR-100, else NULL).
cifar_size <- function(dataset) if (is.null(dataset$labels)) 0 else dataset$labels$numel()

# files: a list of raw vectors; kind: "k10" or "k100".
cifar_parse <- function(files, kind) {
    if (!length(files) || length(kind) != 1L || !kind %in% cifar_kinds)
        stop("need CIFAR files and valid format", call. = FALSE)
    labels <- if (kind == "k100") 2L else 1L
    record <- 3072L + labels
    count <- 0
    for (bytes in files) {
        if (!length(bytes) || length(bytes) %% record)
            stop("incomplete or empty CIFAR records", call. = FALSE)
        starts <- seq.int(1L, length(bytes), by = record)
        count <- count + length(starts)
        if (max(as.integer(bytes[starts])) >= (if (labels == 2L) 20L else 10L) ||
            (labels == 2L && max(as.integer(bytes[starts + 1L])) >= 100L))
            stop("CIFAR label out of range", call. = FALSE)
    }
    if (binary_product(count, 3072) > .Machine$integer.max)
        stop("CIFAR tensor too large", call. = FALSE)
    # Records [N, labels + 3072] -> planar [N, 3, 1024] -> NHWC [N, 32, 32, 3].
    records <- binary_to_tensor(do.call(c, files), c(count, record))
    planar <- nn$reshape(nn$slice_axis(records, 1L, labels, record), c(count, 3, 1024))
    result <- list(
        images = nn$reshape(nn$transpose(planar, c(0L, 2L, 1L))$contiguous(), c(count, 32, 32, 3)),
        labels = nn$reshape(nn$slice_axis(records, 1L, labels - 1L, labels)$contiguous(), c(count)),
        coarse_labels = NULL)
    if (labels == 2L)
        result$coarse_labels <- nn$reshape(nn$slice_axis(records, 1L, 0L, 1L)$contiguous(), c(count))
    result
}

cifar_load <- function(paths, kind) cifar_parse(lapply(paths, binary_read), kind)
