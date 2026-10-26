# Native R NN example. CIFAR-10/CIFAR-100 binary decoding, planar-to-NHWC conversion and labels.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
# CMD: just r-example cifar_loader
# Port of tflow's cifar_loader. Write original-format records with
# deterministic pixel patterns, read them from disk, and check every output
# coordinate. CIFAR-100 exposes both fine and coarse labels. Multiple CIFAR-10
# batch files concatenate in caller order. Normalize the pixels with nn ops
# and verify channel means. Reject empty, truncated and trailing records,
# invalid labels and bad formats.
#
# Optional format=10|100 and repeated file=PATH arguments load real extracted
# binary batches ($DATAMUNGE_DATA_DIR/cifar10/data_batch_*.bin). A single
# CIFAR-10 test_batch.bin is also compared with nn$load_cifar10 (float32 / 255).
# Pixels are int32 holding 0..255 (nn has no uint8 dtype). Fixtures are
# written under output/nn/r/<mode>/fixtures/ (nn_output_dir).
nn_helper("checks")
nn_helper("binary")
nn_helper("cifar")

cifar_loader_record <- function(kind, index) {
    labels <- if (kind == "k100") c(4L + index, 73L + index) else 3L + index
    pixels <- unlist(lapply(0:2, function(channel) (index * 17L + channel * 61L + 0:1023) %% 256L))
    as.raw(c(labels, pixels))
}

cifar_loader_fixtures <- function(kind) {
    a <- cifar_loader_record(kind, 0L)
    b <- cifar_loader_record(kind, 1L)
    directory <- nn_output_dir(if (kind == "k10") "fixtures/cifar10" else "fixtures/cifar100")
    binary_write(file.path(directory, "batch1.bin"), a)
    binary_write(file.path(directory, "batch2.bin"), b)
    dataset <- cifar_load(file.path(directory, c("batch1.bin", "batch2.bin")), kind)
    nn_require(identical(as.numeric(dataset$images$shape()), c(2, 32, 32, 3)), "CIFAR NHWC shape")
    images <- dataset$images$values()
    labels <- dataset$labels$values()
    for (row in 0:1) {
        nn_require(labels[row + 1L] == (if (kind == "k100") 73 else 3) + row, "fine label/order")
        if (kind == "k100")
            nn_require(dataset$coarse_labels$values()[row + 1L] == 4 + row, "coarse label/order")
        # NHWC index row*3072 + pixel*3 + channel holds (row*17 + channel*61 + pixel) %% 256.
        pixel <- rep(0:1023, each = 3L)
        channel <- rep(0:2, times = 1024L)
        expected <- (row * 17 + channel * 61 + pixel) %% 256
        nn_require(all(images[row * 3072 + seq_len(3072L)] == expected), "channel-major bytes to NHWC")
    }
    normalized <- nn$div(nn$cast(dataset$images, nn$DType$Float32), 255)
    means <- nn$mean(normalized, c(0L, 1L, 2L))$values()
    for (i in 1:3) nn_near(means[i], 0.5, 2e-6, "channel mean")
    failures <- 0L
    reject <- function(bytes) {
        rejected <- tryCatch({
            cifar_parse(list(bytes), kind)
            FALSE
        }, error = function(e) TRUE)
        nn_require(rejected, "invalid CIFAR records must fail")
        failures <<- failures + 1L
    }
    reject(raw(0))
    reject(a[-length(a)])
    reject(c(a, as.raw(0)))
    bad <- a
    bad[1] <- as.raw(if (kind == "k100") 20 else 10)
    reject(bad)
    if (kind == "k100") {
        bad <- a
        bad[2] <- as.raw(100)
        reject(bad)
    }
    cat(sprintf("  CIFAR-%d: 2 files -> [2,32,32,3], all 6144 pixels and labels verified\n",
        if (kind == "k10") 10L else 100L))
    cat(sprintf("    normalized channel means=(0.5,0.5,0.5), %d malformed records rejected\n", failures))
}

main <- function(arguments = commandArgs(TRUE)) {
    tryCatch({
        nn$set_num_threads(2L)
        kind <- "k10"
        files <- character()
        for (arg in arguments) {
            at <- regexpr("=", arg, fixed = TRUE)
            key <- if (at > 0) substring(arg, 1L, at - 1L) else arg
            value <- if (at > 0) substring(arg, at + 1L) else ""
            if (key == "file" && nzchar(value)) {
                files <- c(files, value)
            } else if (key == "format" && value %in% c("10", "100")) {
                kind <- if (value == "10") "k10" else "k100"
            } else {
                stop("expected format=10|100 or file=PATH", call. = FALSE)
            }
        }
        cat("== CIFAR binary loader checks\n")
        cifar_loader_fixtures("k10")
        cifar_loader_fixtures("k100")
        if (length(files)) {
            dataset <- cifar_load(files, kind)
            classes <- if (kind == "k10") 10L else 100L
            histogram <- tabulate(dataset$labels$values() + 1L, nbins = classes)
            cat(sprintf("  loaded %d real file(s), shape=[%d,32,32,3]\n", length(files), dataset$images$size(0L)))
            cat("    fine-label counts:", paste0(" ", histogram, collapse = ""), "\n", sep = "")
            if (kind == "k10" && length(files) == 1L && basename(files[1]) == "test_batch.bin") {
                library_data <- nn$load_cifar10(dirname(files[1]), "test")
                ours <- nn$div(nn$cast(dataset$images, nn$DType$Float32), 255)
                nn_require(nn$all(nn$equal(ours, library_data$images))$item() == 1 &&
                    identical(library_data$labels$values(), dataset$labels$values()),
                    "nn::load_cifar10 agrees with the byte decoder")
                cat("    identical to nn::load_cifar10 (pixels / 255, labels)\n")
            }
        }
        cat("All CIFAR loader checks passed.\n")
        0L
    }, error = function(e) {
        message(conditionMessage(e))
        1L
    })
}

if (sys.nframe() == 0L) quit(status = main())
