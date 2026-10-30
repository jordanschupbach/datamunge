# Native R NN example. Image-folder datasets with PNG/JPEG decoding and bilinear resize.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.4_data_loading_and_preprocessing/image_folder_loader.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
# Port of tflow's image_folder_loader. Generate small RGB PNG/JPEG and
# grayscale PNG fixtures. Scan class directories in deterministic order,
# decode mixed image formats to RGB, and assemble a shuffled float32 batch
# with aligned labels. Compare every resized PNG pixel with an independent
# half-pixel bilinear calculation and bound JPEG's lossy error. Check
# grayscale expansion, stable labels when a validation class is absent, and
# explicit corrupt-image errors. Optional root=PATH scans an existing folder
# (one subdirectory per class) and decodes its first four images to 32x32 RGB.
#
# Decoding uses nn$load_image_rgb (datamunge::image PNG reader and JPEG
# decoder, bit-exact with TF's DecodeJpeg); nn$scan_image_folder validates the
# root and class names. The JPEG fixture is TF's EncodeJpeg(quality=100,
# chroma_downsampling=false) byte stream for a 4x4 solid image (datamunge has
# no JPEG encoder). Labels are int64. The SWIG layer exposes
# ImageFolder$images (std::vector<ImageFile>) only as an opaque pointer and has
# no size_t row vector for load_image_batch, so image_folder_files lists the
# class folders with scan_image_folder's documented rules and
# image_folder_load_batch stacks nn$load_image_rgb results. PNG fixtures are
# written with nn$save_png from byte-valued tensors (low=0, high=255, exact).
# Fixtures go to output/nn/r/<mode>/fixtures/images{,_validation}.
nn_helper("checks")

# tf.io.encode_jpeg(4x4 RGB (64, 128, 192), quality=100, chroma_downsampling=False), TF 2.20.
image_folder_jpeg_hex <- paste0(
    "ffd8ffe000104a46494600010101012c012c0000ffdb0043000101010101010101010101010101010101010101010101",
    "0101010101010101010101010101010101010101010101010101010101010101010101010101010101ffdb0043010101",
    "010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101010101",
    "0101010101010101010101010101ffc00011080004000403011100021101031101ffc4001f0000010501010101010100",
    "000000000000000102030405060708090a0bffc400b5100002010303020403050504040000017d010203000411051221",
    "31410613516107227114328191a1082342b1c11552d1f02433627282090a161718191a25262728292a3435363738393a",
    "434445464748494a535455565758595a636465666768696a737475767778797a838485868788898a9293949596979899",
    "9aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4c5c6c7c8c9cad2d3d4d5d6d7d8d9dae1e2e3e4e5e6e7e8e9eaf1",
    "f2f3f4f5f6f7f8f9faffc4001f0100030101010101010101010000000000000102030405060708090a0bffc400b51100",
    "020102040403040705040400010277000102031104052131061241510761711322328108144291a1b1c109233352f015",
    "6272d10a162434e125f11718191a262728292a35363738393a434445464748494a535455565758595a63646566676869",
    "6a737475767778797a82838485868788898a92939495969798999aa2a3a4a5a6a7a8a9aab2b3b4b5b6b7b8b9bac2c3c4",
    "c5c6c7c8c9cad2d3d4d5d6d7d8d9dae2e3e4e5e6e7e8e9eaf2f3f4f5f6f7f8f9faffda000c03010002110311003f00f1",
    "faff00560ff35cffd9")
image_folder_jpeg <- as.raw(strtoi(substring(image_folder_jpeg_hex, seq(1L, nchar(image_folder_jpeg_hex), 2L),
    seq(2L, nchar(image_folder_jpeg_hex), 2L)), 16L))

# std::filesystem::path::extension, lower-cased ("" for dot files like ".png").
image_folder_extension <- function(path) {
    name <- basename(path)
    at <- regexpr("\\.[^.]*$", name)
    if (at <= 1L) "" else tolower(substring(name, at))
}

# list(class_names, files = data.frame(path, label)) in nn::scan_image_folder order;
# the scan itself validates the root and class names and supplies the label order.
image_folder_files <- function(root, class_names = character()) {
    folder <- nn$scan_image_folder(root, class_names)
    names <- folder$class_names
    paths <- character()
    labels <- numeric()
    for (label in seq_along(names) - 1L) {
        directory <- file.path(root, names[label + 1L])
        if (!dir.exists(directory)) next  # a class absent from this split keeps its label
        entries <- list.files(directory, all.files = TRUE, no.. = TRUE, full.names = TRUE)
        entries <- entries[file_test("-f", entries) &
            vapply(entries, image_folder_extension, "") %in% c(".png", ".jpg", ".jpeg")]
        entries <- sort(entries, method = "radix")
        paths <- c(paths, entries)
        labels <- c(labels, rep(label, length(entries)))
    }
    list(class_names = names, files = data.frame(path = paths, label = labels, stringsAsFactors = FALSE))
}

# images [rows, height, width, 3] float32 in [0, 1] and int64 labels; rows are 0-based.
image_folder_load_batch <- function(files, rows, height, width) {
    if (any(rows >= nrow(files)))
        stop("load_image_batch: row index out of range", call. = FALSE)
    decoded <- lapply(rows, function(row) nn$load_image_rgb(files$path[row + 1L], height, width))
    list(images = nn$stack(nn$tensor_vector(decoded), 0L),
        labels = nn$from_vector_int64(files$label[rows + 1L], length(rows)))
}

image_folder_fixtures <- function() {
    root <- nn_output_dir("fixtures/images")
    validation <- nn_output_dir("fixtures/images_validation")
    for (directory in file.path(c(root, root, validation), c("ant", "zebra", "zebra")))
        dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    rgb <- as.numeric(outer(0:2 * 30, 0:3 * 40, "+"))  # pixel * 40 + c * 30, channel fastest
    nn$save_png(nn$from_vector(rgb, c(2, 2, 3)), file.path(root, "ant", "pattern.PNG"), 0, 255)
    nn$save_png(nn$full(c(2, 2, 1), 77), file.path(root, "zebra", "gray.png"), 0, 255)
    writeBin(image_folder_jpeg, file.path(root, "zebra", "solid.jpg"))
    writeBin(image_folder_jpeg, file.path(validation, "zebra", "solid.jpg"))
    writeBin(charToRaw("ignore"), file.path(root, "ant", "notes.txt"))

    scan <- image_folder_files(root)
    nn_require(identical(scan$class_names, c("ant", "zebra")) && nrow(scan$files) == 3L,
        "sorted classes and supported files")
    nn_require(identical(scan$files$label, c(0, 1, 1)), "deterministic class labels")
    held_out <- image_folder_files(validation, scan$class_names)
    nn_require(nrow(held_out$files) == 1L && held_out$files$label[1] == 1,
        "missing validation class preserves training labels")

    batch <- image_folder_load_batch(scan$files, c(2, 0, 1), 4, 4)
    nn_require(identical(as.numeric(batch$images$shape()), c(3, 4, 4, 3)), "resized NHWC batch")
    nn_require(identical(batch$labels$values(), c(1, 0, 1)), "batch label alignment")
    pixels <- batch$images$values()
    at <- function(n, y, x, c) pixels[((n * 4 + y) * 4 + x) * 3 + c + 1]
    png_error <- jpeg_error <- 0
    for (y in 0:3) for (x in 0:3) for (c in 0:2) {
        sy <- min(max((y + 0.5) / 2 - 0.5, 0), 1)
        sx <- min(max((x + 0.5) / 2 - 0.5, 0), 1)
        y0 <- floor(sy)
        x0 <- floor(sx)
        y1 <- min(y0 + 1, 1)
        x1 <- min(x0 + 1, 1)
        wy <- sy - y0
        wx <- sx - x0
        value <- function(row, col) rgb[(row * 2 + col) * 3 + c + 1] / 255
        expected <- (1 - wy) * ((1 - wx) * value(y0, x0) + wx * value(y0, x1)) +
            wy * ((1 - wx) * value(y1, x0) + wx * value(y1, x1))
        png_error <- max(png_error, abs(at(1, y, x, c) - expected))
        jpeg_error <- max(jpeg_error, abs(at(0, y, x, c) - (64 + c * 64) / 255))
        nn_near(at(1, y, x, c), expected, 1e-7, "bilinear half-pixel coordinates")
        nn_near(at(0, y, x, c), (64 + c * 64) / 255, 2 / 255, "JPEG lossy error")
        nn_near(at(2, y, x, c), 77 / 255, 1e-7, "grayscale expands to RGB")
    }
    nn_throws(nn$scan_image_folder(root, "ant"), "unknown validation class must fail")
    # Put corrupt input outside class folders so subsequent scans stay stable.
    corrupt <- file.path(root, "corrupt.png")
    writeBin(as.raw(1:4), corrupt)
    nn_throws(nn$load_image_rgb(corrupt, 4, 4), "corrupt PNG must fail")
    # Extra (not in tflow): a truncated JPEG must fail too, and the decoded fixture
    # must equal TF's DecodeJpeg output (64, 128, 192 at quality 100: lossless here).
    truncated <- file.path(root, "truncated.jpg")
    writeBin(image_folder_jpeg[seq_len(length(image_folder_jpeg) %/% 2L)], truncated)
    nn_throws(nn$load_image_rgb(truncated, 4, 4), "truncated JPEG must fail")
    decoded <- nn$load_image_rgb(file.path(root, "zebra", "solid.jpg"), 4, 4)$values()
    nn_require(all(round(decoded * 255) == 64 + (seq_along(decoded) - 1) %% 3 * 64),
        "JPEG fixture decodes to TF's pixels")
    cat("  classes=(ant,zebra), 3 images, shuffled batch [3,4,4,3]\n")
    cat(sprintf("  max PNG resize error=%g, max JPEG error=%g\n", png_error, jpeg_error))
    cat("  grayscale RGB expansion, stable validation labels and corrupt-file rejection passed\n")
}

main <- function(arguments = commandArgs(TRUE)) {
    tryCatch({
        root <- ""
        for (arg in arguments) {
            if (!startsWith(arg, "root=") || nchar(arg) == 5L)
                stop("expected root=PATH", call. = FALSE)
            root <- substring(arg, 6L)
        }
        cat("== Image folder and PNG/JPEG preprocessing\n")
        image_folder_fixtures()
        if (nzchar(root)) {
            scan <- image_folder_files(root)
            batch <- image_folder_load_batch(scan$files, seq_len(min(4L, nrow(scan$files))) - 1L, 32, 32)
            shape <- as.integer(batch$images$shape())
            cat(sprintf("  requested folder: %d classes, %d files; first batch=[%d,%d,%d,%d]\n",
                length(scan$class_names), nrow(scan$files), shape[1], shape[2], shape[3], shape[4]))
        }
        cat("All image-folder checks passed.\n")
        0L
    }, error = function(e) {
        message(conditionMessage(e))
        1L
    })
}

if (sys.nframe() == 0L) quit(status = main())
