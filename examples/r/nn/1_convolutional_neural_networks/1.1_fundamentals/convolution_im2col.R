# Native R NN example. Explicit patch geometry, packing and checks run in R.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT", getwd()), "examples/r/nn/common.R")
if (!file.exists(.nn_common))
    .nn_common <- system.file("examples/nn/common.R", package = "datamungerNN", mustWork = TRUE)
source(.nn_common, local = TRUE)
rm(.nn_common)

patch_geometry <- function(shape, kh, kw, sh = 1L, sw = 1L, dh = 1L, dw = 1L, padding = "VALID") {
    if (length(shape) != 4L || min(c(shape, kh, kw, sh, sw, dh, dw)) <= 0L || !padding %in% c("VALID", 
        "SAME")) 
        stop("invalid patch geometry")
    n <- shape[1L]
    h <- shape[2L]
    w <- shape[3L]
    channels <- shape[4L]
    eh <- (kh - 1L) * dh + 1L
    ew <- (kw - 1L) * dw + 1L
    oh <- if (padding == "SAME") 
        (h + sh - 1L)%/%sh
    else max(0L, (h - eh)%/%sh + 1L)
    ow <- if (padding == "SAME") 
        (w + sw - 1L)%/%sw
    else max(0L, (w - ew)%/%sw + 1L)
    if (oh == 0L || ow == 0L) 
        stop("patch geometry has no output positions")
    top <- if (padding == "SAME") 
        max(0L, (oh - 1L) * sh + eh - h)%/%2L
    else 0L
    left <- if (padding == "SAME") 
        max(0L, (ow - 1L) * sw + ew - w)%/%2L
    else 0L
    rows <- n * oh * ow
    columns <- kh * kw * channels
    column_ids <- pixel_ids <- integer()
    for (b in 0L:(n - 1L)) for (y in 0L:(oh - 1L)) for (x in 0L:(ow - 1L)) for (ky in 0L:(kh - 1L)) for (kx in 0L:(kw - 
        1L)) {
        iy <- y * sh + ky * dh - top
        ix <- x * sw + kx * dw - left
        if (iy >= 0L && iy < h && ix >= 0L && ix < w) 
            for (channel in 0L:(channels - 1L)) {
                column_ids <- c(column_ids, ((((b * oh + y) * ow + x) * kh + ky) * kw + kx) * channels + 
                  channel + 1L)
                pixel_ids <- c(pixel_ids, ((b * h + iy) * w + ix) * channels + channel + 1L)
            }
    }
    im2col <- function(images) {
        stopifnot(identical(as.numeric(images$shape()), as.numeric(shape)), images$dtype() == nn$DType$Float32)
        output <- numeric(rows * columns)
        output[column_ids] <- images$values()[pixel_ids]
        nn$from_vector(output, c(rows, columns))
    }
    col2im <- function(patches) {
        stopifnot(identical(as.numeric(patches$shape()), as.numeric(c(rows, columns))), patches$dtype() == 
            nn$DType$Float32)
        input <- patches$values()
        output <- numeric(prod(shape))
        for (i in seq_along(column_ids)) output[pixel_ids[i]] <- output[pixel_ids[i]] + input[column_ids[i]]
        nn$from_vector(output, shape)
    }
    list(rows = rows, columns = columns, output = c(n, oh, ow), im2col = im2col, col2im = col2im)
}

tensor_difference <- function(a, b) {
    stopifnot(identical(a$shape(), b$shape()))
    max(abs(a$values() - b$values()))
}

compare_convolution <- function(shape, kh, kw, outputs, sh, sw, dh, dw, padding, generator) {
    geometry <- patch_geometry(shape, kh, kw, sh, sw, dh, dw, padding)
    filter_shape <- c(kh, kw, shape[4L], outputs)
    output_shape <- c(geometry$output, outputs)
    x <- nn$normal(shape, 0, 0.3, generator)
    kernel <- nn$normal(filter_shape, 0, 0.3, generator)
    upstream <- nn$normal(output_shape, 0, 0.3, generator)
    columns <- geometry$im2col(x)
    matrix <- nn$reshape(kernel, c(geometry$columns, outputs))
    manual_of <- function(patches) nn$reshape(nn$matmul(patches, matrix), output_shape)
    manual <- manual_of(columns)
    options <- nn$Conv2DOptions()
    options$strides <- c(sh, sw)
    options$dilations <- c(dh, dw)
    options$padding <- if (padding == "SAME") 
        nn$Padding$Same
    else nn$Padding$Valid
    native <- nn$conv2d(x, kernel, options)
    dy <- nn$reshape(upstream, c(geometry$rows, outputs))
    dcolumns <- nn$matmul(dy, matrix, FALSE, TRUE)
    dx <- geometry$col2im(dcolumns)
    dw_manual <- nn$reshape(nn$matmul(columns, dy, TRUE, FALSE), filter_shape)
    errors <- c(tensor_difference(manual, native), tensor_difference(dx, nn$conv2d_backprop_input(shape, 
        kernel, upstream, options)), tensor_difference(dw_manual, nn$conv2d_backprop_filter(x, filter_shape, 
        upstream, options)))
    stopifnot(max(errors) < 2e-06)
    expanded_shape <- c((kh - 1L) * dh + 1L, (kw - 1L) * dw + 1L, shape[4L], outputs)
    expanded <- numeric(prod(expanded_shape))
    original <- kernel$values()
    indices <- integer()
    for (a in 0L:(kh - 1L)) for (b in 0L:(kw - 1L)) for (channel in 0L:(shape[4L] - 1L)) for (o in 0L:(outputs - 
        1L)) {
        source <- ((a * kw + b) * shape[4L] + channel) * outputs + o + 1L
        target <- ((a * dh * expanded_shape[2L] + b * dw) * shape[4L] + channel) * outputs + o + 1L
        expanded[target] <- original[source]
        indices <- c(indices, target)
    }
    expanded_kernel <- nn$from_vector(expanded, expanded_shape)
    plain <- nn$Conv2DOptions()
    plain$strides <- c(sh, sw)
    plain$padding <- options$padding
    stopifnot(tensor_difference(dx, nn$conv2d_backprop_input(shape, expanded_kernel, upstream, plain)) < 
        2e-06)
    expanded_dw <- nn$conv2d_backprop_filter(x, expanded_shape, upstream, plain)$values()
    stopifnot(tensor_difference(dw_manual, nn$from_vector(expanded_dw[indices], filter_shape)) < 2e-06)
    stopifnot(abs(sum(columns$values() * dcolumns$values()) - sum(x$values() * dx$values())) < 2e-06)
    finite_error <- 0
    for (index in c(1L, x$numel()%/%2L + 1L, x$numel())) {
        values <- numeric()
        for (delta in c(-0.002, 0.002)) {
            changed <- x$values()
            changed[index] <- changed[index] + delta
            output <- manual_of(geometry$im2col(nn$from_vector(changed, shape)))
            values <- c(values, sum(output$values() * upstream$values()))
        }
        finite_error <- max(finite_error, abs((values[2L] - values[1L])/0.004 - dx$values()[index]))
    }
    stopifnot(finite_error < 3e-05)
    cat(padding, paste(shape, collapse = ","), ": errors(y,dx,dw,finite)=", paste(c(errors, finite_error), 
        collapse = ","), "\n")
}

main <- function(arguments = commandArgs(TRUE)) {
    generator <- nn$Generator(149L)
    cases <- list(list(c(2L, 5L, 6L, 2L), 3L, 2L, 3L, 1L, 1L, 1L, 1L, "VALID"), list(c(2L, 5L, 6L, 2L), 
        3L, 2L, 3L, 2L, 2L, 1L, 1L, "SAME"), list(c(1L, 6L, 7L, 3L), 2L, 3L, 2L, 2L, 1L, 1L, 1L, "SAME"), 
        list(c(1L, 7L, 8L, 2L), 3L, 2L, 3L, 1L, 1L, 2L, 2L, "VALID"), list(c(1L, 7L, 8L, 2L), 3L, 2L, 
            3L, 1L, 1L, 2L, 2L, "SAME"), list(c(2L, 4L, 5L, 3L), 1L, 1L, 4L, 1L, 1L, 1L, 1L, "VALID"))
    for (case in cases) do.call(compare_convolution, c(case, list(generator)))
    geometry <- patch_geometry(c(1L, 5L, 5L, 1L), 3L, 3L)
    coverage <- geometry$col2im(geometry$im2col(nn$ones(c(1L, 5L, 5L, 1L))))
    stopifnot(coverage$at(c(0L, 0L, 0L, 0L)) == 1, coverage$at(c(0L, 2L, 2L, 0L)) == 9)
    rejected <- tryCatch({
        patch_geometry(c(1L, 5L, 5L, 1L), 0L, 3L)
        FALSE
    }, error = function(e) TRUE)
    stopifnot(rejected)
    cat("All im2col convolution checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())

