# Explicit NHWC image-to-column extraction and its additive adjoint
# (examples/nn/common/im2col.hpp). Each patch-matrix row is one output location;
# columns order kernel row, kernel column, input channel. SAME puts an odd extra
# pad at the bottom/right, as TensorFlow does. Cross-correlation, rectangular
# kernels, separate strides and dilation are supported.

# im2col::PatchGeometry as a list (n, h, w, c, kh, kw, sh, sw, dh, dw, oh, ow, top, left, rows, columns).
im2col_patch_geometry <- function(input_shape, kernel_h, kernel_w, stride_h = 1L, stride_w = 1L,
                                  dilation_h = 1L, dilation_w = 1L, padding = "VALID") {
    input_shape <- as.numeric(input_shape)
    if (length(input_shape) != 4L || prod(input_shape) < 1 ||
        min(kernel_h, kernel_w, stride_h, stride_w, dilation_h, dilation_w) <= 0 ||
        !padding %in% c("VALID", "SAME"))
        stop("invalid patch geometry")
    g <- list(n = input_shape[1L], h = input_shape[2L], w = input_shape[3L], c = input_shape[4L],
              kh = kernel_h, kw = kernel_w, sh = stride_h, sw = stride_w, dh = dilation_h, dw = dilation_w)
    effective_h <- (g$kh - 1) * g$dh + 1
    effective_w <- (g$kw - 1) * g$dw + 1
    if (padding == "SAME") {
        g$oh <- (g$h + g$sh - 1) %/% g$sh
        g$ow <- (g$w + g$sw - 1) %/% g$sw
    } else {
        g$oh <- if (g$h >= effective_h) (g$h - effective_h) %/% g$sh + 1 else 0
        g$ow <- if (g$w >= effective_w) (g$w - effective_w) %/% g$sw + 1 else 0
    }
    if (g$oh == 0 || g$ow == 0)
        stop("patch geometry has no output positions")
    g$top <- if (padding == "SAME") max(0, (g$oh - 1) * g$sh + effective_h - g$h) %/% 2 else 0
    g$left <- if (padding == "SAME") max(0, (g$ow - 1) * g$sw + effective_w - g$w) %/% 2 else 0
    g$rows <- g$n * g$oh * g$ow
    g$columns <- g$kh * g$kw * g$c
    g$input_shape <- c(g$n, g$h, g$w, g$c)
    g
}

# 1-based (patch-matrix index, input index) pairs of every in-bounds patch entry.
im2col_pairs <- function(g) {
    grid <- expand.grid(c = seq_len(g$c) - 1, kx = seq_len(g$kw) - 1, ky = seq_len(g$kh) - 1,
                        xo = seq_len(g$ow) - 1, y = seq_len(g$oh) - 1, n = seq_len(g$n) - 1)
    iy <- grid$y * g$sh + grid$ky * g$dh - g$top
    ix <- grid$xo * g$sw + grid$kx * g$dw - g$left
    keep <- iy >= 0 & iy < g$h & ix >= 0 & ix < g$w
    grid <- grid[keep, , drop = FALSE]
    column <- ((grid$n * g$oh + grid$y) * g$ow + grid$xo) * g$columns + (grid$ky * g$kw + grid$kx) * g$c + grid$c
    pixel <- ((grid$n * g$h + iy[keep]) * g$w + ix[keep]) * g$c + grid$c
    list(column = column + 1, pixel = pixel + 1)
}

# [N,H,W,C] float32 -> [rows, columns] patch matrix.
im2col_im2col <- function(images, g) {
    if (images$dtype() != nn$DType$Float32 || !identical(as.numeric(images$shape()), g$input_shape))
        stop("Im2Col requires matching float32 NHWC input")
    pairs <- im2col_pairs(g)
    output <- numeric(g$rows * g$columns)
    output[pairs$column] <- images$contiguous()$values()[pairs$pixel]
    nn$from_vector(output, c(g$rows, g$columns))
}

# Adjoint of im2col: scatter-adds each patch entry back to its pixel. R sums
# in double precision and rounds once to float32; the C++ accumulates in float32.
im2col_col2im <- function(columns, g) {
    if (columns$dtype() != nn$DType$Float32 || !identical(as.numeric(columns$shape()), c(g$rows, g$columns)))
        stop("Col2Im requires matching float32 patch matrix")
    pairs <- im2col_pairs(g)
    source <- columns$contiguous()$values()
    output <- numeric(prod(g$input_shape))
    sums <- rowsum(source[pairs$column], pairs$pixel, reorder = TRUE)
    output[as.numeric(rownames(sums))] <- sums[, 1L]
    nn$from_vector(output, g$input_shape)
}
