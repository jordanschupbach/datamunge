# Differentiable dilation by inserting zeros between filter taps
# (examples/nn/common/dilated.hpp). nn$conv2d supports dilation natively; this
# expanded-filter path is the teaching/reference version: only the original
# taps carry gradient, inserted zeros are constants.

# Appends `after` zeros along a 0-based `axis` (nn$pad's paddings vector is not bound; concat is equivalent).
dilated_zero_pad_axis <- function(x, axis, after) {
    if (after <= 0) return(x)
    shape <- as.numeric(x$shape())
    shape[axis + 1L] <- after
    nn$concat(nn$tensor_vector(list(x, nn$zeros(shape, x$dtype()))), axis)
}

# [h, w, in, out] -> [(h-1)d+1, (w-1)d+1, in, out] with zeros between taps.
dilated_expand_dilated_kernel <- function(kernel, dilation) {
    if (kernel$dim() != 4L || kernel$numel() <= 0 || dilation < 1)
        stop("dilation requires a nonempty HWIO filter and positive rate")
    if (dilation == 1) return(kernel)
    s <- as.numeric(kernel$shape())
    d <- dilation
    taps <- nn$reshape(kernel, c(s[1L], 1, s[2L], 1, s[3L], s[4L]))
    padded <- nn$reshape(dilated_zero_pad_axis(dilated_zero_pad_axis(taps, 1L, d - 1), 3L, d - 1),
                         c(s[1L] * d, s[2L] * d, s[3L], s[4L]))
    nn$slice(padded, c(0, 0, 0, 0), c((s[1L] - 1) * d + 1, (s[2L] - 1) * d + 1, s[3L], s[4L]))
}

dilated_portable_dilated_conv2d <- function(x, kernel, dilation, padding = "SAME") {
    nn$conv2d(x, dilated_expand_dilated_kernel(kernel, dilation),
              nn$conv_options(1L, if (padding == "SAME") nn$Padding$Same else nn$Padding$Valid))
}
