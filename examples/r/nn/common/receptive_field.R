# Receptive-field geometry for a chain of spatial layers, one axis at a time
# (examples/nn/common/receptive_field.hpp). Input pixel centers are .5, 1.5, ...;
# jump is the spacing between outputs in input coordinates. A kernel's effective
# width is dilation*(kernel-1)+1. Padding shifts the first center. Bounds are
# the enclosing half-open interval, including holes for dilation.

receptive_field_new <- function(width = 1L, jump = 1L, start = 0.5)
    list(width = width, jump = jump, start = start)

receptive_field_after <- function(field, kernel, stride, padding_before, dilation = 1L) {
    if (kernel <= 0 || stride <= 0 || padding_before < 0 || dilation <= 0)
        stop("invalid receptive-field layer geometry")
    effective <- dilation * (kernel - 1) + 1
    receptive_field_new(field$width + (effective - 1) * field$jump, field$jump * stride,
                        field$start + ((effective - 1) / 2 - padding_before) * field$jump)
}

# std::lround: halfway cases away from zero.
receptive_field_lround <- function(value) sign(value) * floor(abs(value) + 0.5)

# Half-open integer pixel bounds c(lo, hi), before clipping to the image.
receptive_field_bounds <- function(field, coordinate) {
    center <- field$start + coordinate * field$jump
    c(receptive_field_lround(center - (field$width - 1) / 2 - 0.5),
      receptive_field_lround(center + (field$width - 1) / 2 - 0.5) + 1)
}
