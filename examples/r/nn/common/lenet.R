# LeNet-5 with the sparse C3 connectivity and learned subsampling of LeCun et
# al. 1998 (examples/nn/common/lenet.hpp; parameter names match tflow's
# common/nn/lenet.h). Six trainable stages hold exactly 60,000 parameters; C3
# stores only its 60 connected 5x5 kernels; every stage uses 1.7159*tanh(2x/3).
# [0,1] pixels are padded to 32x32 and mapped to [-0.1, 1.175]. Fixed Euclidean
# RBF outputs use 7x12 seven-segment digit codes.

lenet_c3_connections <- function() list(
    c(0, 1, 2), c(1, 2, 3), c(2, 3, 4), c(3, 4, 5), c(4, 5, 0), c(5, 0, 1), c(0, 1, 2, 3), c(1, 2, 3, 4),
    c(2, 3, 4, 5), c(3, 4, 5, 0), c(4, 5, 0, 1), c(5, 0, 1, 2), c(0, 1, 3, 4), c(1, 2, 4, 5), c(0, 2, 3, 5),
    c(0, 1, 2, 3, 4, 5))

lenet_tanh <- function(x) nn$mul(nn$tanh(nn$mul(x, 2 / 3)), 1.7159)

lenet_subsample <- function(parameters, name, x) {
    shape <- as.numeric(x$shape())
    channels <- shape[length(shape)]
    total <- nn$mul(nn$avg_pool2d(x, 2L, 2L), 4)
    scale <- parameters$get(paste0(name, "/scale"), channels, nn$init$constant(0.25))
    bias <- parameters$get(paste0(name, "/bias"), channels, nn$init$zeros())
    lenet_tanh(nn$add(nn$mul(total, scale), bias))
}

lenet_c3 <- function(parameters, name, x) {
    options <- nn$ConvLayerOptions()
    options$padding <- nn$Padding$Valid
    connections <- lenet_c3_connections()
    maps <- vector("list", length(connections))
    for (i in seq_along(connections)) {
        channels <- connections[[i]]
        selected <- nn$gather(x, nn$from_vector_int64(channels, length(channels)), 3L)
        maps[[i]] <- nn$conv2d(parameters, paste0(name, "/map", i - 1L), selected, 1L, 5L, options)
    }
    lenet_tanh(nn$concat(nn$tensor_vector(maps), 3L))
}

# [10, 84] seven-segment codes in {-1, 1} on a 7x12 grid. Segment order: top,
# upper right, lower right, bottom, lower left, upper left, middle.
lenet_digit_codes <- function() {
    codes <- rep(-1, 10L * 84L)
    segments <- c("abcdef", "bc", "abdeg", "abcdg", "bcfg", "acdfg", "acdefg", "abc", "abcdefg", "abcdfg")
    pixel <- function(digit, y, x) digit * 84 + y * 7 + x + 1
    for (digit in 0:9) for (segment in strsplit(segments[digit + 1L], "")[[1L]]) {
        if (segment %in% c("a", "d", "g")) {
            y <- switch(segment, a = 1, d = 10, g = 5)
            codes[pixel(digit, y, 1:5)] <- 1
        } else {
            upper <- segment %in% c("b", "f")
            x <- if (segment %in% c("b", "c")) 5 else 1
            ys <- if (upper) 2:4 else 6:9
            codes[pixel(digit, ys, x)] <- 1
        }
    }
    nn$from_vector(codes, c(10L, 84L))
}

# Negative squared Euclidean distances to the fixed digit codes: [N, 84] -> [N, 10].
lenet_rbf_logits <- function(features) {
    difference <- nn$sub(nn$expand_dims(features, 1L), lenet_digit_codes())
    nn$neg(nn$sum(nn$square(difference), 2L))
}

# nn::pad(x, {{0,0},{a,a},{a,a},{0,0}}) composed from concat (nn$pad's paddings vector is not bound).
lenet_zero_pad_spatial <- function(x, amount) {
    s <- as.numeric(x$shape())
    rows <- nn$zeros(c(s[1L], amount, s[3L], s[4L]), x$dtype())
    x <- nn$concat(nn$tensor_vector(list(rows, x, rows)), 1L)
    columns <- nn$zeros(c(s[1L], s[2L] + 2 * amount, amount, s[4L]), x$dtype())
    nn$concat(nn$tensor_vector(list(columns, x, columns)), 2L)
}

# Flat [N, 784] pixels in [0, 1] -> every stage of LeNet-5 as a list
# (image, c1, s2, c3, s4, c5, f6, logits).
lenet_lenet5 <- function(parameters, flat_pixels) {
    out <- list()
    padded <- lenet_zero_pad_spatial(nn$reshape(flat_pixels, c(-1L, 28L, 28L, 1L)), 2)
    out$image <- nn$sub(nn$mul(padded, 1.275), 0.1)
    options <- nn$ConvLayerOptions()
    options$padding <- nn$Padding$Valid
    out$c1 <- lenet_tanh(nn$conv2d(parameters, "c1", out$image, 6L, 5L, options))
    out$s2 <- lenet_subsample(parameters, "s2", out$c1)
    out$c3 <- lenet_c3(parameters, "c3", out$s2)
    out$s4 <- lenet_subsample(parameters, "s4", out$c3)
    out$c5 <- lenet_tanh(nn$conv2d(parameters, "c5", out$s4, 120L, 5L, options))
    out$f6 <- lenet_tanh(nn$dense(parameters, "f6", nn$reshape(out$c5, c(-1L, 120L)), 84L))
    out$logits <- lenet_rbf_logits(out$f6)
    out
}
