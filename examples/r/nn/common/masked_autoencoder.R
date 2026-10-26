# Tiny asymmetric masked autoencoder (examples/nn/common/masked_autoencoder.hpp).
#
# He et al. MAE on 16 non-overlapping 7x7 MNIST patches: a visible-only width-32
# transformer encoder (4 of 16 tokens), a width-16 mask-token decoder, fixed 2-D
# sin/cos positions and a hidden-only loss. Parameter names match tflow's. Model
# pieces, hash-derived patch selections and native metrics only: training steps
# (forward, masked_autoencoder_loss, nn$grad / optimizer) are written out in each example.

if (!exists("autoencoder_mul32", inherits = TRUE)) nn_helper("autoencoder", environment())

# Fixed 2-D sin/cos positions [16, width]: first half the patch row, second the column.
masked_autoencoder_position <- function(width) {
    width <- as.integer(width)
    out <- numeric(16L * width)
    for (patch in 0L:15L)
        for (axis in 0L:1L)
            for (k in seq_len(width %/% 4L) - 1L) {
                a <- (if (axis == 1L) patch %% 4L else patch %/% 4L) * 10000^(-2 * k / (width %/% 2L))
                out[patch * width + axis * width %/% 2L + k + 1L] <- sin(a)
                out[patch * width + axis * width %/% 2L + width %/% 4L + k + 1L] <- cos(a)
            }
    nn$from_vector(out, c(16L, width))
}

# Pre-norm block: x + MHA(LN x) (2 heads, no biases), then + FF(LN .) ratio 2, tanh-GELU.
masked_autoencoder_block <- function(p, name, x, width) {
    normalized <- nn$layer_norm(p, paste0(name, "/ln1"), x)
    options <- nn$MultiHeadOptions()
    options$need_weights <- TRUE
    attended <- nn$multi_head_attention(p, paste0(name, "/attention"), normalized, normalized, 2L, options)
    residual <- nn$add(x, attended$output)
    ff <- nn$dense(p, paste0(name, "/ff1"), nn$layer_norm(p, paste0(name, "/ln2"), residual), 2L * as.integer(width),
                   nn$Activation$GeluTanh)
    list(output = nn$add(residual, nn$dense(p, paste0(name, "/ff2"), ff, as.integer(width))),
         attention = attended$weights)
}

# [B, 28, 28, 1] -> [B, 16, 49] row-major 7x7 patches.
masked_autoencoder_patches <- function(image) {
    b <- image$size(0L)
    nn$reshape(nn$transpose(nn$reshape(image, c(b, 4L, 7L, 4L, 7L)), c(0L, 1L, 3L, 2L, 4L)), c(b, 16L, 49L))
}

# Encoder half; selector [B, 4, 16] one-hot gather: hidden pixels never enter the encoder.
# Adds patches, visible, mask, code, encoder_attention to the list `out`.
masked_autoencoder_encode <- function(p, image, selector, out = list()) {
    out$patches <- masked_autoencoder_patches(image)
    out$visible <- nn$matmul(selector, out$patches)
    out$mask <- nn$sub(1, nn$sum(selector, 1L))
    embedded <- nn$dense(p, "patch", out$visible, 32L)
    position <- nn$matmul(selector, nn$expand_dims(masked_autoencoder_position(32L), 0L))
    encoder <- masked_autoencoder_block(p, "encoder", nn$add(embedded, position), 32L)
    out$code <- nn$layer_norm(p, "encoder/norm", encoder$output)
    out$encoder_attention <- encoder$attention
    out
}

# Decoder half: from the code and patch locations only (no image/target).
# Adds decoder_attention and prediction to the list `out`.
masked_autoencoder_decode <- function(p, code, selector, out = list()) {
    mask <- nn$sub(1, nn$sum(selector, 1L))
    projected <- nn$dense(p, "decoder_projection", code, 16L)
    restored <- nn$matmul(selector, projected, TRUE)
    token <- p$get("mask_token", 16L, nn$init$zeros())
    decoder_input <- nn$add(nn$add(restored, nn$mul(nn$expand_dims(mask, 2L), token)), masked_autoencoder_position(16L))
    decoder <- masked_autoencoder_block(p, "decoder", decoder_input, 16L)
    out$decoder_attention <- decoder$attention
    out$prediction <- nn$dense(p, "prediction", nn$layer_norm(p, "decoder/norm", decoder$output), 49L,
                               nn$Activation$Sigmoid)
    out
}

masked_autoencoder_forward <- function(p, image, selector) {
    out <- masked_autoencoder_encode(p, image, selector)
    masked_autoencoder_decode(p, out$code, selector, out)
}

# Squared error over hidden patches only, / (49 x number of hidden patches).
masked_autoencoder_loss <- function(prediction, patches, mask) {
    error <- nn$square(nn$sub(prediction, patches))
    nn$div(nn$sum(nn$mul(error, nn$expand_dims(mask, 2L))), nn$mul(nn$sum(mask), 49))
}

# Parameters (seed 1877) + optional Adam; forward pieces only.
masked_autoencoder_model <- function(training = TRUE) {
    model <- new.env()
    model$parameters <- nn$ParameterStore(1877L)
    one <- numeric(64L)
    one[(0L:3L) * 16L + (0L:3L) + 1L] <- 1
    nn$no_grad(function() masked_autoencoder_forward(model$parameters, nn$zeros(c(1L, 28L, 28L, 1L)),
                                                     nn$from_vector(one, c(1L, 4L, 16L))))
    model$optimizer <- if (training) autoencoder_make_adam(model$parameters) else NULL
    model$forward <- function(images, selection) masked_autoencoder_forward(model$parameters, images, selection)
    model$loss <- function(images, selection) {
        n <- model$forward(images, selection)
        masked_autoencoder_loss(n$prediction, n$patches, n$mask)
    }
    # list(code [B, 4, 32], prediction [B, 16, 49], hidden-only MSE)
    model$evaluate <- function(images, selection) nn$no_grad(function() {
        n <- model$forward(images, selection)
        list(n$code, n$prediction, masked_autoencoder_loss(n$prediction, n$patches, n$mask))
    })
    model$decode <- function(codes, selection) nn$no_grad(function()
        masked_autoencoder_decode(model$parameters, codes, selection)$prediction)
    # list(patches, visible, mask, encoder attention, decoder attention)
    model$probe <- function(images, selection) nn$no_grad(function() {
        n <- model$forward(images, selection)
        list(n$patches, n$visible, n$mask, n$encoder_attention, n$decoder_attention)
    })
    model
}

# uint32 avalanche hash (vectorised over doubles holding uint32 values).
masked_autoencoder_mix <- function(x) {
    x <- autoencoder_xor32(x, x %/% 65536)
    x <- autoencoder_mul32(x, 2146121005)  # 0x7feb352d
    x <- autoencoder_xor32(x, x %/% 32768)
    x <- autoencoder_mul32(x, 2221713035)  # 0x846ca68b
    autoencoder_xor32(x, x %/% 65536)
}

# Flat [n, 4, 16] selection values for source ids (hash of source id, epoch, seed).
masked_autoencoder_selection_values <- function(ids, epoch = 0L, seed = 1907) {
    out <- numeric(length(ids) * 64L)
    i <- 0:15
    patch_term <- autoencoder_mul32(i, 3266489909)  # 0xc2b2ae35
    epoch_term <- autoencoder_mul32(epoch, 2246822507)  # 0x85ebca6b
    for (b in seq_along(ids)) {
        base <- autoencoder_xor32(autoencoder_xor32(seed, autoencoder_mul32(ids[b] %% 4294967296, 2654435769)), epoch_term)
        keys <- masked_autoencoder_mix(autoencoder_xor32(rep(base, 16L), patch_term))
        chosen <- i[order(keys, i)][1:4]
        out[(b - 1L) * 64L + (0:3) * 16L + chosen + 1L] <- 1
    }
    out
}

# Reproducible per-example 4-of-16 visible patch selection [n, 4, 16].
masked_autoencoder_selection <- function(source, epoch = 0L, seed = 1907) {
    ids <- source$contiguous()$values()
    nn$from_vector(masked_autoencoder_selection_values(ids, epoch, seed), c(length(ids), 4L, 16L))
}

# Flat (0-based) pixel index of element `within` of 7x7 patch `patch`.
masked_autoencoder_pixel <- function(patch, within)
    ((patch %/% 4L) * 7L + within %/% 7L) * 28L + (patch %% 4L) * 7L + within %% 7L

# 1-based image index for every patch element in [16, 49] patch order (length 784).
masked_autoencoder_pixel_index <- as.vector(outer(0:48, 0:15, function(i, p) masked_autoencoder_pixel(p, i))) + 1L

# [n, 16, 49] patch values -> [n, 784] image values (flat vector).
masked_autoencoder_image_values <- function(values, n) {
    out <- numeric(n * 784L)
    for (b in seq_len(n)) out[(b - 1L) * 784L + masked_autoencoder_pixel_index] <- values[(b - 1L) * 784L + 1:784]
    out
}

# [n, 16, 49] patches -> [n, 784] image.
masked_autoencoder_image <- function(patches) {
    n <- patches$size(0L)
    nn$from_vector(masked_autoencoder_image_values(patches$contiguous()$values(), n), c(n, 784L))
}

# [n, 16] logical: whether patch p of example b is hidden (flat [n, 4, 16] selection values).
masked_autoencoder_hidden_matrix <- function(selection, n) {
    s <- array(selection, c(16L, 4L, n))
    t(apply(s, c(1L, 3L), function(v) all(v == 0)))
}

# Whether patch `patch` (0-based) of example b (0-based) is hidden.
masked_autoencoder_hidden <- function(selection, b, patch)
    all(selection[b * 64L + (0:3) * 16L + patch + 1L] == 0)

# Per-pixel logical [n * 784] (image order): pixel belongs to a hidden patch.
masked_autoencoder_hidden_pixels <- function(selection, n) {
    hidden <- masked_autoencoder_hidden_matrix(selection, n)
    out <- logical(n * 784L)
    for (b in seq_len(n))
        out[(b - 1L) * 784L + masked_autoencoder_pixel_index] <- rep(hidden[b, ], each = 49L)
    out
}

# Independent native hidden-only MSE (C++ summation order: example, patch, element).
masked_autoencoder_mse <- function(prediction, original, selection) {
    n <- prediction$size(0L)
    p <- prediction$contiguous()$values()  # patch order equals the C++ loop order
    o <- original$contiguous()$values()
    hidden <- masked_autoencoder_hidden_matrix(selection$contiguous()$values(), n)
    keep <- as.vector(t(hidden[, rep(1:16, each = 49L), drop = FALSE]))
    target <- o[rep((seq_len(n) - 1L) * 784L, each = 784L) + masked_autoencoder_pixel_index]
    y <- p[keep]
    autoencoder_require(all(is.finite(y) & y >= 0 & y <= 1), "bounded finite patch prediction")
    autoencoder_require(length(y) == n * 12L * 49L, "exact75percent hidden loss denominator")
    autoencoder_sum((y - target[keep])^2) / length(y)
}
