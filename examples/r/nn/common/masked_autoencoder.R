# Tiny asymmetric masked autoencoder (include/datamunge/nn/masked_autoencoder.hpp).
#
# He et al. MAE on 16 non-overlapping 7x7 MNIST patches: a visible-only width-32
# transformer encoder (4 of 16 tokens), a width-16 mask-token decoder, fixed 2-D
# sin/cos positions and a hidden-only loss. Parameter names match tflow's. Model
# pieces, hash-derived patch selections and native metrics only: training steps
# (forward, masked_autoencoder_loss, nn$grad / optimizer) are written out in each example.

if (!exists("autoencoder_make_adam", inherits = TRUE)) nn_helper("autoencoder", environment())

masked_autoencoder_position <- nn$MaePosition
masked_autoencoder_block <- nn$MaeBlock
masked_autoencoder_patches <- nn$MaePatches
masked_autoencoder_forward <- nn$MaskedAutoencoder
masked_autoencoder_loss <- nn$MaeLoss

masked_autoencoder_encode <- function(p,image,selector,out=list()) {
    value <- nn$MaeEncode(p,image,selector)
    out$patches <- value$patches; out$visible <- value$visible; out$mask <- value$mask
    out$code <- value$code; out$encoder_attention <- value$encoder_attention
    out
}
masked_autoencoder_decode <- function(p,code,selector,out=list()) {
    value <- nn$MaeDecode(p,code,selector)
    out$prediction <- value$prediction; out$decoder_attention <- value$decoder_attention
    out
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

# Native mask generation preserves the original wrapping uint32 rank order.
masked_autoencoder_selection_values <- function(ids,epoch=0L,seed=1907) {
    source <- nn$cast(nn$from_vector_double(as.numeric(ids),as.integer(length(ids))),nn$DType$Int64)
    nn$MaeSelection(source,epoch,seed)$values()
}
masked_autoencoder_selection <- nn$MaeSelection
masked_autoencoder_pixel <- Vectorize(function(patch,within) nn$MaePixel(as.integer(patch),as.integer(within)))
masked_autoencoder_pixel_index <- as.vector(outer(0:48,0:15,function(i,p)masked_autoencoder_pixel(p,i)))+1L
masked_autoencoder_image_values <- function(values,n)
    nn$MaeImage(nn$from_vector(values,c(n,16L,49L)))$values()
masked_autoencoder_image <- nn$MaeImage

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
