# Hard winner competition autoencoders (examples/nn/common/competitive_autoencoder.hpp).
#
# 784 -> 64 -> 784 sigmoid autoencoders and their matched-weight plain controls,
# shared by k_sparse_autoencoder (per-image top-8 SIGNED activities of a LINEAR
# encoder, kept at inference) and winner_take_all_autoencoder (lifetime per-feature
# winners of a ReLU encoder, training only). Masks are computed from activation
# values outside the graph (a constant), with TopK's tie rule: equal values select
# the lower index. Training steps are written out in each example.

if (!exists("autoencoder_undercomplete", inherits = TRUE)) nn_helper("autoencoder", environment())
if (!exists("nn_require", inherits = TRUE)) nn_helper("checks", environment())

# Binary [B, F] selector: lifetime -> per feature the ceil(B/8) largest examples;
# otherwise per example the k largest features (stable order: ties keep the lower index).
competitive_autoencoder_mask <- function(scores, lifetime, k = 8L) {
    nn_require(scores$dim() == 2L, "competitive scores [batch, features]")
    batch <- scores$size(0L)
    features <- scores$size(1L)
    v <- matrix(scores$contiguous()$values(), nrow = batch, ncol = features, byrow = TRUE)
    mask <- matrix(0, nrow = batch, ncol = features)
    if (lifetime) {
        keep <- (batch + 7L) %/% 8L
        for (j in seq_len(features)) mask[head(order(-v[, j], method = "radix"), keep), j] <- 1
    } else {
        for (b in seq_len(batch)) mask[b, head(order(-v[b, ], method = "radix"), k)] <- 1
    }
    nn$cast(nn$from_vector_double(as.vector(t(mask)), scores$shape()), scores$dtype())
}

competitive_autoencoder_forward <- function(p, x, lifetime, plain = FALSE) {
    out <- list()
    out$raw <- nn$dense(p, "encoder", x, 64L, if (lifetime) nn$Activation$Relu else nn$Activation$None)
    out$mask <- if (plain) nn$ones_like(out$raw) else competitive_autoencoder_mask(out$raw$detach(), lifetime)
    out$training_code <- nn$mul(out$raw, out$mask)
    # FC-WTA lifetime competition is TRAINING ONLY. k-sparse competition stays at inference.
    out$code <- if (lifetime) out$raw else out$training_code
    out$training_reconstruction <- nn$dense(p, "decoder", out$training_code, 784L, nn$Activation$Sigmoid)
    out$reconstruction <- if (lifetime) nn$dense(p, "decoder", out$code, 784L, nn$Activation$Sigmoid)
                          else out$training_reconstruction
    out
}

# Parameters (seed 1931) + optional Adam; forward pieces only.
competitive_autoencoder_model <- function(lifetime = TRUE, plain = FALSE, training = TRUE) {
    model <- new.env()
    model$lifetime <- lifetime
    model$plain <- plain
    model$parameters <- nn$ParameterStore(1931L)
    nn$no_grad(function() competitive_autoencoder_forward(model$parameters, nn$zeros(c(1L, 784L)), lifetime, plain))
    model$optimizer <- if (training) autoencoder_make_adam(model$parameters) else NULL
    # Graph-building forward of flattened [n, 784] inputs.
    model$forward <- function(x) competitive_autoencoder_forward(model$parameters, x, lifetime, plain)
    # list(code, reconstruction, inference MSE)
    model$evaluate <- function(images) nn$no_grad(function() {
        x <- autoencoder_flatten(images)
        n <- model$forward(x)
        list(n$code, n$reconstruction, autoencoder_reconstruction_mse(n$reconstruction, x))
    })
    model$decode <- function(codes) nn$no_grad(function()
        nn$dense(model$parameters, "decoder", codes, 784L, nn$Activation$Sigmoid))
    # Loads matched initial parameters (all tensors, same shapes) before training.
    model$load <- function(initial) {
        nn_require(identical(initial$names(), model$parameters$names()), "all initial model tensors supplied")
        model$parameters$copy_from(initial)
    }
    # Training-mode diagnostic: list(raw, mask, training code, training reconstruction).
    model$probe <- function(images) nn$no_grad(function() {
        n <- model$forward(autoencoder_flatten(images))
        list(n$raw, n$mask, n$training_code, n$training_reconstruction)
    })
    model
}

competitive_autoencoder_check_support <- function(probe, lifetime, plain) {
    batch <- probe[[1L]]$size(0L)
    raw <- probe[[1L]]$contiguous()$values()
    mask <- probe[[2L]]$contiguous()$values()
    code <- probe[[3L]]$contiguous()$values()
    nn_require(all(mask == 0 | mask == 1), "binary hard selector")
    nn_require(all(code == raw * mask), "code equals selected ACTUAL activation")
    counts <- matrix(mask, nrow = batch, ncol = 64L, byrow = TRUE)
    if (lifetime) {
        nn_require(all(colSums(counts) == (if (plain) batch else (batch + 7L) %/% 8L)),
                   "exact selected EXAMPLES per FEATURE;partial batch correct")
    } else {
        nn_require(all(rowSums(counts) == (if (plain) 64L else 8L)), "exact selected FEATURES per EXAMPLE")
    }
}

competitive_autoencoder_checks <- function() {
    x <- nn$from_vector(c(3, 1, 1, 0, 2, 4, 0, 0, -1, -2, 5, 5, -4, -1, -2, -3), c(4L, 4L))$requires_grad_()
    lifetime <- competitive_autoencoder_mask(x$detach(), TRUE)
    population <- competitive_autoencoder_mask(x$detach(), FALSE, 2L)
    ga <- nn$grad(nn$sum(nn$mul(x, lifetime)), list(x))[[1L]]$values()
    gb <- nn$grad(nn$sum(nn$mul(x, population)), list(x))[[1L]]$values()
    expected_lifetime <- c(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0)
    expected_population <- c(1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0)
    nn_require(identical(lifetime$values(), expected_lifetime), "ALL16lifetime winners acrossbatch")
    nn_require(identical(population$values(), expected_population),
               "ALL16perimage winners,ties lowerindices,negativeSIGNEDnotmagnitude")
    nn_require(identical(ga, expected_lifetime), "lifetime gradient EXACTzero outsidewinners")
    nn_require(identical(gb, expected_population), "perimage gradient EXACTzero outsidewinners")
    cat("known axes/lower-index ties/negativeSIGNEDactivities/ALL32winner masks+32derivatives passed\n")
}
