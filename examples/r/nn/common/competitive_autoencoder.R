# Canonical C++ competition with host-owned experiment state and diagnostics.
#
# 784 -> 64 -> 784 sigmoid autoencoders and their matched-weight plain controls,
# shared by k_sparse_autoencoder (per-image top-8 SIGNED activities of a LINEAR
# encoder, kept at inference) and winner_take_all_autoencoder (lifetime per-feature
# winners of a ReLU encoder, training only). Masks are computed from activation
# values outside the graph (a constant), with TopK's tie rule: equal values select
# the lower index. Training steps are written out in each example.

if (!exists("autoencoder_undercomplete", inherits = TRUE)) nn_helper("autoencoder", environment())
if (!exists("nn_require", inherits = TRUE)) nn_helper("checks", environment())

# Canonical C++ forward, discrete mask and fixtures through the bindings.
competitive_autoencoder_mask <- nn$CompetitiveMask
competitive_autoencoder_forward <- nn$CompetitiveAutoencoder

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
    raw <- probe[[1L]]; mask <- probe[[2L]]; code <- probe[[3L]]
    values <- matrix(mask$values(),nrow=raw$size(0L),byrow=TRUE)
    nn_require(all(values %in% c(0,1)),"binary selector")
    nn_require(identical(code$values(),raw$values()*mask$values()),"code equals selected actual activities")
    counts <- if(lifetime) colSums(values) else rowSums(values)
    expected <- if(plain) (if(lifetime) nrow(values) else ncol(values)) else
                (if(lifetime) ceiling(nrow(values)/8) else min(8,ncol(values)))
    nn_require(all(counts==expected),"exact selected slots on competition axis")
}
competitive_autoencoder_checks <- function() {
    x <- nn$from_vector(c(3,1,1,0,2,4,0,0,-1,-2,5,5,-4,-1,-2,-3),c(4L,4L))$requires_grad_()
    expected <- list(c(1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0),c(1,1,0,0,1,1,0,0,0,0,1,1,0,1,1,0))
    for(i in 1L:2L) {
        mask <- nn$CompetitiveMask(x,i==1L,2L)
        nn_require(identical(mask$values(),expected[[i]]),"native toy winners")
        nn_require(identical(nn$grad(nn$sum(nn$mul(x,mask)),list(x))[[1L]]$values(),expected[[i]]),"native toy winner derivatives")
    }
}
