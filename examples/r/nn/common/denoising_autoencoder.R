# Denoising autoencoder model (examples/nn/common/denoising_autoencoder.hpp).
#
# The dense 784->32 ReLU->784 sigmoid undercomplete autoencoder (seed 1657) with
# target-free inference. The corruption itself is autoencoder_corrupt_for_denoising;
# corrupted-input/clean-target training steps are written out in each example.

if (!exists("autoencoder_undercomplete", inherits = TRUE)) nn_helper("autoencoder", environment())

denoising_autoencoder_model <- function(training = TRUE) {
    model <- new.env()
    model$parameters <- nn$ParameterStore(1657L)
    nn$no_grad(function() autoencoder_undercomplete(model$parameters, nn$zeros(c(1L, 784L))))
    model$optimizer <- if (training) autoencoder_make_adam(model$parameters) else NULL
    # list(code, reconstruction) of the flattened images (graph-building).
    model$forward <- function(images) autoencoder_undercomplete(model$parameters, autoencoder_flatten(images))
    # list(code, reconstruction): inference needs the (corrupted) image ONLY.
    model$predict <- function(images) nn$no_grad(function() {
        n <- model$forward(images)
        list(n$code, n$reconstruction)
    })
    # MSE (double) of the reconstruction of `images` against the CLEAN target.
    model$loss <- function(images, clean) nn$no_grad(function()
        autoencoder_reconstruction_mse(model$forward(images)$reconstruction, autoencoder_flatten(clean))$item())
    model$decode <- function(codes) nn$no_grad(function()
        nn$dense(model$parameters, "decoder", codes, 784L, nn$Activation$Sigmoid))
    model
}
