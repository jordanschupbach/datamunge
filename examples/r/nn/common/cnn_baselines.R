# Small MNIST CNN (examples/nn/common/cnn_baselines.hpp). Parameter names match
# tflow's cnn_mnist checkpoint: conv1, conv2, dense, output.

# 3x3 Conv(16)-ReLU-pool, 3x3 Conv(32)-ReLU-pool, Dense(64)-ReLU, 10 logits.
# `inputs` is [N, 784] or [N, 28, 28, 1]; returns every stage as a list.
cnn_baselines_mnist_cnn <- function(parameters, inputs) {
    options <- nn$ConvLayerOptions()
    options$activation <- nn$Activation$Relu
    out <- list()
    out$image <- nn$reshape(inputs, c(-1L, 28L, 28L, 1L))
    out$conv1 <- nn$conv2d(parameters, "conv1", out$image, 16L, 3L, options)
    out$pool1 <- nn$max_pool2d(out$conv1, 2L, 2L)
    out$conv2 <- nn$conv2d(parameters, "conv2", out$pool1, 32L, 3L, options)
    out$pool2 <- nn$max_pool2d(out$conv2, 2L, 2L)
    out$hidden <- nn$dense(parameters, "dense", nn$reshape(out$pool2, c(-1L, 7L * 7L * 32L)), 64L,
                           nn$Activation$Relu)
    out$logits <- nn$dense(parameters, "output", out$hidden, 10L)
    out
}
