"""Small MNIST CNN (examples/nn/common/cnn_baselines.hpp). Parameter names match
tflow's cnn_mnist checkpoint: conv1, conv2, dense, output. Intermediate
activations are returned for interpretation examples."""
from types import SimpleNamespace

from pydatamunge import nn


def mnist_cnn(parameters, inputs):
    """3x3 Conv(16)-ReLU-pool, 3x3 Conv(32)-ReLU-pool, Dense(64)-ReLU, 10 logits.

    `inputs` is [N, 784] or [N, 28, 28, 1]. Returns every stage
    (image, conv1, pool1, conv2, pool2, hidden, logits)."""
    options = nn.ConvLayerOptions()
    options.activation = nn.Activation.Relu
    out = SimpleNamespace()
    out.image = nn.reshape(inputs, [-1, 28, 28, 1])
    out.conv1 = nn.conv2d(parameters, "conv1", out.image, 16, 3, options)
    out.pool1 = nn.max_pool2d(out.conv1, 2, 2)
    out.conv2 = nn.conv2d(parameters, "conv2", out.pool1, 32, 3, options)
    out.pool2 = nn.max_pool2d(out.conv2, 2, 2)
    out.hidden = nn.dense(parameters, "dense", nn.reshape(out.pool2, [-1, 7 * 7 * 32]), 64, nn.Activation.Relu)
    out.logits = nn.dense(parameters, "output", out.hidden, 10)
    return out
