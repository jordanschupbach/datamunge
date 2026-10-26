"""Denoising autoencoder model (examples/nn/common/denoising_autoencoder.hpp).

The dense 784->32 ReLU->784 sigmoid undercomplete autoencoder (seed 1657) with
target-free inference. The corruption itself is autoencoder.corrupt_for_denoising;
corrupted-input/clean-target training steps are written out in each example.
"""
from pydatamunge import nn
from pydatamunge.nn_workflows import helper

ae = helper("autoencoder")


class DenoisingModel:
    def __init__(self, training=True):
        self.parameters = nn.ParameterStore(1657)
        with nn.no_grad():
            ae.undercomplete_autoencoder(self.parameters, nn.zeros([1, 784]))
        self.optimizer = ae.make_adam(self.parameters) if training else None

    def forward(self, images):
        """DenseAeOutput(code, reconstruction) of the flattened images (graph-building)."""
        return ae.undercomplete_autoencoder(self.parameters, ae.flatten(images))

    def predict(self, images):
        """[code, reconstruction]: inference needs the (corrupted) image ONLY."""
        with nn.no_grad():
            n = self.forward(images)
            return [n.code, n.reconstruction]

    def loss(self, images, clean):
        """MSE (float) of the reconstruction of `images` against the CLEAN target."""
        with nn.no_grad():
            return ae.reconstruction_mse(self.forward(images).reconstruction, ae.flatten(clean)).item()

    def decode(self, codes):
        with nn.no_grad():
            return nn.dense(self.parameters, "decoder", codes, 784, nn.Activation.Sigmoid)
