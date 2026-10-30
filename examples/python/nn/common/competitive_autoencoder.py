"""Hard winner competition autoencoders (examples/nn/common/competitive_autoencoder.hpp).

784 -> 64 -> 784 sigmoid autoencoders and their matched-weight plain controls,
shared by k_sparse_autoencoder (per-image top-8 SIGNED activities of a LINEAR
encoder, kept at inference) and winner_take_all_autoencoder (lifetime per-feature
winners of a ReLU encoder, training only). Masks are computed from activation
values outside the graph (a constant), with TopK's tie rule: equal values select
the lower index. Training steps are written out in each example.
"""
from pydatamunge import nn
from pydatamunge.nn_components import ae as _core
from pydatamunge.nn_workflows import helper

ae = helper("autoencoder")
checks = helper("checks")


competitive_mask = _core.CompetitiveMask
competitive_autoencoder = _core.CompetitiveAutoencoder


class CompetitiveAutoencoderModel:
    """Parameters (seed 1931) + optional Adam; forward pieces only."""

    def __init__(self, lifetime=True, plain=False, training=True):
        self.lifetime, self.plain = lifetime, plain
        self.parameters = nn.ParameterStore(1931)
        with nn.no_grad():
            competitive_autoencoder(self.parameters, nn.zeros([1, 784]), lifetime, plain)
        self.optimizer = ae.make_adam(self.parameters) if training else None

    def forward(self, x):
        """Graph-building forward of flattened [n, 784] inputs."""
        return competitive_autoencoder(self.parameters, x, self.lifetime, self.plain)

    def evaluate(self, images):
        """[code, reconstruction, inference MSE]"""
        with nn.no_grad():
            x = ae.flatten(images)
            n = self.forward(x)
            return [n.code, n.reconstruction, ae.reconstruction_mse(n.reconstruction, x)]

    def decode(self, codes):
        with nn.no_grad():
            return nn.dense(self.parameters, "decoder", codes, 784, nn.Activation.Sigmoid)

    def load(self, initial):
        """Loads matched initial parameters (all tensors, same shapes) before training."""
        checks.require(list(initial.names()) == list(self.parameters.names()), "all initial model tensors supplied")
        self.parameters.copy_from(initial)

    def probe(self, images):
        """Training-mode diagnostic: [raw, mask, training code, training reconstruction]."""
        with nn.no_grad():
            n = self.forward(ae.flatten(images))
            return [n.raw, n.mask, n.training_code, n.training_reconstruction]


# The same C++ support/gradient fixtures used by the canonical examples.
check_competitive_support = _core.CheckCompetitiveSupport
competitive_checks = _core.CompetitiveChecks
