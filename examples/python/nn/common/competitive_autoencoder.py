"""Canonical C++ hard winner competition with host-owned experiment state.

784 -> 64 -> 784 sigmoid autoencoders and their matched-weight plain controls,
shared by k_sparse_autoencoder (per-image top-8 SIGNED activities of a LINEAR
encoder, kept at inference) and winner_take_all_autoencoder (lifetime per-feature
winners of a ReLU encoder, training only). Masks are computed from activation
values outside the graph (a constant), with TopK's tie rule: equal values select
the lower index. Training steps are written out in each example.
"""
from pydatamunge import nn
from pydatamunge.nn_workflows import helper

ae = helper("autoencoder")
checks = helper("checks")


competitive_mask = nn.CompetitiveMask
competitive_autoencoder = nn.CompetitiveAutoencoder


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


def check_competitive_support(probe, lifetime, plain):
    """Check diagnostic invariants; the C++ core computes all masks and outputs."""
    raw, mask, code = probe[:3]
    b, f = raw.shape()
    values = list(mask.values())
    checks.require(all(v in (0, 1) for v in values), "binary selector")
    checks.require(list(code.values()) == [a*m for a,m in zip(raw.values(),values)],
                   "code equals selected actual activities")
    if lifetime:
        counts = [sum(values[row*f+column] for row in range(b)) for column in range(f)]
        expected = b if plain else (b+7)//8
    else:
        counts = [sum(values[row*f:(row+1)*f]) for row in range(b)]
        expected = f if plain else min(8,f)
    checks.require(all(v == expected for v in counts), "exact selected slots on competition axis")


def competitive_checks():
    """Exercise native mask axes, stable signed ties and winner-only gradients."""
    x = nn.from_vector([3,1,1,0,2,4,0,0,-1,-2,5,5,-4,-1,-2,-3], [4,4]).requires_grad_()
    expected = ([1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0],
                [1,1,0,0,1,1,0,0,0,0,1,1,0,1,1,0])
    for lifetime, values in zip((True,False),expected):
        mask = nn.CompetitiveMask(x,lifetime,2)
        checks.require(list(mask.values()) == values, "native toy winners")
        checks.require(list(nn.grad(nn.sum(nn.mul(x,mask)),[x])[0].values()) == values, "native toy winner derivatives")
