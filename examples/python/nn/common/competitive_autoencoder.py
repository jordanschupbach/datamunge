"""Hard winner competition autoencoders (examples/nn/common/competitive_autoencoder.hpp).

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


def competitive_mask(scores, lifetime, k=8):
    """Binary [B, F] selector: lifetime -> per feature the ceil(B/8) largest examples;
    otherwise per example the k largest features (stable: ties keep the lower index)."""
    checks.require(scores.dim() == 2, "competitive scores [batch, features]")
    batch, features = scores.size(0), scores.size(1)
    v = scores.contiguous().values()
    mask = [0.0] * len(v)

    def pick(count, keep, index):
        order = sorted(range(count), key=lambda i: -v[index(i)])
        for i in order[:keep]:
            mask[index(i)] = 1.0

    if lifetime:
        keep = (batch + 7) // 8
        for j in range(features):
            pick(batch, keep, lambda b, j=j: b * features + j)
    else:
        for b in range(batch):
            pick(features, k, lambda j, b=b: b * features + j)
    return nn.cast(nn.from_vector_double(mask, list(scores.shape())), scores.dtype())


class CompetitiveOutput:
    raw = mask = training_code = code = training_reconstruction = reconstruction = None


def competitive_autoencoder(p, x, lifetime, plain=False):
    out = CompetitiveOutput()
    out.raw = nn.dense(p, "encoder", x, 64, nn.Activation.Relu if lifetime else nn.Activation.Identity)
    out.mask = nn.ones_like(out.raw) if plain else competitive_mask(out.raw.detach(), lifetime)
    out.training_code = nn.mul(out.raw, out.mask)
    # FC-WTA lifetime competition is TRAINING ONLY. k-sparse competition stays at inference.
    out.code = out.raw if lifetime else out.training_code
    out.training_reconstruction = nn.dense(p, "decoder", out.training_code, 784, nn.Activation.Sigmoid)
    out.reconstruction = (nn.dense(p, "decoder", out.code, 784, nn.Activation.Sigmoid) if lifetime
                          else out.training_reconstruction)
    return out


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
    batch = probe[0].size(0)
    raw, mask, code = (t.contiguous().values() for t in probe[:3])
    for i in range(batch * 64):
        checks.require(mask[i] in (0.0, 1.0), "binary hard selector")
        checks.near(code[i], raw[i] * mask[i], 0, "code equals selected ACTUAL activation")
    if lifetime:
        for j in range(64):
            n = sum(int(mask[b * 64 + j]) for b in range(batch))
            checks.require(n == (batch if plain else (batch + 7) // 8),
                           "exact selected EXAMPLES per FEATURE;partial batch correct")
    else:
        for b in range(batch):
            n = sum(int(mask[b * 64 + j]) for j in range(64))
            checks.require(n == (64 if plain else 8), "exact selected FEATURES per EXAMPLE")


def competitive_checks():
    x = nn.from_vector([3, 1, 1, 0, 2, 4, 0, 0, -1, -2, 5, 5, -4, -1, -2, -3], [4, 4]).requires_grad_()
    lifetime = competitive_mask(x.detach(), True)
    population = competitive_mask(x.detach(), False, 2)
    ga = nn.grad(nn.sum(nn.mul(x, lifetime)), [x])[0].values()
    gb = nn.grad(nn.sum(nn.mul(x, population)), [x])[0].values()
    lv, pv = lifetime.values(), population.values()
    expected_lifetime = [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0]
    expected_population = [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0]
    for i in range(16):
        checks.near(lv[i], expected_lifetime[i], 0, "ALL16lifetime winners acrossbatch")
        checks.near(pv[i], expected_population[i], 0,
                    "ALL16perimage winners,ties lowerindices,negativeSIGNEDnotmagnitude")
        checks.near(ga[i], expected_lifetime[i], 0, "lifetime gradient EXACTzero outsidewinners")
        checks.near(gb[i], expected_population[i], 0, "perimage gradient EXACTzero outsidewinners")
    print("known axes/lower-index ties/negativeSIGNEDactivities/ALL32winner masks+32derivatives passed")
