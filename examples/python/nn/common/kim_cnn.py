"""Kim CNN-rand model pieces (examples/nn/common/kim_cnn.hpp).

Learned word embeddings with a structural zero PAD row, parallel 3/4/5 ReLU
word convolutions, max-over-time pooling, dropout and a binary head, plus
Kim's Adadelta (rho .95, eps 1e-6) with the softmax-column max-norm 3.
"""
from dataclasses import dataclass, field

from pydatamunge import nn


@dataclass
class KimConfig:
    vocabulary: int = 8000
    time: int = 64
    embedding: int = 300
    filters: int = 100


@dataclass
class KimOutput:
    logits: object = None
    features: object = None
    embedding: object = None
    maps: list = field(default_factory=list)


def pad_axis(x, axis, before, after):
    """nn::pad of one axis with zeros (nn.pad's paddings vector is not bound; concat is equivalent)."""
    parts = []
    shape = [int(v) for v in x.shape()]
    if before > 0:
        shape[axis] = before
        parts.append(nn.zeros(list(shape), x.dtype()))
    parts.append(x)
    if after > 0:
        shape[axis] = after
        parts.append(nn.zeros(list(shape), x.dtype()))
    return x if len(parts) == 1 else nn.concat(parts, axis)


def length_mask(lengths, time, dtype=nn.DType.Float32):
    """[N, T] mask: t < lengths[n]."""
    positions = nn.arange(0, float(time), 1, nn.DType.Int32)
    keep = nn.less(nn.expand_dims(positions, 0), nn.expand_dims(nn.cast(lengths, nn.DType.Int32), 1))
    return nn.cast(keep, dtype)


def kim_max_norm(w):
    """Kim's max-norm constraint: columns of w scaled to norm <= 3."""
    norm = nn.sqrt(nn.sum(nn.square(w), [0], True))
    return nn.div(w, nn.maximum(nn.scalar_like(1.0, w), nn.div(norm, 3.0)))


def kim_cnn(p, tokens, lengths, config=None, training=False, keep=1.0, gen=None):
    """tokens [N, T] int, lengths [N]; "embedding/nonpad" holds rows 1..V-1 (U(-.25, .25))."""
    config = config or KimConfig()
    if config.vocabulary < 2 or config.time < 1 or config.embedding < 1 or config.filters < 1:
        raise ValueError("Kim CNN dimensions")
    out = KimOutput()
    learned = p.get("embedding/nonpad", [config.vocabulary - 1, config.embedding], nn.init.uniform(.25))
    out.embedding = nn.concat([nn.zeros([1, config.embedding], learned.dtype()), learned], 0)
    mask = length_mask(lengths, config.time, learned.dtype())
    x = nn.mul(nn.gather(out.embedding, tokens, 0), nn.expand_dims(mask, 2))
    padded = pad_axis(x, 1, 4, 4)
    pooled = []
    for kernel in (3, 4, 5):
        options = nn.ConvLayerOptions()
        options.padding = nn.Padding.Valid
        options.activation = nn.Activation.Relu
        feature_map = nn.conv1d(p, "window" + str(kernel), padded, config.filters, kernel, options)
        out.maps.append(feature_map)
        pooled.append(nn.max(feature_map, [1]))
    out.features = nn.concat(pooled, 1)
    feature = out.features
    if training and keep < 1:
        feature = nn.dropout(feature, 1 - keep, True, gen)
    out.logits = nn.dense(p, "classifier", feature, 2)
    return out


class KimAdadelta:
    """Adadelta rho=.95, eps=1e-6 and softmax-column norm <= 3 as in Kim.

    Slots "gradient<i>" / "update<i>" start at zero; every new value is computed
    before any parameter or slot is written (TF's all-RHS barrier).
    """

    def __init__(self, p):
        self.p = p
        self.slots = nn.ParameterStore(319)
        for i, value in enumerate(p.parameters()):
            shape = list(value.shape())
            self.slots.get("gradient" + str(i), shape, nn.init.zeros())
            self.slots.get("update" + str(i), shape, nn.init.zeros())

    def step(self, gradients):
        gradients = list(gradients)
        if len(gradients) != self.p.size():
            raise ValueError("Adadelta gradient alignment")
        with nn.no_grad():
            params, slot_values = [], []
            names = self.p.names()
            for i, gradient in enumerate(gradients):
                name = names[i]
                g = gradient.detach()
                g2 = self.slots.at("gradient" + str(i)).detach()
                d2 = self.slots.at("update" + str(i)).detach()
                next_g = nn.add(nn.mul(g2, .95), nn.mul(nn.square(g), .05))
                delta = nn.mul(nn.div(nn.sqrt(nn.add(d2, 1e-6)), nn.sqrt(nn.add(next_g, 1e-6))), g)
                next_d = nn.add(nn.mul(d2, .95), nn.mul(nn.square(delta), .05))
                next_p = nn.sub(self.p.at(name).detach(), delta)
                if name == "classifier/W":
                    next_p = kim_max_norm(next_p)
                params.append((name, next_p))
                slot_values.append(("gradient" + str(i), next_g))
                slot_values.append(("update" + str(i), next_d))
            for name, value in params:
                self.p.set(name, value.clone())
            for name, value in slot_values:
                self.slots.set(name, value.clone())
