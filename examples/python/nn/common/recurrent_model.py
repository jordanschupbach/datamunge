"""Model container for tflow's small explicit-unroll recurrent models
(examples/nn/common/recurrent_model.hpp).

A model is a ParameterStore, a forward function
``forward(store, inputs, lengths, initial_states) -> {name: tensor}`` that must
return "logits", a loss ``loss_fn(outputs, labels) -> (loss, chosen)`` and the
optimizer. Gradients, updates and evaluations are written out in each example
script (forward, loss, ``nn.grad``, ``optimizer.step``); this module only holds
the components and the tensor comparison utilities the examples share.
"""
from collections import namedtuple

from pydatamunge import nn

# One batch fed to a model: inputs, int32 lengths and labels.
Feed = namedtuple("Feed", "inputs lengths labels")

# A gradient probe: d target(outputs) / d (input when wrt == 0, else initial state wrt - 1).
Probe = namedtuple("Probe", "name target wrt")


def softmax_loss(outputs, labels):
    """Two-class softmax cross-entropy with int32 labels and the summed target logit (tflow's loss and chosen)."""
    logits = outputs["logits"]
    one_hot = nn.one_hot(labels, logits.shape()[-1])
    loss = nn.neg(nn.mean(nn.sum(nn.mul(one_hot, nn.log_softmax(logits, -1)), [1]), [0]))
    return loss, nn.sum(nn.mul(logits, one_hot))


def same_bits(a, b):
    """Bit-exact equality of two tensors, or of two named-tensor lists/dicts (names, order and values)."""
    if isinstance(a, dict):
        a, b = list(a.items()), list(b.items())
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(x[0] == y[0] and same_bits(x[1], y[1]) for x, y in zip(a, b))
    return a.dtype() == b.dtype() and tuple(a.shape()) == tuple(b.shape()) and a.values() == b.values()


def named(outputs):
    """Outputs dict -> [(name, tensor)] in the sorted order of the C++ std::map."""
    return sorted(outputs.items(), key=lambda item: item[0])


def find(arrays, name):
    for key, value in arrays:
        if key == name:
            return value
    raise RuntimeError("missing array " + name)


def max_abs_diff(a, b):
    x, y = a.values(), b.values()
    if len(x) != len(y):
        return float("inf")
    return max((abs(p - q) for p, q in zip(x, y)), default=0.0)


def defined_gradient(gradient, like):
    """A gradient from nn.grad (detached), or zeros like `like` when the target did not use it."""
    return gradient.detach() if gradient.defined() else nn.zeros_like(like.detach())


class RecurrentModel:
    """Parameters, forward function, loss and optimizer of one recurrent model."""

    def __init__(self, seed, forward, input_shape, initial_shapes, initial_gradient_names, fields):
        self.store = nn.ParameterStore(seed)
        self.forward = forward
        self.loss_fn = softmax_loss
        self.probability = lambda logits: nn.softmax(logits, -1)
        self.input_shape = list(input_shape)  # per-example input shape
        self.initial_shapes = [list(s) for s in initial_shapes]  # per-example zero initial states
        self.initial_gradient_names = list(initial_gradient_names)  # "" = not reported
        self.fields = list(fields)  # forward fields returned by predict()
        self.evaluate_probes = []
        self.side_inputs = {}  # extra named inputs a forward may read (dropout masks, timestamps)
        self.optimizer = None

    def build(self, training, learning_rate=0.01, kind=nn.OptimizerKind.Adam):
        """Creates every parameter with one no-grad example forward; with `training`, adds the optimizer."""
        with nn.no_grad():
            self.forward(self.store, nn.zeros([1] + self.input_shape), nn.zeros([1], nn.DType.Int32), self.initials(1))
        if training:
            config = nn.OptimizerConfig()
            config.kind = kind
            self.optimizer = nn.Optimizer(self.store, learning_rate, config)
        return self

    def initials(self, n):
        return [nn.zeros([n] + shape) for shape in self.initial_shapes]

    def predict(self, feed):
        """Sequence and lengths only, no labels or gradients (tflow's Predict)."""
        with nn.no_grad():
            out = self.forward(self.store, feed.inputs, feed.lengths, self.initials(feed.inputs.shape()[0]))
        return {f: (self.probability(out["logits"]) if f == "probability" else out[f]) for f in self.fields}

    def arrays(self):
        """Copies of every parameter, by name, in creation order."""
        return [(name, self.store.at(name).detach().clone()) for name in self.store.names()]

    def save(self, path):
        self.store.save(str(path))

    def restore(self, path):
        self.store.load(str(path), True)

    def load_arrays(self, path):
        """Overwrites every parameter from an .npz written by tflow (initial_arrays.npz)."""
        arrays = nn.load_npz(str(path))
        if len(arrays) != self.store.size():
            raise RuntimeError("init arrays cover exactly the model's parameters")
        for name, value in arrays:
            if not self.store.contains(name):
                raise RuntimeError("init array " + name + " is a model parameter")
            self.store.set(name, nn.cast(value, self.store.at(name).dtype()))
