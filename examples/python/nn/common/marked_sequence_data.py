"""Synthetic variable-length single-marker memory task; padding never reaches
the model through the length mask (examples/nn/common/marked_sequence_data.hpp;
tflow's common/ml/marked_sequence_data.h plus its per-example SaveData, Copy,
Accuracy and Heatmap helpers).

The generator is std::mt19937 exactly as in tflow (raw 32-bit draws), so every
split is bit-identical to tflow's data.npz."""
import math

from pydatamunge import nn
from pydatamunge.nn_workflows import helper

recurrent_model = helper("recurrent_model")


def int32_tensor(values, shape=None):
    values = list(values)
    return nn.cast(nn.from_vector_int64(values, [len(values)] if shape is None else shape), nn.DType.Int32)


class MarkedSequences:
    """inputs float [n, 12, 2] (value, marker), lengths/labels/source_ids int32 [n], uniforms float64 [n, 14]."""

    def __init__(self, inputs, lengths, labels, source_ids, uniforms):
        self.inputs, self.lengths, self.labels = inputs, lengths, labels
        self.source_ids, self.uniforms = source_ids, uniforms

    def feed(self):
        return recurrent_model.Feed(self.inputs, self.lengths, self.labels)

    def size(self):
        return self.inputs.shape()[0]


def raw_draws(seed):
    """The raw 32-bit outputs of std::mt19937(seed)."""
    engine = nn.Mt19937(seed)
    while True:
        yield engine.modulo(4294967296)


def make_marked_sequences(n, seed, offset):
    inputs = [0.0] * (n * 24)
    lengths, labels, ids, uniforms = [], [], [], []
    draws = raw_draws(seed)
    for b in range(n):
        u = [(next(draws) + 0.5) / 4294967296.0 for _ in range(14)]
        uniforms += u
        length = 4 + int(9 * u[0])
        cue = int((length - 1) * u[1])
        label = b % 2
        lengths.append(length)
        labels.append(label)
        ids.append(offset + b)
        for t in range(length):
            inputs[b * 24 + t * 2] = -1.0 if u[t + 2] < 0.5 else 1.0
        inputs[b * 24 + cue * 2] = 1.0 if label else -1.0
        inputs[b * 24 + cue * 2 + 1] = 1.0
    return MarkedSequences(nn.from_vector(inputs, [n, 12, 2]), int32_tensor(lengths), int32_tensor(labels),
                           int32_tensor(ids), nn.from_vector_double(uniforms, [n, 14]))


def marked_rows(x, rows):
    rows = list(rows)
    return MarkedSequences(nn.take_rows(x.inputs, rows), nn.take_rows(x.lengths, rows), nn.take_rows(x.labels, rows),
                           nn.take_rows(x.source_ids, rows), nn.take_rows(x.uniforms, rows))


def copy(x):
    return marked_rows(x, range(x.size()))


def save_data(path, x):
    nn.save_npz(str(path), [("inputs", x.inputs), ("lengths", x.lengths), ("labels", x.labels),
                            ("source_ids", x.source_ids), ("uniforms", x.uniforms)])


def load_data(path):
    a = nn.load_npz(str(path))
    find = recurrent_model.find
    return MarkedSequences(find(a, "inputs"), find(a, "lengths"), find(a, "labels"), find(a, "source_ids"),
                           find(a, "uniforms"))


def marked_pools(arrays=None):
    """The three splits (512 fit / 128 validation / 128 test); appends their data.npz arrays to `arrays`."""
    pools = []
    for k, split in enumerate(("fit", "validation", "test")):
        x = make_marked_sequences(128 if k else 512, 26001 + 2 * k, 10000 * k)
        pools.append(x)
        if arrays is not None:
            arrays += [(split + "_inputs", x.inputs), (split + "_lengths", x.lengths), (split + "_labels", x.labels),
                       (split + "_source_ids", x.source_ids), (split + "_uniforms", x.uniforms)]
    return pools


def predictions(logits):
    """1 where logit 1 exceeds logit 0, per row."""
    l = logits.values()
    return [int(l[b * 2 + 1] > l[b * 2]) for b in range(len(l) // 2)]


def accuracy(logits, labels):
    """Fraction of rows whose logit 1 exceeds logit 0 exactly when the label is 1."""
    y = labels.values()
    return sum(p == (v != 0) for p, v in zip(predictions(logits), y)) / len(y)


def heatmap(path, x, gradients=False):
    """tflow's Heatmap: first 24 sequences, 12 columns of 29x15 cells; blue = -1,
    red = +1, white = 0, black strip = marker; gradients are scaled by the global
    peak |value| of channel 0."""
    n = min(24, x.shape()[0])
    v = x.values()
    rgb = [220.0] * (n * 16 * 360 * 3)

    def put(y, col, r, g, blue):
        p = (y * 360 + col) * 3
        rgb[p], rgb[p + 1], rgb[p + 2] = r, g, blue

    peak = 1.0
    if gradients:
        peak = max([0.0] + [abs(v[(b * 12 + t) * 2]) for b in range(n) for t in range(12)])
        peak = max(peak, 1e-12)
    for b in range(n):
        for t in range(12):
            value = v[(b * 12 + t) * 2] / peak
            r = math.floor(255 * (1 - max(0.0, -value)) + 0.5)
            g = math.floor(255 * (1 - abs(value)) + 0.5)
            blue = math.floor(255 * (1 - max(0.0, value)) + 0.5)
            for y in range(15):
                for col in range(29):
                    put(b * 16 + y, t * 30 + col, r, g, blue)
            if not gradients and v[(b * 12 + t) * 2 + 1] > 0:
                for y in range(3):
                    for col in range(29):
                        put(b * 16 + y, t * 30 + col, 0, 0, 0)
    nn.save_png(nn.from_vector(rgb, [n * 16, 360, 3]), str(path), 0, 255)
