"""Fully convolutional time-series classifier and GunPoint data (examples/nn/common/time_series.hpp).

8/5/3 temporal kernels, batch normalization, ReLU, global average pooling and
a binary head, fit-only saved global normalization, BN calibration/frozen
modes, the GunPoint reader and Euclidean 1-NN / class-centroid baselines.
No training loops live here.
"""
import math
from dataclasses import dataclass, field
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import helper

checks = helper("checks")
replay = helper("replay")
sentence = helper("sentence")
f32, fmt = sentence.f32, sentence.fmt


@dataclass
class SeriesConfig:
    base: int = 128
    mean: float = 0.0
    standard_deviation: float = 1.0


@dataclass
class SeriesOutput:
    logits: object = None
    normalized: object = None
    features: object = None
    stages: list = field(default_factory=list)


class BnMode:
    """How series_cnn treats batch normalization: frozen moving statistics, batch
    statistics (optionally updating the moving averages), or tflow's calibration
    (each layer's moving mean / variance set to the population statistics)."""
    Infer, Train, TrainNoUpdate, Calibrate = range(4)


def series_cnn(p, state, signal, config=None, mode=BnMode.Infer):
    """signal [N, T, 1]. Parameters "conv<i>/W|b", "bn<i>/gamma|beta", "head/W|b";
    state "normalization/mean|std" (scalars) and "bn<i>/moving_mean|moving_var"."""
    config = config or SeriesConfig()
    if (config.base < 1 or not math.isfinite(config.mean) or not math.isfinite(config.standard_deviation)
            or config.standard_deviation <= 0):
        raise ValueError("time-series configuration")
    out = SeriesOutput()
    mean = state.get("normalization/mean", [], nn.init.constant(config.mean))
    stddev = state.get("normalization/std", [], nn.init.constant(config.standard_deviation))
    out.normalized = nn.div(nn.sub(signal, mean.detach()), stddev.detach())
    x = out.normalized
    for layer, kernel in enumerate((8, 5, 3)):
        width = config.base * (2 if layer == 1 else 1)
        bn = "bn" + str(layer + 1)
        x = nn.conv1d(p, "conv" + str(layer + 1), x, width, kernel)
        options = nn.BatchNormOptions()
        options.momentum = .99
        options.eps = 1e-3
        options.update_stats = mode == BnMode.Train
        if mode == BnMode.Calibrate:
            nn.batch_norm(p, state, bn, x, False, options)  # creates the tensors on first use
            with nn.no_grad():
                population_mean = nn.mean(x, [0, 1])
                population_variance = nn.mean(nn.square(nn.sub(x, population_mean)), [0, 1])
                state.set(bn + "/moving_mean", population_mean.detach())
                state.set(bn + "/moving_var", population_variance.detach())
        x = nn.relu(nn.batch_norm(p, state, bn, x, mode in (BnMode.Train, BnMode.TrainNoUpdate), options))
        out.stages.append(x)
    out.features = x
    out.logits = nn.dense(p, "head", nn.mean(x, [1]), 2)
    return out


# ---- data ----------------------------------------------------------------------------------

@dataclass
class SeriesData:
    signals: object
    labels: object
    source_ids: object

    def size(self):
        return int(self.labels.shape()[0])


def series_rows(d, rows):
    return SeriesData(sentence.select(d.signals, rows), sentence.select(d.labels, rows),
                      sentence.select(d.source_ids, rows))


def load_series(root):
    """GunPoint fit/validation/test pools and the fit-only global normalization:
    returns (pools, SeriesConfig(16, mean, std))."""
    arrays = replay.to_map(nn.load_npz(str(Path(root) / "time_series.npz")))
    seen = set()
    pools = []
    for split in ("fit", "validation", "test"):
        d = SeriesData(sentence.npz(arrays, split + "_signals"), sentence.npz(arrays, split + "_labels"),
                       sentence.npz(arrays, split + "_source_ids"))
        n = int(d.signals.shape()[0])
        checks.require(n > 0 and d.signals.dtype() == nn.DType.Float32 and list(d.signals.shape()) == [n, 150, 1],
                       "GunPoint raw univariate signal shape")
        for t in (d.labels, d.source_ids):
            checks.require(t.dtype() == nn.DType.Int32 and list(t.shape()) == [n], "aligned series labels/source IDs")
        labels, ids = sentence.ints(d.labels), sentence.ints(d.source_ids)
        positive = 0
        for i in range(n):
            checks.require(0 <= labels[i] < 2 and 0 <= ids[i] < 200 and ids[i] not in seen,
                           "series source IDs unique/disjoint,label range")
            seen.add(ids[i])
            checks.require(ids[i] >= 50 if split == "test" else ids[i] < 50, "official train/test source boundary")
            positive += labels[i]
        checks.require(all(math.isfinite(v) for v in d.signals.values()), "finite observed time samples")
        print(f" GunPoint {split}:{n} signals,point={positive}", flush=True)
        pools.append(d)
    x = pools[0].signals.values()
    mean = 0.0
    for v in x:
        mean += v
    mean /= len(x)
    variance = 0.0
    for v in x:
        variance += (v - mean) ** 2
    variance /= len(x)
    checks.require(variance > 1e-12, "nonconstant fitting signal population")
    print(f" fitting-only global normalization mean={fmt(mean)},std={fmt(math.sqrt(variance))}", flush=True)
    return pools, SeriesConfig(16, f32(mean), f32(math.sqrt(variance)))


# ---- model -------------------------------------------------------------------------------------

class SeriesModel:
    """Parameters (seed 419), BN/normalization state (421) and Adam .001 when training."""

    def __init__(self, config=None, training=True):
        self.parameters = nn.ParameterStore(419)
        self.state = nn.ParameterStore(421)
        self.config = config or SeriesConfig(16, 0.0, 1.0)
        with nn.no_grad():
            series_cnn(self.parameters, self.state, nn.zeros([1, 150, 1]), self.config)
        self.optimizer = nn.Optimizer(self.parameters, .001, replay.adam()) if training else None

    def forward(self, d, mode):
        return series_cnn(self.parameters, self.state, d.signals, self.config, mode)

    def calibrate_fit(self, fitting):
        """Sequential fitting-only population BN calibration, without labels or gradients."""
        with nn.no_grad():
            self.forward(fitting, BnMode.Calibrate)

    def loss(self, d):
        """Frozen-BN evaluation CE (no gradient)."""
        with nn.no_grad():
            value = sentence.sentence_ce(self.forward(d, BnMode.Infer).logits, d.labels).item()
        checks.require(math.isfinite(value) and value >= 0, "finite time-series CE")
        return value

    def predict(self, d):
        with nn.no_grad():
            return self.forward(d, BnMode.Infer).logits.detach()

    def snapshot(self):
        return sentence.snapshot([self.parameters, self.state])

    def save(self, path):
        sentence.save_stores(path, self.parameters, self.state)

    def restore(self, path):
        sentence.restore_stores(path, self.parameters, self.state)

    def export(self, path):
        sentence.export_arrays(path, self.parameters, self.state)


def series_metrics(name, d, logits):
    sentence.sentence_metrics(name, d.labels, logits)


def series_reference(fit, test, centroid):
    """Euclidean 1-NN one-hot pseudo-logits or negative class-centroid MSE [N, 2]."""
    T = 150
    fx, tx, fy = fit.signals.values(), test.signals.values(), sentence.ints(fit.labels)
    centers = [[0.0] * T for _ in range(2)]
    count = [0, 0]
    for n in range(fit.size()):
        count[fy[n]] += 1
        row = centers[fy[n]]
        for t in range(T):
            row[t] += fx[n * T + t]
    for c in range(2):
        checks.require(count[c] > 0, "both fitting classes for baselines")
        centers[c] = [x / count[c] for x in centers[c]]
    out = [0.0] * (2 * test.size())
    for n in range(test.size()):
        query = tx[n * T:(n + 1) * T]
        if centroid:
            for c in range(2):
                distance = 0.0
                for t in range(T):
                    difference = query[t] - centers[c][t]
                    distance += difference * difference
                out[2 * n + c] = f32(-distance / T)
        else:
            best, prediction = math.inf, 0
            for i in range(fit.size()):
                distance = 0.0
                base = i * T
                for t in range(T):
                    difference = query[t] - fx[base + t]
                    distance += difference * difference
                if distance < best:
                    best, prediction = distance, fy[i]
            out[2 * n] = 1.0 if prediction == 0 else 0.0
            out[2 * n + 1] = 1.0 if prediction == 1 else 0.0
    return nn.from_vector(out, [test.size(), 2])


def reversed_series(d):
    """d with axis 1 reversed (the tests' time-reversal control)."""
    return SeriesData(nn.reverse(d.signals, [1]).contiguous(), d.labels, d.source_ids)
