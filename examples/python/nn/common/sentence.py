"""MR sentence data, Kim CNN-rand model wrapper and the generic 1-D convolution helpers
(examples/nn/common/sentence.hpp): row selection, stable CE, metrics, confusion
matrices, fixture parameter assignment, host finite differences and store
persistence. No training loops live here.
"""
import math
import shutil
import struct
from array import array
from dataclasses import dataclass
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import dataset_assets
from pydatamunge.nn_workflows import helper

checks = helper("checks")
replay = helper("replay")
kim = helper("kim_cnn")
KimConfig = kim.KimConfig


# ---- formatting / float32 emulation ------------------------------------------------------

def fmt(value):
    """std::cout << value for a double/float (default precision 6, %g)."""
    if isinstance(value, int):
        return str(value)
    return format(float(value), "g")


def f32(value):
    """static_cast<float>(value) as a Python float."""
    return struct.unpack("f", struct.pack("f", value))[0]


def f32_list(values):
    """Element-wise static_cast<float>."""
    return list(array("f", values))


def ints(tensor):
    return [int(v) for v in tensor.values()]


def int_tensor(values, shape):
    return nn.cast(nn.from_vector_int64([int(v) for v in values], list(shape)), nn.DType.Int32)


def quoted(path):
    """std::cout << std::filesystem::path."""
    return '"' + str(path) + '"'


# ---- generic helpers ----------------------------------------------------------------------

def select(x, rows):
    return nn.take_rows(x, [int(r) for r in rows])


def range_rows(begin, end):
    return list(range(int(begin), int(end)))


def fix_parameters(p, value):
    """Overwrites every parameter of p (creation order v, flat index i) with
    value(v, i, shape) rounded to float32; returns the values as lists of floats."""
    weights = []
    names = p.names()
    for v, tensor in enumerate(p.parameters()):
        shape = [int(s) for s in tensor.shape()]
        count = 1
        for s in shape:
            count *= s
        data = f32_list([value(v, i, shape) for i in range(count)])
        p.set(names[v], nn.cast(nn.from_vector(data, shape), p.dtype()))
        weights.append(data)
    return weights


def host_gradient_error(weights, gradients, objective):
    """max |central difference of objective (step 1e-5) - gradient| over every
    coordinate; returns (error, coordinates)."""
    error = 0.0
    coordinates = 0
    for v, values in enumerate(weights):
        g = gradients[v].values()
        for i in range(len(values)):
            plus = [list(w) for w in weights]
            minus = [list(w) for w in weights]
            plus[v][i] += 1e-5
            minus[v][i] -= 1e-5
            error = max(error, abs((objective(plus) - objective(minus)) / 2e-5 - g[i]))
            coordinates += 1
    return error, coordinates


def squared_norms(gradients):
    out = []
    for g in gradients:
        s = 0.0
        for x in g.values():
            s = math.fma(x, x, s)
        out.append(s)
    return out


def require_live_gradients(p, gradients, what, may_vanish=None, floor=0.0):
    """Every coordinate finite and every tensor nonzero (unless may_vanish(name));
    returns the number of nonzero tensors."""
    nonzero = 0
    names = p.names()
    for v, gradient in enumerate(gradients):
        norm = 0.0
        for x in gradient.values():
            checks.require(math.isfinite(x), "finite " + what + " gradient coordinate")
            norm = math.fma(x, x, norm)
        if not (may_vanish and may_vanish(names[v])):
            checks.require(norm > floor, "nonzero " + what + " gradient: " + names[v])
        nonzero += norm > floor
    return nonzero


def sentence_ce(logits, labels):
    """-mean_n sum_c onehot(labels) log_softmax(logits) (tflow's SentenceCE)."""
    target = nn.one_hot(labels, int(logits.shape()[-1]), 1.0, 0.0, logits.dtype())
    return nn.neg(nn.mean(nn.sum(nn.mul(target, nn.log_softmax(logits)), [1])))


def npz(arrays, key):
    return replay.get(arrays, key)


def snapshot(stores):
    """Clones of every tensor (tflow's Snapshot); compare with replay.same_bits."""
    out = {}
    for k, store in enumerate(stores):
        for name in store.names():
            out[str(k) + ":" + name] = store.at(name).detach().clone()
    return out


def save_stores(path, p, s=None):
    """Parameters (and optional state) as one .npz with "p:"/"s:" prefixes."""
    arrays = [("p:" + n, p.at(n).detach()) for n in p.names()]
    if s is not None:
        arrays += [("s:" + n, s.at(n).detach()) for n in s.names()]
    nn.save_npz(str(path), arrays)


def restore_stores(path, p, s=None):
    arrays = nn.load_npz(str(path))
    checks.require(len(arrays) == p.size() + (s.size() if s is not None else 0),
                   "checkpoint covers exactly the model tensors")
    for key, value in arrays:
        store = p if key.startswith("p:") else s
        name = key[2:]
        checks.require(store.contains(name), "checkpoint tensor " + name + " belongs to the model")
        store.set(name, nn.cast(value, store.dtype()))


def export_arrays(path, p, s=None):
    """Parameters as "p<i>" and state as "s<i>" (tflow's model_arrays.npz layout)."""
    arrays = [("p" + str(i), t.detach()) for i, t in enumerate(p.parameters())]
    if s is not None:
        arrays += [("s" + str(i), t.detach()) for i, t in enumerate(s.parameters())]
    nn.save_npz(str(path), arrays)


def copy_if_exists(source, destination):
    if Path(source).exists():
        shutil.copyfile(source, destination)


# ---- MR sentence data ---------------------------------------------------------------------

@dataclass
class SentenceData:
    tokens: object
    lengths: object
    labels: object
    source_ids: object

    def size(self):
        return int(self.labels.shape()[0])


def sentence_rows(d, rows):
    return SentenceData(select(d.tokens, rows), select(d.lengths, rows), select(d.labels, rows),
                        select(d.source_ids, rows))


def load_sentences(root):
    """Returns (pools [fit, validation, test], KimConfig) from <root>/sentence.npz."""
    dataset_assets.EnsureSentences(str(root))
    arrays = replay.to_map(nn.load_npz(str(Path(root) / "sentence.npz")))
    config_tensor = npz(arrays, "config")
    checks.require(config_tensor.dtype() == nn.DType.Int32 and list(config_tensor.shape()) == [2],
                   "sentence vocabulary metadata")
    cfg = ints(config_tensor)
    config = KimConfig(cfg[0], cfg[1], 32, 32)
    checks.require(2 <= config.vocabulary <= 8000 and config.time == 64, "sentence fit vocabulary/time bounds")
    seen = set()
    pools = []
    for split in ("fit", "validation", "test"):
        d = SentenceData(npz(arrays, split + "_tokens"), npz(arrays, split + "_lengths"),
                         npz(arrays, split + "_labels"), npz(arrays, split + "_source_ids"))
        n = int(d.tokens.shape()[0])
        checks.require(n > 0 and d.tokens.dtype() == nn.DType.Int32 and list(d.tokens.shape()) == [n, 64],
                       "sentence token matrix")
        for t in (d.lengths, d.labels, d.source_ids):
            checks.require(t.dtype() == nn.DType.Int32 and list(t.shape()) == [n], "aligned sentence metadata")
        tokens, lengths, labels, sources = ints(d.tokens), ints(d.lengths), ints(d.labels), ints(d.source_ids)
        positive = 0
        for i in range(n):
            length, label, source = lengths[i], labels[i], sources[i]
            checks.require(1 <= length <= 64 and 0 <= label < 2 and 0 <= source < 10662 and source not in seen,
                           "valid unique sentence ID/label/length")
            seen.add(source)
            positive += label
            for j in range(64):
                token = tokens[i * 64 + j]
                checks.require((1 <= token < config.vocabulary) if j < length else token == 0,
                               "sentence legal tokens/right PAD")
        print(f" MR {split}:{n} sentences,positive={positive}", flush=True)
        pools.append(d)
    return pools, config


# ---- metrics and baselines ------------------------------------------------------------------

def sentence_metrics(name, labels_tensor, logits):
    n = int(labels_tensor.shape()[0])
    checks.require(list(logits.shape()) == [n, 2], "sentiment score shape")
    z = logits.values()
    labels = ints(labels_tensor)
    correct = 0
    confusion = [[0, 0], [0, 0]]
    nll = 0.0
    for i in range(n):
        a, b = z[2 * i], z[2 * i + 1]
        checks.require(math.isfinite(a) and math.isfinite(b), "finite binary logits")
        label, prediction = labels[i], int(b > a)
        confusion[label][prediction] += 1
        correct += label == prediction
        m = max(a, b)
        nll += m + math.log(math.exp(a - m) + math.exp(b - m)) - (b if label else a)
    print(f" {name}:accuracy={fmt(100.0 * correct / n)}%,CE={fmt(nll / n)},confusion={confusion[0][0]}/"
          f"{confusion[0][1]}/{confusion[1][0]}/{confusion[1][1]}")


def sentence_nb(fit, test, vocabulary):
    """Fit-only add-one unigram multinomial naive Bayes log scores [N, 2]."""
    count = [[1.0] * vocabulary for _ in range(2)]
    total = [float(vocabulary), float(vocabulary)]
    prior = [1.0, 1.0]
    ft, fl, fy = ints(fit.tokens), ints(fit.lengths), ints(fit.labels)
    T = int(fit.tokens.shape()[1])
    for n in range(fit.size()):
        label = fy[n]
        prior[label] += 1
        for t in range(fl[n]):
            count[label][ft[n * T + t]] += 1
            total[label] += 1
    tt, tl = ints(test.tokens), ints(test.lengths)
    out = []
    for n in range(test.size()):
        for c in range(2):
            score = math.log(prior[c] / (prior[0] + prior[1]))
            for t in range(tl[n]):
                score += math.log(count[c][tt[n * T + t]] / total[c])
            out.append(score)
    return nn.from_vector(out, [test.size(), 2])


def positive(logits):
    """1 when logits[n,1] > logits[n,0]."""
    z = logits.values()
    return [int(z[2 * n + 1] > z[2 * n]) for n in range(len(z) // 2)]


def confusion(labels_tensor, predictions, classes):
    """cells[truth * classes + prediction] counts."""
    labels = ints(labels_tensor)
    checks.require(len(labels) == len(predictions), "confusion labels/predictions alignment")
    cells = [0] * (classes * classes)
    for label, prediction in zip(labels, predictions):
        cells[label * classes + int(prediction)] += 1
    return cells


def print_confusion(name, cells, classes):
    """Prints a confusion matrix (rows = truth, columns = prediction)."""
    print(f" Confusion matrix {name} (rows=truth, columns=prediction):")
    print(" truth/pred" + "".join(f" {c:5d}" for c in range(classes)))
    for r in range(classes):
        print(f" {r:10d}" + "".join(f" {int(cells[r * classes + c]):5d}" for c in range(classes)))


# ---- Kim CNN-rand model ---------------------------------------------------------------------

class SentenceModel:
    """Parameters (seed 317), Kim's Adadelta when training and the dropout stream (313)."""

    def __init__(self, config, training=True):
        self.parameters = nn.ParameterStore(317)
        self.config = config
        self.dropout = nn.Generator(313)
        with nn.no_grad():
            kim.kim_cnn(self.parameters, nn.zeros([1, config.time], nn.DType.Int32),
                        nn.full([1], 1, nn.DType.Int32), config)
        self.optimizer = kim.KimAdadelta(self.parameters) if training else None

    def forward(self, d, training):
        return kim.kim_cnn(self.parameters, d.tokens, d.lengths, self.config, training, .5, self.dropout)

    def loss(self, d):
        """Inference (dropout-free, no gradient) CE."""
        with nn.no_grad():
            value = sentence_ce(self.forward(d, False).logits, d.labels).item()
        checks.require(math.isfinite(value) and value >= 0, "finite sentence CE")
        return value

    def predict(self, d):
        with nn.no_grad():
            return self.forward(d, False).logits.detach()

    def predict_all(self, d):
        """Predictions in batches of 50."""
        parts = [self.predict(sentence_rows(d, range_rows(start, min(start + 50, d.size()))))
                 for start in range(0, d.size(), 50)]
        return nn.concat(parts, 0)

    def snapshot(self):
        return snapshot([self.parameters])

    def save(self, path):
        save_stores(path, self.parameters)

    def restore(self, path):
        restore_stores(path, self.parameters)
