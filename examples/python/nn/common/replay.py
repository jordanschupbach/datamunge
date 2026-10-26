"""Named-array helpers for replaying recorded runs (examples/nn/common/tf_replay.hpp)."""
import shutil

from pydatamunge import nn

from pydatamunge.nn_workflows import helper

checks = helper("checks")
tf_models = checks.tf_models


def to_map(arrays):
    """load_npz result -> {name: tensor}."""
    return {pair[0]: pair[1] for pair in arrays} if not isinstance(arrays, dict) else dict(arrays)


def get(arrays, key):
    mapping = to_map(arrays)
    if key not in mapping:
        raise RuntimeError("missing array " + key)
    return mapping[key]


def rows(mapping, indices):
    return {k: nn.take_rows(v, list(indices)) for k, v in mapping.items()}


def split(arrays, name, keys):
    return {k: get(arrays, name + "_" + k) for k in keys}


def max_abs_diff(a, b):
    x, y = a.values(), b.values()
    checks.require(len(x) == len(y), "compared tensors have the same size")
    return max((abs(p - q) for p, q in zip(x, y)), default=0.0)


def same_bits(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same_bits(v, b[k]) for k, v in a.items())
    return a.dtype() == b.dtype() and list(a.shape()) == list(b.shape()) and a.values() == b.values()


def arrays(store):
    return {name: store.at(name).detach().clone() for name in store.names()}


def named(store, gradients):
    values = store.parameters()
    return {name: (g.detach() if g.defined() else nn.zeros_like(values[i]).detach())
            for i, (name, g) in enumerate(zip(store.names(), gradients))}


def load_into(store, path):
    loaded = to_map(nn.load_npz(str(path)))
    checks.require(len(loaded) == store.size(), "array file covers exactly the model's parameters: " + str(path))
    for name, value in loaded.items():
        checks.require(store.contains(name), "array " + name + " is a model parameter")
        store.set(name, nn.cast(value, store.at(name).dtype()))


def max_diff(a, b):
    if not isinstance(b, dict):
        b = to_map(nn.load_npz(str(b)))
    result = 0.0
    for k, v in a.items():
        checks.require(k in b, "reference has array " + k)
        result = max(result, max_abs_diff(v, b[k]))
    return result


def history(path, column=1):
    lines = open(path).read().splitlines()[1:]
    return [float(line.split("\t")[column]) for line in lines if line]


def adam():
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    return config


def copy_assets(source, destination, files):
    for name in files:
        if (source / name).exists():
            shutil.copyfile(source / name, destination / name)
