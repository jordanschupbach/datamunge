"""Named-tensor marshalling for the bound C++ memory models; no fitting workflow."""
from pydatamunge import nn


def records(batch):
    return sorted(batch.items()) if isinstance(batch, dict) else batch


def rows(batch, indices):
    return {key: nn.take_rows(value, list(indices)) for key, value in batch.items()}


def evaluate(model, batch):
    return dict(model.Evaluate(records(batch)))


def same(first, second):
    if isinstance(first, dict):
        return first.keys() == second.keys() and all(same(value, second[key]) for key, value in first.items())
    return nn.equal_bits(first, second)


def save(path, batch):
    nn.save_npz(str(path), records(batch))
