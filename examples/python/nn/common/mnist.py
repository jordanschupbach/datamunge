"""MNIST evaluation helpers for the example-local classifier workflows (examples/nn/common/mnist.hpp).

The official 60,000 training rows are split once with std::mt19937_64(2026)
(nn.Mt19937_64 reproduces the shuffle exactly). Metrics weight batch means by
row count. Training loops stay in each example script.
"""
from types import SimpleNamespace

from pydatamunge import nn


def development_order(count=60000, seed=2026):
    """0..count-1 shuffled by std::shuffle with std::mt19937_64(seed)."""
    return list(nn.Mt19937_64(seed).shuffle(list(range(count))))


def evaluate(store, logits_of, images, labels, rows):
    """mnist::Evaluate: cross-entropy, accuracy and confusion of fixed parameters over `rows`, in batches of 512.

    `logits_of(store, x, training)` builds [batch, 10] logits. Returns a namespace
    with WeightedMean `loss` and `accuracy` and a 10-class ConfusionMatrix `confusion`."""
    result = SimpleNamespace(loss=nn.WeightedMean(), accuracy=nn.WeightedMean(), confusion=nn.ConfusionMatrix(10))
    rows = [int(r) for r in rows]
    with nn.no_grad():
        for start in range(0, len(rows), 512):
            ids = rows[start:start + 512]
            x, y = nn.take_rows(images, ids), nn.take_rows(labels, ids)
            logits = logits_of(store, x, False)
            prediction = nn.argmax(logits, 1)
            result.loss.add(nn.sparse_softmax_cross_entropy(logits, y).item(), len(ids))
            result.accuracy.add(nn.accuracy(prediction, y), len(ids))
            result.confusion.add(y, prediction)
    return result


def confusion_cells(confusion):
    """ConfusionMatrix -> list of rows (truth) of integer counts (predictions)."""
    classes = confusion.classes()
    values = confusion.to_tensor().values()
    return [[int(values[i * classes + j]) for j in range(classes)] for i in range(classes)]


def print_confusion(confusion, title="test confusion (truth rows):"):
    """Prints the matrix in the C++ examples' layout (truth rows, prediction columns); returns the cells."""
    cells = confusion_cells(confusion)
    print("  " + title)
    for row in cells:
        print("   " + "".join(" " + str(v).rjust(4) for v in row))
    return cells
