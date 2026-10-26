"""Image-classifier training pieces with per-image random crop/flip augmentation
(examples/nn/common/image_training.hpp).

Images stay NHWC; pixels map from [0,1] to [-1,1]. Training can zero-pad by four
pixels, independently crop back and flip horizontally (nn.augment.random_pad_crop
from nn.Generator(151)). Evaluation only normalizes. The objective is (optionally
label-smoothed) mean sparse cross-entropy plus 0.5 * l2 * sum(W^2) over parameters
of rank >= 2, with optional weighted auxiliary heads, extra penalties or a custom
loss (TrainingNetwork). Model builders have the signature
model(params, state, x, training) -> logits; `state` holds batch-norm statistics.
Training loops, checkpoint selection and reporting stay in each example.
"""
import math
import os
from pathlib import Path
from types import SimpleNamespace

from pydatamunge import nn


def image_batch(dataset, rows, augmentation=None):
    """Normalized NHWC float images and int64 labels for `rows`; with a generator,
    pad-4 random crops and random horizontal flips per image."""
    if dataset.images.dim() != 4 or dataset.images.numel() == 0:
        raise ValueError("image batching requires nonempty NHWC data")
    rows = [int(r) for r in rows]
    count = int(dataset.images.shape()[0])
    for row in rows:
        if row < 0 or row >= count:
            raise IndexError("image batch row")
    x = nn.take_rows(dataset.images, rows)
    if augmentation is not None:
        x = nn.augment.random_pad_crop(x, 4, augmentation, True, 0.0)
    return nn.sub(nn.mul(x, 2.0), 1.0), nn.take_rows(dataset.labels, rows)


def evaluate_images(params, state, model, dataset, rows, classes):
    """Cross-entropy, accuracy and confusion of inference-mode parameters over `rows`, in batches of 256.

    Returns a namespace of WeightedMean `loss`/`accuracy` and a ConfusionMatrix `confusion`."""
    result = SimpleNamespace(loss=nn.WeightedMean(), accuracy=nn.WeightedMean(), confusion=nn.ConfusionMatrix(classes))
    rows = [int(r) for r in rows]
    with nn.no_grad():
        for start in range(0, len(rows), 256):
            ids = rows[start:start + 256]
            x, y = image_batch(dataset, ids)
            logits = model(params, state, x, False)
            prediction = nn.argmax(logits, 1)
            result.loss.add(nn.sparse_softmax_cross_entropy(logits, y).item(), len(ids))
            result.accuracy.add(nn.accuracy(prediction, y), len(ids))
            result.confusion.add(y, prediction)
    return result


class ImageTrainingConfig:
    """image_training::ImageTrainingConfig (maximum_fit_rows=0 uses every non-validation row)."""

    def __init__(self, **values):
        self.name = ""
        self.classes, self.epochs, self.batch, self.validation_rows = 10, 10, 128, 5000
        self.maximum_fit_rows = 0
        self.learning_rate, self.l2, self.label_smoothing = 0.001, 0.0001, 0.0
        self.augment = True
        self.minimum_test_accuracy = 0.55
        self.seed = 17
        for key, value in values.items():
            if not hasattr(self, key):
                raise ValueError("unknown image training setting: " + key)
            setattr(self, key, value)


def training_cross_entropy(logits, labels, classes, smoothing):
    """Mean sparse cross-entropy with optional label smoothing."""
    if smoothing <= 0:
        return nn.sparse_softmax_cross_entropy(logits, labels)
    targets = nn.add(nn.mul(nn.one_hot(labels, classes, 1.0, 0.0, logits.dtype()), 1 - smoothing), smoothing / classes)
    return nn.neg(nn.mean(nn.sum(nn.mul(targets, nn.log_softmax(logits, 1)), [1])))


def training_network(logits, auxiliary=(), penalties=(), loss=None):
    """TrainingNetwork: training logits, [(aux_logits, weight)], extra scalar penalties, optional loss(labels)."""
    return SimpleNamespace(logits=logits, auxiliary=list(auxiliary), penalties=list(penalties), loss=loss)


def l2_penalty(params):
    """sum(W^2) over parameters of rank >= 2 in creation order, or None when there are none."""
    penalty = None
    for p in params.parameters():
        if p.dim() >= 2:
            term = nn.sum(nn.square(p))
            penalty = term if penalty is None else nn.add(penalty, term)
    return penalty


def training_objective(params, state, model, x, y, cfg, training=None):
    """(Smoothed) cross-entropy of the training logits, plus weight * cross-entropy of every
    auxiliary head, plus penalties, plus 0.5 * l2 * sum(W^2). `training(params, state, x)`
    optionally returns a training_network(); otherwise model(..., True) gives the logits."""
    network = training(params, state, x) if training is not None else training_network(model(params, state, x, True))
    objective = (network.loss(y) if network.loss is not None
                 else training_cross_entropy(network.logits, y, cfg.classes, cfg.label_smoothing))
    for logits, weight in network.auxiliary:
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("auxiliary loss weight must be positive and finite")
        objective = nn.add(objective, nn.mul(training_cross_entropy(logits, y, cfg.classes, cfg.label_smoothing), weight))
    for penalty in network.penalties:
        objective = nn.add(objective, penalty)
    if cfg.l2 > 0:
        penalty = l2_penalty(params)
        if penalty is not None:
            objective = nn.add(objective, nn.mul(penalty, 0.5 * cfg.l2))
    return objective


def cifar_option_defaults(data):
    """Option defaults of image_training::ParseCifarOption, for pydatamunge.nn_workflows.options()."""
    return dict(data=str(data), epochs=10, batch=128, lr=0.001, fit=0, augment=1, threads=0, seed=17)


def apply_cifar_options(args, cfg):
    """Copies parsed ParseCifarOption values into cfg (threads=0 keeps nn's default thread count)."""
    if args["augment"] not in (0, 1):
        raise ValueError("augment must be 0 or 1")
    cfg.epochs, cfg.batch, cfg.learning_rate = int(args["epochs"]), int(args["batch"]), float(args["lr"])
    cfg.maximum_fit_rows, cfg.augment, cfg.seed = int(args["fit"]), args["augment"] == 1, int(args["seed"])
    if args["threads"]:
        nn.set_num_threads(int(args["threads"]))
    return cfg


def load_cifar10(directory):
    """The five official development batches and the separate test batch."""
    directory = Path(directory)
    if not (directory / "data_batch_1.bin").exists():
        raise RuntimeError("CIFAR-10 missing in " + str(directory) +
                           "; run python3 scripts/fetch_datasets.py cifar10 --split both in tflow_examples, or set data=")
    development = nn.load_cifar10(str(directory), "train")
    test = nn.load_cifar10(str(directory), "test")
    if list(development.images.shape()) != [50000, 32, 32, 3] or list(test.images.shape()) != [10000, 32, 32, 3]:
        raise RuntimeError("official CIFAR-10 split sizes")
    return development, test
