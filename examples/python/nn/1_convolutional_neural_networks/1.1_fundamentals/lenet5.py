# Native Python NN example. LeNet-5 architecture checks and MNIST fitting run in Python.
"""CMD: just python-example lenet5
LeNet-5: sparse convolutions, learned subsampling and a fixed RBF head.

Port of tflow's lenet5. Preparation (in ~/projects/tflow_examples):
python3 scripts/fetch_datasets.py mnist --split both.
Preserve the 1998 architecture's C1/S2/C3/S4/C5/F6 dimensions, 60 sparse C3
connections, scaled tanh activations and exactly 60,000 learned parameters.
Use our own fixed seven-segment 7x12 digit prototypes for the RBF output.
Adam and cross entropy on negative squared distances are modern training
adaptations; this is not a reproduction of the original training recipe.
Reference: https://cognitivemedium.com/assets/rmnist/LeCun1998.pdf

Check stage shapes/counts, C3's absent-channel derivatives and the RBF
classifier's prototype predictions before fitting the official MNIST data.
Fit 55k rows, select on 5k validation rows, then evaluate all 10k test rows.
Default eight epochs, lr=.001, batch=256; require 97.5% test accuracy.
Weights initialise from nn's generator (seed 17), so the trajectory matches
TF only statistically. Options: data=PATH epochs=8 batch=256 lr=.001 threads=N.
Quick mode uses 64 fit / 32 validation / 32 test rows, batch <= 16 and one
epoch unless epochs= is given. Checkpoint and validation history: output/nn/models/lenet5/.
"""
import math
import os
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
lenet = helper("lenet")
mnist = helper("mnist")


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def architecture_checks():
    p = nn.ParameterStore(17)
    pixels = nn.ones([2, 784])
    network = lenet.lenet5(p, pixels)
    checks.require(p.count() == 60000, "LeNet-5 has exactly 60000 learned parameters")
    names = ("c1", "s2", "c3", "s4", "c5", "f6")
    counts = (156, 12, 1516, 32, 48120, 10164)
    for stage, expected in zip(names, counts):
        count = sum(p.at(name).numel() for name in p.names() if name.startswith(stage + "/"))
        checks.require(count == expected, "LeNet stage parameter count: " + stage)
    connections = sum(len(channels) for channels in lenet.lenet_c3_connections())
    checks.require(connections == 60, "C3 has 60 channel connections")
    # Map 0 reads channels 0,1,2 only. Its derivative must be exactly zero for
    # channels 3,4,5, while at least one derivative in each connected map lives.
    c3_input = nn.full([1, 14, 14, 6], 0.2)
    c3_input.requires_grad_()
    c3 = lenet.lenet_c3(p, "c3", c3_input)
    probe = nn.sum(nn.slice_axis(c3, 3, 0, 1))
    gradient = nn.grad(probe, [c3_input])[0].values()
    prototype_logits = lenet.lenet_rbf_logits(lenet.lenet_digit_codes())

    stages = [(network.image, [2, 32, 32, 1]), (network.c1, [2, 28, 28, 6]), (network.s2, [2, 14, 14, 6]),
              (network.c3, [2, 10, 10, 16]), (network.s4, [2, 5, 5, 16]), (network.c5, [2, 1, 1, 120]),
              (network.f6, [2, 84]), (network.logits, [2, 10])]
    for value, shape in stages:
        checks.require(list(value.shape()) == shape, "LeNet stage shape")
    checks.near(network.image.at([0, 0, 0, 0]), -0.1, 1e-7, "padding background normalization")
    checks.near(network.image.at([0, 2, 2, 0]), 1.175, 1e-7, "foreground normalization")
    magnitude = [0.0] * 6
    for i in range(14 * 14 * 6):
        magnitude[i % 6] += abs(gradient[i])
    for channel in range(6):
        if channel < 3:
            checks.require(magnitude[channel] > 0, "connected C3 channel has a gradient")
        else:
            checks.near(magnitude[channel], 0, 0, "disconnected C3 channel has no gradient")
    for a in range(10):
        for b in range(10):
            logit = prototype_logits.at([a, b])
            if a == b:
                checks.near(logit, 0, 0, "RBF self distance")
            else:
                checks.require(logit < 0, "distinct RBF prototypes classify themselves uniquely")
    print("  architecture checks: stage shapes/counts, sparse C3 derivatives and RBF prototypes passed")


def logits_of(store, x, training=False):
    return lenet.lenet5(store, x).logits


def main(arguments=None):
    architecture_checks()
    name, default_epochs, minimum_test_accuracy = "lenet5", 8, 0.975
    args = options(arguments, dict(data=str(checks.dataset("mnist")), epochs=0, batch=256, lr=0.001, threads=0,
                                   full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    os.environ["DATAMUNGE_EXAMPLES_FULL"] = "1" if args["full"] else "0"
    explicit_epochs = args["epochs"] != 0
    epochs = args["epochs"] if explicit_epochs else default_epochs
    batch, rate = args["batch"], args["lr"]
    if args["threads"]:
        nn.set_num_threads(args["threads"])
    checks.require(epochs > 0 and batch > 0 and math.isfinite(rate) and rate > 0,
                   "positive epochs, batch and learning rate required")
    if not explicit_epochs:
        epochs = checks.limit(epochs, 1)  # an explicit budget is honored
    batch = checks.limit(batch, 16)
    checks.describe()
    data = Path(args["data"])
    checks.require((data / "train-images-idx3-ubyte").exists() or (data / "train-images-idx3-ubyte.gz").exists(),
                   "MNIST missing in " + str(data) + "; run python3 scripts/fetch_datasets.py mnist --split both "
                   "in tflow_examples, or set data=")
    train, test = nn.load_mnist(str(data), "train"), nn.load_mnist(str(data), "test")
    checks.require(list(train.images.shape()) == [60000, 28, 28, 1] and list(test.images.shape()) == [10000, 28, 28, 1],
                   "official MNIST dimensions required")
    train_x, test_x = nn.reshape(train.images, [60000, 784]), nn.reshape(test.images, [10000, 784])

    order = mnist.development_order(60000, 2026)
    fit_rows, validation_rows = order[:55000], order[55000:]
    fit_rows = fit_rows[:checks.limit(len(fit_rows), 64)]
    validation_rows = validation_rows[:checks.limit(len(validation_rows), 32)]
    test_rows = list(range(checks.limit(10000, 32)))

    l2 = 1e-4
    store = nn.ParameterStore(17)
    with nn.no_grad():  # create every parameter before building the optimizer
        logits_of(store, nn.slice_axis(train_x, 0, 0, 1), True)
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(store, rate, config)
    weights = [p for p in store.parameters() if p.dim() >= 2]

    model_dir = checks.output_dir("models/" + name)
    checkpoint = str(model_dir / "best.npz")
    history = (model_dir / "history.tsv").open("w")
    history.write("epoch\tvalidation_loss\tvalidation_accuracy\n")
    print(f"== {name}: {len(fit_rows)} fit / {len(validation_rows)} validation / {len(test_rows)} test")
    print(f"  parameters={store.count()}; batch={batch}; lr={g(rate)}")

    batches = nn.BatchOrder(len(fit_rows), batch, 29)
    best, best_epoch = math.inf, -1
    started = time.perf_counter()
    for epoch in range(1, epochs + 1):
        epoch_start = time.perf_counter()
        batches.begin_epoch()
        samples = 0
        while True:
            ids = list(batches.next())
            if not ids:
                break
            samples += len(ids)
            ids = [fit_rows[i] for i in ids]
            x, y = nn.take_rows(train_x, ids), nn.take_rows(train.labels, ids)
            objective = nn.sparse_softmax_cross_entropy(logits_of(store, x, True), y)
            if l2 > 0:
                penalty = nn.sum(nn.square(weights[0]))
                for weight in weights[1:]:
                    penalty = nn.add(penalty, nn.sum(nn.square(weight)))
                objective = nn.add(objective, nn.mul(penalty, 0.5 * l2))
            optimizer.minimize(objective)
        checks.require(samples == len(fit_rows), "each selected training row used once per epoch")
        validation = mnist.evaluate(store, logits_of, train_x, train.labels, validation_rows)
        loss, accuracy = validation.loss.value(), validation.accuracy.value()
        if loss < best:
            best, best_epoch = loss, epoch
            store.save(checkpoint)
        history.write(f"{epoch}\t{g(loss)}\t{g(accuracy)}\n")
        seconds = time.perf_counter() - epoch_start
        print(f"  epoch {epoch}: validation NLL={g(loss)}, accuracy={g(accuracy)} ({seconds:.3g} s)", flush=True)
    train_seconds = time.perf_counter() - started
    history.close()
    store.load(checkpoint)
    restored = mnist.evaluate(store, logits_of, train_x, train.labels, validation_rows)
    checks.near(restored.loss.value(), best, 1e-7, "best checkpoint matches selected validation loss")
    result = mnist.evaluate(store, logits_of, test_x, test.labels, test_rows)
    print(f"  selected epoch={best_epoch}; test NLL={g(result.loss.value())}; accuracy={g(result.accuracy.value())}")
    cells = mnist.print_confusion(result.confusion, "test confusion (truth rows):")
    total = sum(sum(row) for row in cells)
    correct = sum(cells[i][i] for i in range(10))
    checks.require(total == len(test_rows), "all selected test images evaluated")
    checks.near(result.accuracy.value(), correct / len(test_rows), 1e-7, "confusion matches accuracy")
    checks.training_quality(result.accuracy.value() >= minimum_test_accuracy,
                            "classifier did not reach demonstration accuracy threshold")
    print(f"  training time {train_seconds:.1f} s ({train_seconds / epochs:.2f} s/epoch)")
    print(f"  best checkpoint and history: {model_dir}\nAll {name} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
