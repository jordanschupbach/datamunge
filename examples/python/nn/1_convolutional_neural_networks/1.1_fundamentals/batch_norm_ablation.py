# Native Python NN example. Paired MNIST CNN batch-norm ablation trained and checked in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.1_fundamentals/batch_norm_ablation.py
"""Paired MNIST CNN experiments with and without batch normalization.

Port of tflow's batch_norm_ablation. Use the same 12,000 fitting rows,
3,000 validation rows, initial convolution and dense weights, six epochs,
batch order, Adam(.001) and kernel L2(.0001). Three initialization seeds
(11,23,47) produce paired runs. Batch norm adds gamma/beta after each
convolution, before ReLU, with momentum .9 and eps .001. Keep convolution
biases in both variants to isolate normalization.
Select checkpoints on validation NLL; evaluate the official 10k test set
after selection. Report paired differences and sample standard deviations;
three runs on a small MNIST training subset do not establish a general
advantage or disadvantage for batch normalization. Do not require BN to win.

Check identical starting weights, one moving-statistics update per batch,
inference invariance to batch companions, no statistics mutation during
evaluation, and checkpoint restoration into a fresh model including state.
The split (std::mt19937_64(2026), reproduced by nn.Mt19937_64) and batch
orders (nn.BatchOrder) are identical to TF's; weights come from nn's generator
per seed, so the paired numbers match TF only statistically. Batch norm is
nn.batch_norm (moving statistics in a separate state store); checkpoints are
.npz pairs seed<S>_{plain,bn}_{params,state}.npz.
Learning curves, selected checkpoints and summary: output/nn/models/bn_ablation/.
Preparation: python3 scripts/fetch_datasets.py mnist --split both (in tflow).
Options: data=PATH epochs=6 threads=N (quick mode: 64 fit / 32 validation /
32 test rows, batch 16, one epoch unless epochs= is given).
"""
import math
import os
import time

from pydatamunge.nn_components import dataset_assets
from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
mnist = helper("mnist")


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def bn_options():
    o = nn.BatchNormOptions()
    o.momentum = 0.9
    o.eps = 0.001
    return o


def architecture(use_bn, p, state, x, training):
    """The paired CNN: conv(16)-[BN]-ReLU-pool, conv(32)-[BN]-ReLU-pool, Dense(64)-ReLU, 10 logits."""
    h = nn.reshape(x, [-1, 28, 28, 1])
    for block in range(2):
        width = 16 if block == 0 else 32
        h = nn.conv2d(p, "conv" + str(block + 1), h, width, 3)
        if use_bn:
            h = nn.batch_norm(p, state, "bn" + str(block + 1), h, training, bn_options())
        h = nn.max_pool2d(nn.relu(h), 2, 2)
    h = nn.dense(p, "dense", nn.reshape(h, [-1, 7 * 7 * 32]), 64, nn.Activation.Relu)
    return nn.dense(p, "output", h, 10)


def moments(p, x):
    """Batch mean/variance (unbiased, as TF FusedBatchNorm reports) that a training
    step on `x` feeds into each moving average, from the current weights."""
    result = []
    with nn.no_grad():
        h = nn.reshape(x, [-1, 28, 28, 1])
        for block in range(2):
            i = str(block + 1)
            h = nn.conv2d(p, "conv" + i, h, 16 if block == 0 else 32, 3)
            r = nn.batch_norm_train(h, p.at("bn" + i + "/gamma"), p.at("bn" + i + "/beta"), bn_options().eps)
            result += [r.mean, r.variance]
            h = nn.max_pool2d(nn.relu(r.y), 2, 2)
    return result


class Model:
    """Parameters, batch-norm state, Adam and the L2-regularized kernels of one variant."""

    def __init__(self, use_bn, seed):
        self.use_bn = use_bn
        self.params, self.state = nn.ParameterStore(seed), nn.ParameterStore(0)
        with nn.no_grad():
            architecture(use_bn, self.params, self.state, nn.zeros([1, 784]), False)
        config = nn.OptimizerConfig()
        config.kind = nn.OptimizerKind.Adam
        self.optimizer = nn.Optimizer(self.params, 0.001, config)
        self.weights = [w for w in self.params.parameters() if w.dim() >= 2]

    def builder(self):
        return lambda p, x, training: architecture(self.use_bn, p, self.state, x, training)

    def probabilities(self, x):
        with nn.no_grad():
            return nn.softmax(architecture(self.use_bn, self.params, self.state, x, False))

    def save(self, prefix):
        self.params.save(prefix + "_params.npz")
        if self.use_bn:
            self.state.save(prefix + "_state.npz")

    def load(self, prefix):
        self.params.load(prefix + "_params.npz")
        if self.use_bn:
            self.state.load(prefix + "_state.npz")


def snapshot(store):
    return [t.detach().clone() for t in store.parameters()]


def same(a, b, tolerance, message):
    checks.require(list(a.shape()) == list(b.shape()), message + " shape")
    x, y = a.contiguous().values(), b.contiguous().values()
    if tolerance == 0:
        checks.require(x == y, message)
        return
    for u, v in zip(x, y):
        checks.near(u, v, tolerance, message)


def describe(values):
    mean = sum(values) / len(values)
    variance = sum((x - mean) * (x - mean) for x in values)
    return mean, math.sqrt(variance / (len(values) - 1))


def main(arguments=None):
    args = options(arguments, dict(data=str(checks.dataset("mnist")), epochs=0, threads=0,
                                   full=os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"))
    os.environ["DATAMUNGE_EXAMPLES_FULL"] = "1" if args["full"] else "0"
    if args["threads"]:
        nn.set_num_threads(args["threads"])
    checks.require(args["epochs"] >= 0, "positive epochs required")
    epochs = args["epochs"] or checks.limit(6, 1)
    training = dataset_assets.LoadMnist(args["data"], "train")
    test = dataset_assets.LoadMnist(args["data"], "test")
    checks.require(training.images.shape()[0] == 60000 and test.images.shape()[0] == 10000,
                   "official MNIST row counts")
    train_x, test_x = nn.reshape(training.images, [60000, 784]), nn.reshape(test.images, [10000, 784])
    test_rows = list(range(checks.limit(10000, 32)))
    order = mnist.development_order(60000, 2026)
    fit, validation = order[:12000], order[12000:15000]
    fit = fit[:checks.limit(len(fit), 64)]
    validation = validation[:checks.limit(len(validation), 32)]
    checks.describe()
    directory = checks.output_dir("models/bn_ablation")
    curves, summary = (directory / "history.tsv").open("w"), (directory / "summary.tsv").open("w")
    curves.write("seed\tbatch_norm\tepoch\tvalidation_nll\tvalidation_accuracy\ttrain_seconds\n")
    summary.write("seed\tbatch_norm\tselected_epoch\ttest_nll\ttest_accuracy\ttrain_seconds\n")
    trials = [[], []]
    started = time.perf_counter()
    for seed in (11, 23, 47):
        models = [Model(False, seed), Model(True, seed)]
        for name in models[0].params.names():
            same(models[0].params.at(name), models[1].params.at(name), 0, "identical paired initialization")
        print(f"== seed {seed}: parameters {models[0].params.count()} / {models[1].params.count()}; "
              "identical starting weights")
        best, seconds, selected = [math.inf, math.inf], [0.0, 0.0], [0, 0]
        checkpoints = [str(directory / (f"seed{seed}" + ("_bn" if variant else "_plain"))) for variant in range(2)]
        batches = nn.BatchOrder(len(fit), checks.limit(128, 16), seed + 1000)
        first_update = True
        for epoch in range(1, epochs + 1):
            batches.begin_epoch()
            samples = 0
            while True:
                rows = list(batches.next())
                if not rows:
                    break
                samples += len(rows)
                rows = [fit[row] for row in rows]
                x, y = nn.take_rows(train_x, rows), nn.take_rows(training.labels, rows)
                if first_update:
                    old_state = snapshot(models[1].state)
                    expected_moments = moments(models[1].params, x)
                for variant in range(2):
                    # One explicit training step: forward, mean cross-entropy plus
                    # 0.5 * l2 * sum(W^2) over the kernels, then the Adam update.
                    start = time.perf_counter()
                    model = models[variant]
                    objective = nn.sparse_softmax_cross_entropy(
                        architecture(model.use_bn, model.params, model.state, x, True), y)
                    penalty = nn.sum(nn.square(model.weights[0]))
                    for weight in model.weights[1:]:
                        penalty = nn.add(penalty, nn.sum(nn.square(weight)))
                    model.optimizer.minimize(nn.add(objective, nn.mul(penalty, 0.5 * 1e-4)))
                    seconds[variant] += time.perf_counter() - start
                if first_update:
                    updated = snapshot(models[1].state)
                    checks.require(len(updated) == len(expected_moments), "one mean/variance pair per batch norm")
                    for u_tensor, o_tensor, m_tensor in zip(updated, old_state, expected_moments):
                        for u, o, m in zip(u_tensor.values(), o_tensor.values(), m_tensor.values()):
                            checks.near(u, 0.9 * o + 0.1 * m, 2e-6,
                                        "exactly one moving-statistics update from pre-update activations")
                    first_update = False
            checks.require(samples == len(fit), "all fitting rows used once per epoch")
            for variant in range(2):
                model = models[variant]
                metrics = mnist.evaluate(model.params, model.builder(), train_x, training.labels, validation)
                nll, accuracy = metrics.loss.value(), metrics.accuracy.value()
                if nll < best[variant]:
                    best[variant], selected[variant] = nll, epoch
                    model.save(checkpoints[variant])
                curves.write(f"{seed}\t{variant}\t{epoch}\t{g(nll)}\t{g(accuracy)}\t{g(seconds[variant])}\n")
                curves.flush()
                print(f"  epoch {epoch}{' BN: ' if variant else ' plain: '}validation NLL={g(nll)}, "
                      f"accuracy={g(accuracy)}", flush=True)
        for variant in range(2):
            model = models[variant]
            label = "BN" if variant else "plain"
            model.load(checkpoints[variant])
            validation_result = mnist.evaluate(model.params, model.builder(), train_x, training.labels, validation)
            checks.near(validation_result.loss.value(), best[variant], 1e-7, "restored validation loss")
            result = mnist.evaluate(model.params, model.builder(), test_x, test.labels, test_rows)
            checks.training_quality(result.accuracy.value() >= 0.95, "both models learn the classification task")
            trials[variant].append(dict(accuracy=result.accuracy.value(), nll=result.loss.value(),
                                        seconds=seconds[variant], epoch=selected[variant]))
            summary.write(f"{seed}\t{variant}\t{selected[variant]}\t{g(result.loss.value())}\t"
                          f"{g(result.accuracy.value())}\t{g(seconds[variant])}\n")
            summary.flush()
            print(f"  selected {label} epoch={selected[variant]}; test NLL={g(result.loss.value())}, "
                  f"accuracy={g(result.accuracy.value())}")
            cells = mnist.print_confusion(result.confusion, label + " test confusion (truth rows):")
            total = sum(sum(row) for row in cells)
            checks.require(total == len(test_rows), "all selected test images evaluated")
            checks.near(result.accuracy.value(), sum(cells[i][i] for i in range(10)) / total, 1e-7,
                        "confusion matches accuracy")
        bn = models[1]
        before_evaluation = snapshot(bn.state)
        batch_x, single_x = nn.take_rows(test_x, list(range(8))), nn.take_rows(test_x, [0])
        batch_result, single_result = bn.probabilities(batch_x), bn.probabilities(single_x)
        for c in range(10):
            checks.near(batch_result.at([0, c]), single_result.at([0, c]), 1e-6,
                        "inference predictions do not depend on batch companions")
        after_evaluation = snapshot(bn.state)
        for before, after in zip(before_evaluation, after_evaluation):
            same(before, after, 0, "evaluation leaves moments unchanged")
        fresh = Model(True, 999)
        fresh.load(checkpoints[1])
        same(fresh.probabilities(batch_x), batch_result, 1e-7, "fresh model checkpoint predictions")
        for restored, before in zip(snapshot(fresh.state), before_evaluation):
            same(restored, before, 0, "checkpoint contains moving moments")
    differences = [bn["accuracy"] - plain["accuracy"] for plain, bn in zip(trials[0], trials[1])]
    for variant in range(2):
        mean, sd = describe([trial["accuracy"] for trial in trials[variant]])
        print(f"  {'BN' if variant else 'plain'} test accuracy mean={g(mean)}; sample SD={g(sd)}")
    mean, sd = describe(differences)
    print(f"  paired BN-plain accuracy difference mean={g(mean)}; sample SD={g(sd)}")
    print("  six epochs on 12000 fitting rows; descriptive results from three paired seeds")
    curves.close()
    summary.close()
    print(f"  total time {time.perf_counter() - started:.1f} s")
    print("All batch-normalization ablation checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
