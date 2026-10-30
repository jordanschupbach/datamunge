# Native Python NN example. CIFAR-10 CNN with crop/flip augmentation, trained in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.1_fundamentals/cnn_cifar10.py
"""A small CIFAR-10 CNN trained with independent random crops and flips.

Port of tflow's cnn_cifar10. Preparation (in ~/projects/tflow_examples):
python3 scripts/fetch_datasets.py cifar10 --split both.
Architecture: Conv(32,3)-ReLU-pool -> Conv(64,3)-ReLU-pool -> Dense(128)-
ReLU -> 10 logits. Train on 45k original images with four-pixel zero padding,
random 32x32 crops and horizontal flips; normalize RGB pixels to [-1,1].
Select by NLL on 5k untouched validation images, restore that checkpoint,
then evaluate all 10k original test images once. Adam/L2; no pretrained data.
Options: data=PATH epochs=10 batch=128 lr=.001 augment=0|1 threads=N seed=17 fit=N.
Quick mode uses 32 fit / 32 validation / 32 test images, batch <= 8 and one
epoch unless epochs= is given.
The default run requires 55% test accuracy as a training sanity check.
Checkpoint/history: output/nn/models/cnn_cifar10/ (or cnn_cifar10_no_augmentation).
"""
import math
import os
import time

from pydatamunge import nn
from pydatamunge.nn_workflows import helper, options

checks = helper("checks")
image_training = helper("image_training")
mnist = helper("mnist")


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def augmentation_checks():
    fixture = nn.ImageDataset()
    fixture.images = nn.ones([2, 32, 32, 3])
    fixture.labels = nn.from_vector_int64([2, 7], [2])
    first, repeat = nn.Generator(151), nn.Generator(151)
    plain = image_training.image_batch(fixture, [1, 0])
    augmented = image_training.image_batch(fixture, [1, 0], first)
    reproduced = image_training.image_batch(fixture, [1, 0], repeat)
    p, a, r = plain[0].values(), augmented[0].values(), reproduced[0].values()
    padded = interior = 0
    for i in range(len(p)):
        checks.near(p[i], 1, 0, "evaluation only normalizes pixels")
        checks.near(a[i], r[i], 0, "seeded per-image augmentation")
        checks.require(a[i] == -1 or a[i] == 1, "crop preserves white image or zero padding")
        padded += a[i] == -1
        interior += a[i] == 1
    checks.require(padded > 0 and interior > 0, "random crop includes padded and original pixels")
    labels = augmented[1].values()
    checks.require(labels[0] == 7 and labels[1] == 2, "augmentation preserves labels and row order")
    print("  augmentation checks: reproducible crops, normalized padding and paired labels passed")


def model(p, state, image, training=False):
    """Conv(32,3)-ReLU-pool, Conv(64,3)-ReLU-pool, Dense(128)-ReLU, 10 logits."""
    conv = nn.ConvLayerOptions()
    conv.activation = nn.Activation.Relu
    first = nn.max_pool2d(nn.conv2d(p, "conv1", image, 32, 3, conv), 2, 2)
    second = nn.max_pool2d(nn.conv2d(p, "conv2", first, 64, 3, conv), 2, 2)
    hidden = nn.dense(p, "dense", nn.reshape(second, [-1, 8 * 8 * 64]), 128, nn.Activation.Relu)
    return nn.dense(p, "output", hidden, 10)


def main(arguments=None):
    defaults = image_training.cifar_option_defaults(checks.dataset("cifar10"))
    defaults["epochs"] = 0  # zero: the configuration default (one epoch in quick mode)
    defaults["full"] = os.environ.get("DATAMUNGE_EXAMPLES_FULL") == "1"
    args = options(arguments, defaults)
    os.environ["DATAMUNGE_EXAMPLES_FULL"] = "1" if args["full"] else "0"
    explicit_epochs = args["epochs"] != 0
    cfg = image_training.ImageTrainingConfig(name="cnn_cifar10")
    default_epochs = cfg.epochs
    image_training.apply_cifar_options(args, cfg)
    if not explicit_epochs:
        cfg.epochs = default_epochs
    if not cfg.augment:
        cfg.name += "_no_augmentation"
    augmentation_checks()
    development, test = image_training.load_cifar10(args["data"])

    # Quick mode bounds the data and batch; an explicit epoch budget is honored.
    if not explicit_epochs:
        cfg.epochs = checks.limit(cfg.epochs, 1)
    cfg.batch = checks.limit(cfg.batch, 8)
    if not checks.full():
        cfg.maximum_fit_rows = min(cfg.maximum_fit_rows, 32) if cfg.maximum_fit_rows else 32
    checks.describe()
    development_rows = int(development.images.shape()[0])
    test_size = checks.limit(int(test.images.shape()[0]), 32)
    checks.require(development.images.dim() == 4 and test.images.dim() == 4 and cfg.validation_rows > 0
                   and cfg.validation_rows < development_rows and cfg.epochs > 0 and cfg.batch > 0
                   and cfg.classes > 1, "valid image training dimensions/config")
    for d in range(1, 4):
        checks.require(development.images.shape()[d] == test.images.shape()[d], "training/test image dimensions match")
    available = development_rows - cfg.validation_rows
    checks.require(0 <= cfg.maximum_fit_rows <= available, "valid optional fitting-row limit")
    fit_count = cfg.maximum_fit_rows or available

    # Split the development rows once (std::mt19937_64(2026) shuffle); keep the
    # validation split fixed when limiting the fitting subset.
    order = list(nn.Mt19937_64(2026).shuffle(list(range(development_rows))))
    fit = order[:fit_count]
    validation = order[available:]
    validation = validation[:checks.limit(len(validation), 32)]
    test_rows = list(range(test_size))

    params, state = nn.ParameterStore(cfg.seed), nn.ParameterStore(cfg.seed)
    with nn.no_grad():  # create every parameter before building the optimizer
        model(params, state, image_training.image_batch(development, [fit[0], fit[1]])[0], False)
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(params, cfg.learning_rate, config)

    directory = checks.output_dir("models/" + cfg.name)
    params_path = str(directory / "best_params.npz")
    history = (directory / "history.tsv").open("w")
    history.write("epoch\tvalidation_loss\tvalidation_accuracy\n")
    print(f"== {cfg.name}: {fit_count} fit / {len(validation)} validation / {test_size} test")
    print(f"  parameters={params.count()}; augmentation={int(cfg.augment)}; batch={cfg.batch}; "
          f"lr={g(cfg.learning_rate)}; threads={nn.get_num_threads()}", flush=True)
    batches = nn.BatchOrder(fit_count, cfg.batch, 29)
    augmentation = nn.Generator(151)
    best, best_epoch = math.inf, -1
    epoch_seconds = []
    for epoch in range(1, cfg.epochs + 1):
        epoch_start = time.perf_counter()
        batches.begin_epoch()
        used = 0
        while True:
            rows = list(batches.next())
            if not rows:
                break
            used += len(rows)
            rows = [fit[row] for row in rows]
            # Augmented batch, cross-entropy plus 0.5 * l2 * sum(W^2), one Adam step.
            x, y = image_training.image_batch(development, rows, augmentation if cfg.augment else None)
            optimizer.minimize(image_training.training_objective(params, state, model, x, y, cfg))
        checks.require(used == fit_count, "each fit image used once per epoch")
        train_seconds = time.perf_counter() - epoch_start
        result = image_training.evaluate_images(params, state, model, development, validation, cfg.classes)
        loss, accuracy = result.loss.value(), result.accuracy.value()
        if loss < best:
            best, best_epoch = loss, epoch
            params.save(params_path)
        history.write(f"{epoch}\t{g(loss)}\t{g(accuracy)}\n")
        history.flush()
        seconds = time.perf_counter() - epoch_start
        epoch_seconds.append(seconds)
        print(f"  epoch {epoch}: validation NLL={g(loss)}, accuracy={g(accuracy)} "
              f"({train_seconds:.3g} s train, {seconds:.3g} s total)", flush=True)
    history.close()
    params.load(params_path)
    checks.near(image_training.evaluate_images(params, state, model, development, validation, cfg.classes).loss.value(),
                best, 1e-6, "best image checkpoint validation loss")
    result = image_training.evaluate_images(params, state, model, test, test_rows, cfg.classes)
    print(f"  selected epoch={best_epoch}; test NLL={g(result.loss.value())}; accuracy={g(result.accuracy.value())}")
    cells = mnist.print_confusion(result.confusion, "confusion (truth rows):")
    total = sum(sum(row) for row in cells)
    correct = sum(cells[i][i] for i in range(cfg.classes))
    checks.require(total == test_size, "all test images evaluated")
    checks.near(result.accuracy.value(), correct / total, 1e-7, "confusion agrees with accuracy")
    checks.training_quality(result.accuracy.value() >= cfg.minimum_test_accuracy,
                            "image classifier demonstration accuracy threshold")
    train_total = sum(epoch_seconds)
    print(f"  training time {train_total:.1f} s ({train_total / cfg.epochs:.2f} s/epoch)")
    print(f"  best checkpoint and history: {directory}\nAll {cfg.name} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
