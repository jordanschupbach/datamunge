# Native Python NN example. ZFNet architecture checks, compact CIFAR-10 training and deconvnet visualization; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.2_classic_architectures/zfnet.py
"""ZFNet: finer early sampling, dense connectivity and deconvnet visualization.

Port of tflow's zfnet. Build the Figure-3 network from Zeiler/Fergus, ECCV
2014. Check full 224-pixel stage shapes, 62,357,608 parameters and finite
nonzero derivatives through every learned tensor. The shared header
documents exact padding.
Reference: https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf

Train a separate compact version on 12k CIFAR-10 rows: quarter channel
widths, 256-unit dense layers, 67-pixel resized input, ten epochs, crop/flip
augmentation, He initialization and Adam. Keep dense connectivity, 7x7/5x5
stride-2 early convolutions, LRN after pooling and .5 dropout. Select on a
fixed 5k validation split and evaluate all 10k official test rows.
These compact results are not full-model ImageNet results.

Restore the selected model and project its strongest conv5 activation for
the first test image back to pixels: unpool at forward switches, rectify the
backward signal and apply convolution adjoints, omitting LRN. Compare with
the ordinary input gradient. A switch fixture and a ReLU sign fixture
distinguish this reconstruction rule from standard or guided backprop.
CIFAR-10 is read from paths::Dataset("cifar10"). Export:
output/nn/visualizations/zfnet/features.npz (tflow: output/visualizations/zfnet,
rendered by tflow's scripts/plot_zfnet.py). This is a single-feature
diagnostic, not an inverse network or a complete explanation of the class.
Options: data=PATH epochs=10 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

The model components (zfnet::ZFNet and ZFNetDeconvolution, the deconvnet
primitives, architecture fixtures, image batching, cross-entropy and L2) are
the C++ ones through SWIG (pydatamunge.nn_components); the checks and the
experiment workflow below mirror examples/cpp/source/nn/.../zfnet.cpp line by line.
"""
import math
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import (architecture, check, dataset_assets, deconvnet, example_runtime,
                                       image_training, paths, zfnet)


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def compact_config():
    """Quarter-width, 256-unit, 67-pixel, ten-class ZFNet (CompactConfig in the C++)."""
    cfg = zfnet.ZFNetConfig()
    cfg.image_size = 67
    cfg.classes = 10
    cfg.width_divisor = 4
    cfg.dense_width = 256
    return cfg


def reconstruction_checks():
    """Switch fixture and ReLU sign fixture: deconvnet rectification versus ordinary backprop."""
    values = nn.from_vector([1, 9, 2, 3, 4, 5, 6, 10, 7, 8, 12, 0, 11, 2, 1, 4], [1, 4, 4, 1])
    signal = nn.from_vector([1, 2, 3, 4], [1, 2, 2, 1])
    unpooled = list(deconvnet.UnpoolAtSwitches(values, signal, 2, 2, "VALID").values())
    x = nn.from_vector([-1, 2], [1, 1, 1, 2])
    x.requires_grad_(True)
    backward = nn.from_vector([3, -4], [1, 1, 1, 2])
    identity = nn.from_vector([1, 0, 0, 1], [1, 1, 2, 2])
    ordinary = list(nn.grad(nn.sum(nn.mul(nn.relu(x), backward)), [x])[0].values())
    projection = list(deconvnet.DeconvLayer(x, identity, backward, 1, "VALID").values())
    expected = [0.0] * 16
    expected[1] = 1
    expected[7] = 2
    expected[12] = 3
    expected[10] = 4
    for i in range(16):
        check.Near(unpooled[i], expected[i], 0, "unpool uses original maximum locations")
    check.Near(ordinary[0], 0, 0, "ordinary ReLU blocks negative forward input")
    check.Near(ordinary[1], -4, 0, "ordinary ReLU can pass a negative derivative")
    check.Near(projection[0], 3, 0, "deconvnet passes positive backward signal")
    check.Near(projection[1], 0, 0, "deconvnet blocks negative backward signal")
    print("  unpool switches and deconvnet-vs-backprop rectification checks passed")


def full_architecture_checks():
    """Figure-3 stage shapes, 62,357,608 parameters and finite nonzero parameter derivatives at 224 pixels."""
    p = nn.ParameterStore(17)
    net = zfnet.ZFNet(p, architecture.SinePixels([1, 224, 224, 3], .5, .13), zfnet.ZFNetConfig())
    check.Require(p.count() == 62357608, "full ZFNet parameter count")
    stages = [net.convolution(i) for i in range(5)] + [net.pooling(i) for i in range(3)] + [net.fc6, net.fc7, net.logits]
    expected = [[1, 110, 110, 96], [1, 26, 26, 256], [1, 13, 13, 384], [1, 13, 13, 384], [1, 13, 13, 256],
                [1, 55, 55, 96], [1, 13, 13, 256], [1, 6, 6, 256], [1, 4096], [1, 4096], [1, 1000]]
    for stage, shape in zip(stages, expected):
        architecture.RequireShape(stage, shape, "ZFNet Figure-3 stage shape")
    minimum, maximum = architecture.GradientNormRange(p, nn.mean(nn.square(net.logits)), "ZFNet parameter tensor")
    print(f"  full ZFNet: {p.count()} parameters; conv sizes 110/26/13/13/13; pool sizes 55/13/6")
    print(f"  dense widths 4096/4096/1000; parameter gradient norms [{g(minimum)},{g(maximum)}]")


def visualize(directory):
    """Reload the selected compact checkpoint and project the first test image's strongest conv5 unit to pixels."""
    p = nn.ParameterStore(17)
    test = dataset_assets.LoadCifar10(str(directory), "test")
    pixels, labels = image_training.ImageBatch(test, [0])
    image = nn.resize_bilinear(pixels, [67, 67], False, True)
    with nn.no_grad():
        zfnet.ZFNet(p, image, compact_config())
    p.load(str(Path(paths.OutputDir("models/zfnet_compact_cifar10")) / "best_params.npz"))
    image.requires_grad_(True)
    net = zfnet.ZFNet(p, image, compact_config())
    activation_map = list(net.convolution(4).values())
    check.Require(net.convolution(4).shape() == (1, 3, 3, 64), "compact conv5 map")
    index = 0
    for i in range(1, len(activation_map)):
        if activation_map[i] > activation_map[index]:
            index = i
    activation = activation_map[index]
    check.Require(activation > 0, "positive selected conv5 activation")
    seed = [0.0] * len(activation_map)
    seed[index] = activation
    signal = nn.from_vector(seed, [1, 3, 3, 64])
    gradient = nn.grad(nn.sum(nn.mul(net.convolution(4), signal)), [image])[0]
    with nn.no_grad():
        deconvolution = zfnet.ZFNetDeconvolution(p, net, signal)
    check.Require(deconvolution.shape() == (1, 67, 67, 3) and gradient.shape() == deconvolution.shape(),
                  "projection image dimensions")
    a, b = list(deconvolution.values()), list(gradient.values())
    deconv_energy, gradient_energy, difference = 0.0, 0.0, 0.0
    for ai, bi in zip(a, b):
        check.Require(math.isfinite(ai) and math.isfinite(bi), "finite feature projections")
        deconv_energy += ai * ai
        gradient_energy += bi * bi
        difference += (ai - bi) * (ai - bi)
    check.Require(deconv_energy > 0 and gradient_energy > 0 and difference > 1e-8,
                  "nonzero distinct deconvnet and ordinary gradients")
    unit_row, unit_column, unit_channel = index // (3 * 64), (index // 64) % 3, index % 64
    prediction = nn.argmax(net.logits.detach(), 1)
    path = Path(paths.OutputDir("visualizations/zfnet")) / "features.npz"
    nn.save_npz(str(path), [("image", image.detach()),
                            ("label", labels),
                            ("prediction", prediction),
                            ("unit", nn.from_vector_int64([unit_row, unit_column, unit_channel], [3])),
                            ("deconvolution", deconvolution),
                            ("gradient", gradient)])
    print(f"  first test image conv5 unit={unit_row},{unit_column},{unit_channel}; activation={g(activation)}; "
          f"deconv energy={g(deconv_energy)}; ordinary gradient energy={g(gradient_energy)}")
    print(f"  wrote {path}")
    print("All ZFNet architecture and visualization checks passed.")


def parse_cifar_options(arguments, cfg):
    """image_training::ParseCifarOption over key=value arguments; returns (data directory, explicit epochs)."""
    directory, explicit_epochs = paths.Dataset("cifar10"), False
    for argument in arguments:
        key, separator, value = argument.partition("=")
        if not separator:
            raise ValueError("expected key=value option")
        if key == "data":
            directory = value
        elif key == "epochs":
            cfg.epochs, explicit_epochs = int(value), True
        elif key == "batch":
            cfg.batch = int(value)
        elif key == "lr":
            cfg.learning_rate = float(value)
        elif key == "fit":
            cfg.maximum_fit_rows = int(value)
        elif key == "augment" and value in ("0", "1"):
            cfg.augment = value == "1"
        elif key == "threads":
            nn.set_num_threads(int(value))
        elif key == "seed":
            cfg.seed = int(value)
        else:
            raise ValueError("unknown option: " + argument)
    return directory, explicit_epochs


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    reconstruction_checks()
    full_architecture_checks()
    cfg = image_training.ImageTrainingConfig()
    cfg.name = "zfnet_compact_cifar10"
    cfg.maximum_fit_rows = 12000
    cfg.minimum_test_accuracy = .35
    data, explicit_epochs = parse_cifar_options(arguments, cfg)
    development, test = image_training.LoadCifar10(str(data))

    # Compact ZFNet on CIFAR-10: fixed development split, Adam on cross-entropy
    # plus L2, validation-NLL checkpoint selection, one test pass; visualize then
    # reloads the selected checkpoint.
    def model(params, state, x, training):
        return zfnet.ZFNet(params, nn.resize_bilinear(x, [67, 67], False, True), compact_config(), training).logits

    # Quick mode bounds the data and batch; an explicit epoch budget is honored.
    if not explicit_epochs:
        cfg.epochs = example_runtime.Limit(cfg.epochs, 1)
    cfg.batch = example_runtime.Limit(cfg.batch, 8)
    if not example_runtime.Full():
        cfg.maximum_fit_rows = min(cfg.maximum_fit_rows, 32) if cfg.maximum_fit_rows else 32
    example_runtime.Describe()
    development_rows = int(development.images.shape()[0])
    test_size = example_runtime.LimitInt64(int(test.images.shape()[0]), 32)
    check.Require(development.images.dim() == 4 and test.images.dim() == 4 and cfg.validation_rows > 0
                  and cfg.validation_rows < development_rows and cfg.epochs > 0 and cfg.batch > 0 and cfg.classes > 1,
                  "valid image training dimensions/config")
    for d in range(1, 4):
        check.Require(development.images.shape()[d] == test.images.shape()[d], "training/test image dimensions match")
    available = development_rows - cfg.validation_rows
    check.Require(0 <= cfg.maximum_fit_rows <= available, "valid optional fitting-row limit")
    fit_count = cfg.maximum_fit_rows or available
    check.Require(fit_count >= 2, "at least two fitting rows for model initialization")

    # Split the development rows once (std::mt19937_64(2026) shuffle); keep the
    # validation split fixed when limiting the fitting subset.
    order = [int(i) for i in nn.Mt19937_64(2026).shuffle(list(range(development_rows)))]
    fit = order[:fit_count]
    validation = order[available:]
    validation = validation[:example_runtime.Limit(len(validation), 32)]
    test_rows = list(range(test_size))

    params, state = nn.ParameterStore(cfg.seed), nn.ParameterStore(cfg.seed)
    with nn.no_grad():  # create every parameter (and batch-norm state) before building the optimizer
        model(params, state, image_training.ImageBatch(development, [fit[0], fit[1]])[0], False)
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(params, cfg.learning_rate, config)

    def evaluate(dataset, rows):
        """Inference-mode cross-entropy, accuracy and confusion over `rows`, in batches of 256."""
        loss, accuracy, confusion = nn.WeightedMean(), nn.WeightedMean(), nn.ConfusionMatrix(cfg.classes)
        with nn.no_grad():
            for start in range(0, len(rows), 256):
                ids = rows[start:start + 256]
                x, y = image_training.ImageBatch(dataset, ids)
                logits = model(params, state, x, False)
                prediction = nn.argmax(logits, 1)
                loss.add(nn.sparse_softmax_cross_entropy(logits, y).item(), len(ids))
                accuracy.add(nn.accuracy(prediction, y), len(ids))
                confusion.add(y, prediction)
        return loss.value(), accuracy.value(), confusion

    directory = Path(paths.OutputDir("models/" + cfg.name))
    params_path, state_path = str(directory / "best_params.npz"), str(directory / "best_state.npz")
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
            # Augmented batch; training-mode forward; cross-entropy plus 0.5 * l2 * sum(W^2); one Adam step.
            x, y = image_training.ImageBatch(development, rows, augmentation if cfg.augment else None)
            logits = model(params, state, x, True)
            objective = image_training.WithL2(
                image_training.TrainingCrossEntropy(logits, y, cfg.classes, cfg.label_smoothing), params, cfg.l2)
            optimizer.minimize(objective)
        check.Require(used == fit_count, "each fit image used once per epoch")
        train_seconds = time.perf_counter() - epoch_start
        loss, accuracy, _confusion = evaluate(development, validation)
        if loss < best:
            best, best_epoch = loss, epoch
            params.save(params_path)
            if state.size():
                state.save(state_path)
        history.write(f"{epoch}\t{g(loss)}\t{g(accuracy)}\n")
        history.flush()
        seconds = time.perf_counter() - epoch_start
        epoch_seconds.append(seconds)
        print(f"  epoch {epoch}: validation NLL={g(loss)}, accuracy={g(accuracy)} "
              f"({train_seconds:.3g} s train, {seconds:.3g} s total)", flush=True)
    history.close()
    params.load(params_path)
    if state.size():
        state.load(state_path)
    check.Near(evaluate(development, validation)[0], best, 1e-6, "best image checkpoint validation loss")
    test_loss, test_accuracy, confusion = evaluate(test, test_rows)
    print(f"  selected epoch={best_epoch}; test NLL={g(test_loss)}; accuracy={g(test_accuracy)}")
    # Confusion matrix: every evaluation row, truth on rows and predictions on columns.
    print("  confusion (truth rows):")
    values = [int(round(v)) for v in confusion.to_tensor().values()]
    cells = [values[i * cfg.classes:(i + 1) * cfg.classes] for i in range(cfg.classes)]
    for row in cells:
        print("   " + "".join(f" {v:4d}" for v in row))
    total, correct = sum(map(sum, cells)), sum(cells[i][i] for i in range(cfg.classes))
    check.Require(total == test_size, "all test images evaluated")
    check.Near(test_accuracy, correct / total, 1e-7, "confusion agrees with accuracy")
    check.TrainingQuality(test_accuracy >= cfg.minimum_test_accuracy, "image classifier demonstration accuracy threshold")
    train_total = sum(epoch_seconds)
    print(f"  training time {train_total:.1f} s ({train_total / cfg.epochs:.2f} s/epoch)")
    print(f"  best checkpoint and history: {directory}\nAll {cfg.name} checks passed.", flush=True)
    visualize(data)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
