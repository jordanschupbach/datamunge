# Native Python NN example. RegNet design/cost/gradient checks and Y-400MF CIFAR-10 training; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/regnet.py
"""RegNet: X and Y models generated from a quantized linear width rule.

Generate all fourteen released RegNetX/Y models (400MF to 32GF) from
(depth, w0, wa, wm, group width) and check their stage widths and depths,
learned parameters (5,495,976 for X-400MF to 145,046,770 for Y-32GF,
torchvision's counts) and multiply-adds (0.414 to 31.736 GFLOPs for X and
0.402 to 32.280 for Y, torchvision's figures) at 224 pixels. Show the rule
at work for Y-400MF. Check every derivative of Y-400MF in training mode
with shifted BN betas and SE biases; with a ReLU after every addition, none
vanish.
https://arxiv.org/abs/2003.13678
https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py

Train Y-400MF on CIFAR-10 with a stride-one stem (all sixteen blocks and
the four stage downsamplings). Use 12k fit / 5k validation, twelve epochs,
crop/flip, Adam(.001), kernel L2(.0001) and the best-validation checkpoint
before evaluating all 10k test images.
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1.

The model components (regnet::RegNet and its design rule, the grouped
convolutions, mobile_checks fixtures, image batching, cross-entropy and L2)
are the C++ ones through SWIG (pydatamunge.nn_components); the checks and the
experiment workflow below mirror examples/cpp/source/nn/.../regnet.cpp line by line.
"""
import math
import os
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import check, example_runtime, grouped, image_training, mobile_checks, paths, regnet


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def join(values):
    """regnet.cpp's Join: slash-separated integers."""
    return "/".join(str(int(v)) for v in values)


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


def fused_group_checks():
    """Compare the packed grouped kernel with independent per-group convolutions,
    including every filter, bias and input derivative in float64."""
    for bias in (False, True):
        for activation in (nn.Activation.Identity, nn.Activation.Relu):
            p = nn.ParameterStore(17, nn.DType.Float64)
            values = [math.sin(.13 * i) for i in range(2 * 7 * 6 * 12)]
            x = nn.from_vector_double(values, [2, 7, 6, 12]).requires_grad_()
            o = nn.ConvLayerOptions()
            o.bias = bias
            o.activation = activation
            o.stride = 2
            o.dilation = 2
            split = grouped.GroupedConv2D(p, "fixture", x, 15, 3, 3, o)
            inputs = list(p.parameters()) + [x]
            a = nn.grad(nn.sum(nn.square(split)), inputs)
            fused = grouped.FusedGroupedConv2D(p, "fixture", x, 15, 3, 3, o)
            b = nn.grad(nn.sum(nn.square(fused)), inputs)
            check.Near(nn.max(nn.abs(nn.sub(split, fused))).item(), 0, 1e-12, "packed grouped convolution values")
            for i in range(len(a)):
                check.Near(nn.max(nn.abs(nn.sub(a[i], b[i]))).item(), 0, 1e-11,
                           "every packed grouped filter/bias/input derivative")


def design_checks():
    d = regnet.RegNetDesign(regnet.RegNetModel("y400mf"))
    continuous = list(d.continuous)
    print(f"  Y-400MF (d=16, w0=48, wa=27.89, wm=2.09): u_j = 48, {g(continuous[1])}, {g(continuous[2])}, ..., "
          f"{g(continuous[-1])} quantize to {join(d.quantized)} -> stages {join(d.widths)} with depths "
          f"{join(d.depths)}", flush=True)


def cost_checks():
    cases = [
        ("x400mf", 5495976, 37856, 413812608, [32, 64, 160, 400], [1, 2, 7, 12]),
        ("x800mf", 7259656, 37312, 799699712, [64, 128, 288, 672], [1, 3, 7, 5]),
        ("x1.6gf", 9190136, 43504, 1602849792, [72, 168, 408, 912], [2, 4, 10, 2]),
        ("x3.2gf", 15296552, 62560, 3176621952, [96, 192, 432, 1008], [2, 6, 15, 2]),
        ("x8gf", 39572648, 90464, 7995132416, [80, 240, 720, 1920], [2, 5, 15, 1]),
        ("x16gf", 54278536, 111168, 15940755456, [256, 512, 896, 2048], [2, 6, 13, 1]),
        ("x32gf", 107811560, 162016, 31735929792, [336, 672, 1344, 2520], [2, 7, 13, 1]),
        ("y400mf", 4344144, 27152, 401842848, [48, 104, 208, 440], [1, 3, 6, 6]),
        ("y800mf", 6432512, 30432, 833856256, [64, 144, 320, 784], [1, 3, 8, 2]),
        ("y1.6gf", 11202430, 52672, 1612076160, [48, 120, 336, 888], [2, 6, 17, 2]),
        ("y3.2gf", 19436338, 66160, 3176482752, [72, 216, 576, 1512], [2, 5, 13, 1]),
        ("y8gf", 39381472, 86528, 8473070592, [224, 448, 896, 2016], [2, 4, 10, 1]),
        ("y16gf", 83590140, 122816, 15911510272, [224, 448, 1232, 3024], [2, 4, 11, 1]),
        ("y32gf", 145046770, 158288, 32279552832, [232, 696, 1392, 3712], [2, 5, 12, 1]),
    ]
    for name, parameters, state_count, expected_macs, widths, depths in cases:
        design = regnet.RegNetDesign(regnet.RegNetModel(name))
        check.Require(list(design.widths) == widths and list(design.depths) == depths, "RegNet stage design " + name)
        with nn.no_grad():
            p, state = nn.ParameterStore(17), nn.ParameterStore(17)
            cfg = regnet.RegNetConfig()
            cfg.model = name
            net = regnet.RegNet(p, state, nn.zeros([1, 224, 224, 3]), cfg)
            # classifier_macs plus every layer's Macs(): MobileNetOutputs::Macs() is that sum.
            macs = net.Macs()
            check.Require(p.count() == parameters and state.count() == state_count and macs == expected_macs,
                          "RegNet cost " + name)
            print(f"{name}:widths={join(design.widths)} depths={join(design.depths)} parameters={p.count()} "
                  f"MACs={macs}", flush=True)


def gradient_checks():
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    cfg = regnet.RegNetConfig()
    cfg.update_stats = False
    x = mobile_checks.Pattern([2, 224, 224, 3], 1, 0)
    with nn.no_grad():
        regnet.RegNet(p, state, x, cfg, True)
        mobile_checks.ShiftBetas(p)
    net = regnet.RegNet(p, state, x, cfg, True)
    norms = list(mobile_checks.GradientNorms(p, nn.mean(nn.square(net.logits))))
    for norm in norms:
        check.Require(math.isfinite(norm), "finite RegNet derivative")
    low, high = min(norms), max(norms)
    check.Require(low > 1e-5 * high, "nonzero RegNet gradients with shifted BN/SE")
    print(f"Y-400MF gradient norms {g(low)} to {g(high)}", flush=True)


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    nn.set_num_threads(2)
    fused_group_checks()
    design_checks()
    cost_checks()
    gradient_checks()
    cfg = image_training.ImageTrainingConfig()
    cfg.name = "regnet_y400mf_cifar10"
    cfg.maximum_fit_rows = 12000
    cfg.epochs = 12
    cfg.minimum_test_accuracy = .4
    directory, explicit_epochs = parse_cifar_options(arguments, cfg)
    development, test = image_training.LoadCifar10(str(directory))

    def model(params, state, image, training):
        """CIFAR RegNet Y-400MF: stride-one stem and ten classes; the parameter count is checked on every pass."""
        options = regnet.RegNetConfig()
        options.classes = 10
        options.small_stem = True
        net = regnet.RegNet(params, state, image, options, training)
        check.Require(params.count() == 3907554, "CIFAR RegNet Y-400MF count")
        return net.logits

    # The training workflow is written out below, mirroring the C++. Quick mode
    # bounds the data and batch; an explicit epoch budget is honored.
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
    print(f"  best checkpoint and history: {directory}\nAll {cfg.name} checks passed.")
    print("All RegNet checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
