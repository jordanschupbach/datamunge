# Native Python NN example. MobileNet v1 cost/Table-2/gradient checks and alpha 1 / 0.5 CIFAR-10 training; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mobilenet_v1.py
"""MobileNet v1: depthwise separable convolutions with width and resolution multipliers.

Port of tflow's mobilenet_v1. Measure every layer's output size from a
forward pass and tabulate multiply-adds and parameters. Check the paper's
Tables 6 and 7 exactly as counted here (alpha 1/.75/.5/.25 at 224:
569/325/149/41M mult-adds and 4.2/2.6/1.3/0.5M parameters; 224/192/160/128
pixels: 569/418/290/186M), the full-convolution baseline of Table 4 (29.3M
parameters, 4866M mult-adds without the classifier) and Table 2's breakdown
by layer type (1x1: 94.86% of mult-adds, 74.59% of weights). Table 2 prints
1.19% for the first convolution, but 1.91% is what makes the shares sum to
100%. Check every parameter derivative of the full model in training mode
(moving averages not updated), with shifted BN betas: at beta=0, ReLU6 is
positively homogeneous below 6, so a pointwise BN's scale reaches the next
training-mode BN only as a removable per-channel factor and its derivative
nearly vanishes.
https://arxiv.org/abs/1704.04861
https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py

Train alpha 1 and alpha 0.5 on CIFAR-10 with a stride-one first convolution
(all thirteen blocks and their four remaining downsamplings). Use 12k fit /
5k validation, twelve epochs for alpha 1 and twenty-four for alpha 0.5,
crop/flip, Adam(.001), kernel L2(.0001) and the best-validation checkpoint
before evaluating all 10k test images. These are CIFAR adaptations, not the
ImageNet benchmark. Initialization uses nn's generator (He-uniform, as
tflow), so trained numbers match TF only statistically. CIFAR-10 is read
from the tflow dataset cache (common/paths.hpp) or data=.
Options: alpha=1|0.5 (default both), data=PATH epochs=N batch=128 lr=.001
fit=12000 augment=0|1 threads=N seed=17.

The model components (mobilenet::MobileNetV1 and its config, image batching,
cross-entropy and L2) are the C++ ones through SWIG (pydatamunge.nn_components);
the checks and the experiment workflow below mirror
examples/cpp/source/nn/.../mobilenet_v1.cpp line by line.
"""
import math
import struct
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import check, example_runtime, image_training, mobilenet, paths


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def f32(value):
    """Rounds a double to float32, as the C++ float arithmetic does."""
    return struct.unpack("f", struct.pack("f", value))[0]


def copy_config(cfg):
    """ImageTrainingConfig is passed by value to each width in the C++; copy every field."""
    copy = image_training.ImageTrainingConfig()
    for field in ("name", "classes", "epochs", "batch", "validation_rows", "maximum_fit_rows", "learning_rate",
                  "l2", "label_smoothing", "augment", "minimum_test_accuracy", "seed"):
        setattr(copy, field, getattr(cfg, field))
    return copy


def parse_cifar_option(key, value, cfg):
    """image_training::ParseCifarOption; returns the data directory (None when unchanged) or raises."""
    directory = None
    if key == "data":
        directory = value
    elif key == "epochs":
        cfg.epochs = int(value)
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
        return False, directory
    return True, directory


def measure(cfg, resolution):
    """Multiply-adds from measured output sizes; weight shares exclude BN and the classifier bias.
    Returns (params, state, macs, mac_share, weight_share)."""
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    with nn.no_grad():
        net = mobilenet.MobileNetV1(p, state, nn.zeros([1, resolution, resolution, 3]), cfg)
        params, state_count = p.count(), state.count()
        macs, weights = {}, {}
        for layer in net.layers:
            name = layer.name
            kind = ("conv 3x3" if name == "conv0" else "depthwise 3x3" if "depthwise" in name
                    else "conv 1x1" if "pointwise" in name else "conv 3x3")
            macs[kind] = macs.get(kind, 0) + layer.Macs()
            weights[kind] = weights.get(kind, 0) + p.at(name + "/W").numel()
        macs["classifier"] = net.classifier_macs
        weights["classifier"] = p.at("output/W").numel()
    total_macs = sum(macs.values())
    total_weights = sum(weights.values())
    mac_share = {kind: 100. * value / total_macs for kind, value in macs.items()}
    weight_share = {kind: 100. * value / total_weights for kind, value in weights.items()}
    return params, state_count, total_macs, mac_share, weight_share


def cost_checks():
    cases = [
        (1, 224, 4231976, 21888, 568740352),
        (.75, 224, 2585560, 16416, 325400448),
        (.5, 224, 1331592, 10944, 149497088),
        (.25, 224, 470072, 5472, 41030272),
        (1, 192, 4231976, 21888, 418121728),
        (1, 160, 4231976, 21888, 290675200),
        (1, 128, 4231976, 21888, 186400768),
    ]
    for alpha, resolution, expected_params, expected_state, expected_macs in cases:
        cfg = mobilenet.MobileNetV1Config()
        cfg.alpha = alpha
        params, state_count, macs, mac_share, weight_share = measure(cfg, resolution)
        check.Require(params == expected_params and state_count == expected_state and macs == expected_macs,
                      "MobileNet v1 parameters and multiply-adds")
        print(f"  alpha {g(alpha)}, {resolution} px: {macs / 1e6:.1f}M mult-adds, {params / 1e6:.2f}M parameters "
              f"({params}, state {state_count})", flush=True)
        if alpha == 1 and resolution == 224:
            # std::map order: sorted by kind.
            table2 = {"conv 1x1": (94.86, 74.59), "depthwise 3x3": (3.06, 1.06), "conv 3x3": (1.91, .02),
                      "classifier": (.18, 24.33)}
            line = "   "
            for kind in sorted(table2):
                expected = table2[kind]
                check.Near(mac_share[kind], expected[0], .005, "Table 2 mult-add share: " + kind)
                check.Near(weight_share[kind], expected[1], .005, "Table 2 parameter share: " + kind)
                line += f" {kind} {mac_share[kind]:.2f}% / {weight_share[kind]:.2f}%;"
            print(line + " (mult-adds / weights)", flush=True)
    dense = mobilenet.MobileNetV1Config()
    dense.full_convolutions = True
    params, _state_count, macs, _mac_share, _weight_share = measure(dense, 224)
    classifier = 1024 * 1000
    check.Require(params == 29294088 and macs - classifier == 4866269184, "full-convolution MobileNet")
    print(f"  full 3x3 convolutions: {(macs - classifier) / 1e6:.1f}M mult-adds without the classifier, "
          f"{params / 1e6:.1f}M parameters ({macs / 568740352:.2f}x and {params / 4231976:.2f}x MobileNet)", flush=True)


def gradient_checks():
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    cfg = mobilenet.MobileNetV1Config()
    cfg.update_stats = False
    # pixels[i] = std::sin(.13f * float(i)) in float arithmetic.
    pixels = [f32(math.sin(f32(f32(.13) * i))) for i in range(2 * 224 * 224 * 3)]
    x = nn.from_vector(pixels, [2, 224, 224, 3])
    with nn.no_grad():  # create the parameters
        mobilenet.MobileNetV1(p, state, x, cfg, True)
        # At initialization (beta=0) ReLU6 is positively homogeneous below 6, so a
        # pointwise BN's scale passes through ReLU6 and the per-channel depthwise
        # convolution into the next training-mode BN, which removes it: those gamma
        # derivatives nearly vanish. Shifted betas, as after training, avoid that.
        for name in p.names():
            if len(name) > 5 and name.endswith("/beta"):
                numel = p.at(name).numel()
                beta = [f32(.5 * f32(math.sin(f32(f32(f32(.7) * i) + float(len(name)))))) for i in range(numel)]
                p.set(name, nn.from_vector(beta, list(p.at(name).shape())))
    net = mobilenet.MobileNetV1(p, state, x, cfg, True)
    check.Require(list(net.features.shape()) == [2, 7, 7, 1024] and list(net.logits.shape()) == [2, 1000],
                  "MobileNet output dimensions")
    gradients = nn.grad(nn.mean(nn.square(net.logits)), p.parameters())
    minimum, maximum, smallest = math.inf, 0., ""
    names = p.names()
    for i in range(len(gradients)):
        norm = nn.sqrt(nn.sum(nn.square(gradients[i]))).item()
        check.Require(math.isfinite(norm), "finite MobileNet derivative")
        if norm < minimum:
            minimum, smallest = norm, names[i]
        maximum = max(maximum, norm)
    check.Require(minimum > 1e-4 * maximum, "clearly nonzero MobileNet derivatives with shifted BN")
    print(f"  alpha 1 at 224 with shifted BN betas: 7x7x1024 features; training-mode gradient norms "
          f"[{g(minimum)},{g(maximum)}], smallest {smallest}", flush=True)


def train_compact_mobilenet_v1(alpha, development, test, cfg, explicit_epochs):
    """Fits one CIFAR MobileNet v1 width: fixed development split, Adam on
    cross-entropy plus L2, validation-NLL checkpoint selection, one test pass.
    Without an explicit epochs= budget, alpha 1 trains twelve epochs and alpha
    0.5 twenty-four."""
    cfg = copy_config(cfg)
    cfg.name = "mobilenet_v1_cifar10" if alpha == 1 else "mobilenet_v1_050_cifar10"
    if not explicit_epochs:
        cfg.epochs = 12 if alpha == 1 else 24
    expected = 3217226 if alpha == 1 else 823722

    def model(params, state, image, training):
        options = mobilenet.MobileNetV1Config()
        options.alpha = alpha
        options.classes = 10
        options.small_stem = True
        logits = mobilenet.MobileNetV1(params, state, image, options, training).logits
        check.Require(params.count() == expected, "CIFAR MobileNet parameter count")
        return logits

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


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    widths = [1., .5]
    directory = paths.Dataset("cifar10")
    cfg = image_training.ImageTrainingConfig()
    cfg.maximum_fit_rows = 12000
    cfg.minimum_test_accuracy = .4
    explicit_epochs = False
    for argument in arguments:
        if argument == "alpha=1":
            widths = [1.]
        elif argument == "alpha=0.5":
            widths = [.5]
        elif argument.startswith("alpha="):
            raise ValueError("alpha must be 1 or 0.5")
        else:
            key, separator, value = argument.partition("=")
            if not separator:
                raise ValueError("expected key=value option")
            explicit_epochs = explicit_epochs or key == "epochs"
            # Parsed before the checks so threads= applies to them too.
            known, data = parse_cifar_option(key, value, cfg)
            if not known:
                raise ValueError("unknown option: " + argument)
            if data is not None:
                directory = data
    cost_checks()
    gradient_checks()
    development, test = image_training.LoadCifar10(str(directory))
    for alpha in widths:
        train_compact_mobilenet_v1(alpha, development, test, cfg, explicit_epochs)
    print("All MobileNet v1 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
