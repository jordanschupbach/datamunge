# Native Python NN example. Deformable convolutions v1/v2: layer checks against ordinary, shifted and host-bilinear convolutions, compact deformable ResNet-50 counts/derivatives, then DCNv1 and DCNv2 CIFAR-10 training; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.5_attention_and_modern_convnets/deformable_conv.py
"""Deformable convolutions v1 / v2: learned sampling offsets (and v2 modulation masks).

Port of tflow's deformable_conv (layer: common/deformable.hpp on
nn::deformable_conv2d). Check that a new deformable layer (zero offsets)
equals an ordinary 3x3 convolution with the same weights at strides one and
two, and that a new modulated layer gives half of it (masks start at
sigmoid(0)). Check integer offsets (every tap moved one row down reads the
convolution of the shifted input, with zeros beyond the border) and
fractional offsets against a host bilinear computation, and derivatives
with respect to the offsets against central differences. Count the compact
ResNet-50 with deformable 3x3 convolutions in stages 3 and 4 and check its
derivatives (training mode, moving averages not updated).
https://arxiv.org/abs/1703.06211
https://arxiv.org/abs/1811.11168

Train the compact ResNet-50 of resnet_bottleneck with v1 and with v2
(modulated) deformable 3x3 convolutions in its last two stages (8x8 and
4x4 maps) on CIFAR-10, for comparison with the plain compact ResNet-50. Use
12k fit / 5k validation, twelve epochs, crop/flip, Adam(.001), kernel
L2(.0001) and the best-validation checkpoint before evaluating all 10k test
images. Weights come from nn's generator, so trained numbers match TF only
statistically. CIFAR-10 is read from the tflow dataset cache
(common/paths.hpp) or data=.
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

The components (deformable::DeformableConv2D, attention_cnn::AttentionResNet,
mobile_checks::Pattern and GradientNorms, image batching, cross-entropy and
L2) are the C++ ones through SWIG (pydatamunge.nn_components); the checks and
the experiment workflow below mirror examples/cpp/source/nn/.../deformable_conv.cpp
line by line.
"""
import math
import struct
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import (attention_cnn, check, deformable, example_runtime, image_training,
                                       mobile_checks, paths)

Pattern = mobile_checks.Pattern

K_SIZE, K_IN, K_OUT = 7, 3, 4


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def f32(value):
    """Single-precision rounding, where the C++ holds a `float`."""
    return struct.unpack("f", struct.pack("f", value))[0]


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


def sample(x, y, xpos, c):
    """Host bilinear sample of x [1, H, W, C] (flattened) at (y, x), zero outside."""
    y0, x0 = math.floor(y), math.floor(xpos)
    v = 0.0
    for dy in range(2):
        for dx in range(2):
            yy, xx = y0 + dy, x0 + dx
            if yy < 0 or yy >= K_SIZE or xx < 0 or xx >= K_SIZE:
                continue
            v += ((y - y0) if dy else (1 - (y - y0))) * ((xpos - x0) if dx else (1 - (xpos - x0))) * \
                x[(yy * K_SIZE + xx) * K_IN + c]
    return v


def max_difference(a, b):
    """Largest absolute single-precision difference between two flattened tensors."""
    e = 0.0
    for ai, bi in zip(a, b):
        e = max(e, abs(f32(ai - bi)))
    return e


def layer_checks():
    """A new (zero-offset) deformable layer against the ordinary 3x3 convolution,
    integer and fractional offsets against host sampling, offset derivatives
    against central differences."""
    x = Pattern([1, K_SIZE, K_SIZE, K_IN], 1.0, .2)
    for stride in (1, 2):
        for modulated in (False, True):
            if stride == 2 and modulated:
                continue
            p = nn.ParameterStore(17)
            layer = deformable.DeformableConv2D(p, "dcn", x, K_OUT, 3, stride, modulated)
            o = nn.conv_options(stride, nn.Padding_Explicit)
            o.explicit_paddings = [1, 1, 1, 1]
            reference = nn.mul(nn.conv2d(x, p.at("dcn/W"), o), .5 if modulated else 1.)
            check.Require(tuple(reference.shape()) == tuple(layer.output.shape()), "deformable output shape")
            error = max_difference(layer.output.values(), reference.values())
            check.Near(error, 0, 1e-5, "a new deformable layer is an ordinary (or half an ordinary) convolution")
            print(f"  new {'modulated ' if modulated else ''}deformable layer, stride {stride}: equals "
                  f"{'half the ' if modulated else 'the '}ordinary 3x3 convolution (error {g(error)})", flush=True)
    # Offsets set through the offset predictor's bias.
    p = nn.ParameterStore(17)
    weights = Pattern([1, K_SIZE, K_SIZE, K_OUT], 1.0, .6)

    def loss_of(bias):
        if p.contains("dcn/offset/b"):
            with nn.no_grad():
                p.set("dcn/offset/b", nn.from_vector(bias, [18]))
        layer = deformable.DeformableConv2D(p, "dcn", x, K_OUT, 3, 1, False)
        return layer.output, nn.sum(nn.mul(layer.output, weights))

    loss_of([0.0] * 18)  # create the parameters
    value, w = x.values(), p.at("dcn/W").values()

    def host(b):
        y = [0.0] * (K_SIZE * K_SIZE * K_OUT)  # single-precision accumulation, as the C++ host vector
        for i in range(K_SIZE):
            for j in range(K_SIZE):
                for k in range(9):
                    for c in range(K_IN):
                        v = sample(value, i - 1 + k // 3 + b[2 * k], j - 1 + k % 3 + b[2 * k + 1], c)
                        for o in range(K_OUT):
                            index = (i * K_SIZE + j) * K_OUT + o
                            y[index] = f32(y[index] + v * w[(k * K_IN + c) * K_OUT + o])
        return y

    down = [0.0] * 18
    for k in range(9):
        down[2 * k] = 1.0
    integer = max_difference(loss_of(down)[0].values(), host(down))
    fraction = []
    # Fractional parts in [0.2, 0.8]: central differences then stay on one bilinear piece.
    for i in range(18):
        v = f32(.37 + .9 * math.sin(1.7 * i) + (.25 if i % 3 == 0 else 0.0))
        fraction.append(f32(math.floor(v) + .2 + .6 * (v - math.floor(v))))
    r_output, r_loss = loss_of(fraction)
    fractional = max_difference(r_output.values(), host(fraction))
    symbolic = nn.grad(r_loss, [p.at("dcn/offset/b")])[0].values()
    derivative = 0.0
    with nn.no_grad():
        for i in range(18):
            plus, minus = list(fraction), list(fraction)
            h = f32(1e-2)
            plus[i] = f32(plus[i] + h)
            minus[i] = f32(minus[i] - h)
            numeric = f32(f32(f32(loss_of(plus)[1].item()) - f32(loss_of(minus)[1].item())) / (2 * h))
            derivative = max(derivative, abs(numeric - symbolic[i]) / max(1., abs(numeric)))
    check.Near(integer, 0, 1e-5, "integer offsets read the shifted input")
    check.Near(fractional, 0, 1e-5, "fractional offsets interpolate bilinearly")
    check.Near(derivative, 0, 2e-3, "offset derivatives match central differences")
    print(f"  offsets: all taps one row down matches the shifted convolution (error {g(integer)}), fractional offsets "
          f"match host bilinear sampling ({g(fractional)}), offset derivatives match central differences "
          f"(relative {g(derivative)})", flush=True)


def compact_config(modulated):
    """The compact ResNet-50 of resnet_bottleneck with deformable 3x3 convolutions in stages 3 and 4."""
    cfg = attention_cnn.AttentionResNetConfig()
    cfg.classes = 10
    cfg.width_divisor = 4
    cfg.small_stem = True
    cfg.stride_on_3x3 = False
    cfg.deformable_from = 3
    cfg.modulated = modulated
    return cfg


def network_checks():
    """Parameter counts and finite nonzero derivatives of the compact deformable
    ResNet-50 (v1 and modulated v2), training mode with frozen moving averages."""
    for modulated in (False, True):
        p, state = nn.ParameterStore(17), nn.ParameterStore(17)
        cfg = compact_config(modulated)
        cfg.update_stats = False
        net = attention_cnn.AttentionResNet(p, state, Pattern([2, 32, 32, 3], 1.0, 0.0), cfg, True)
        # The plain compact ResNet-50 (1,484,186) plus offset predictors: 3x3 convolutions from the
        # 64 / 128 channels of the six stage-3 and three stage-4 layers to 18 (v1) or 27 (v2) maps.
        maps = 27 if modulated else 18
        expected = 1484186 + 6 * (9 * 64 * maps + maps) + 3 * (9 * 128 * maps + maps)
        check.Require(p.count() == expected, "compact deformable ResNet-50 parameter count")
        norms = mobile_checks.GradientNorms(p, nn.mean(nn.square(net.logits)))
        minimum, maximum, smallest = math.inf, 0.0, ""
        names = p.names()
        for i, norm in enumerate(norms):
            check.Require(math.isfinite(norm) and norm > 0, "finite nonzero deformable ResNet derivative")
            if norm < minimum:
                minimum, smallest = norm, names[i]
            maximum = max(maximum, norm)
        print(f"  compact ResNet-50 with {'v2 (modulated)' if modulated else 'v1'} deformable layers in stages 3-4: "
              f"{p.count()} parameters; derivatives in [{g(minimum)},{g(maximum)}] (smallest {smallest})", flush=True)


def train_deformable_resnet(modulated, development, test, cfg, explicit_epochs):
    """Fits the compact deformable ResNet-50 (plain DCNv1 or modulated DCNv2
    offsets) on CIFAR-10: fixed development split, Adam on cross-entropy plus
    L2, validation-NLL checkpoint selection, one test pass."""
    cfg.name = "dcnv2_resnet50_compact_cifar10" if modulated else "dcnv1_resnet50_compact_cifar10"

    def model(p, state, image, training):
        return attention_cnn.AttentionResNet(p, state, image, compact_config(modulated), training).logits

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
    for argument in arguments:  # threads= also applies to the checks
        if argument.startswith("threads="):
            nn.set_num_threads(int(argument[8:]))
    layer_checks()
    network_checks()
    cfg = image_training.ImageTrainingConfig()
    cfg.maximum_fit_rows = 12000
    cfg.epochs = 12
    cfg.minimum_test_accuracy = .4
    directory, explicit_epochs = parse_cifar_options(arguments, cfg)
    development, test = image_training.LoadCifar10(str(directory))
    for modulated in (False, True):
        train_deformable_resnet(modulated, development, test, cfg, explicit_epochs)
    print("All deformable convolution checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
