# Native Python NN example. MobileNet v3 hard-activation/SE/cost/gradient checks and Large/Small CIFAR-10 training; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mobilenet_v3.py
"""MobileNet v3 Large and Small: searched blocks, hard-swish and squeeze-and-excitation.

Compare hard-swish and the hard sigmoid with swish and the logistic
function they replace (exact zeros below -3, exact identity / one above 3).
Check that a squeeze-and-excitation module with a zeroed second layer
scales every channel by relu6(bias+3)/6. Measure the released Large and
Small models at 224 pixels: 5,483,032 / 2,542,856 learned parameters
(torchvision's counts) and 216.6M / 56.5M convolution and dense
multiply-adds (the paper reports about 219M / 56M). Check every derivative
in training mode with shifted BN betas and SE biases: all are clearly
nonzero except the projection BN shifts, whose per-channel constants only
reach 1x1 convolutions followed by training-mode BN.
https://arxiv.org/abs/1905.02244
https://github.com/pytorch/vision/blob/main/torchvision/models/mobilenetv3.py

Train Large and Small on CIFAR-10 with a stride-one first convolution
(every block, remaining downsamplings and the efficient last stage). Use
12k fit / 5k validation, twelve epochs, crop/flip, Adam(.001), kernel
L2(.0001) and the best-validation checkpoint before evaluating all 10k
test images. These are CIFAR adaptations, not the ImageNet benchmark.
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1.

The model components (mobilenet::MobileNetV3, HardSwish/HardSigmoid,
SqueezeExcite, mobile_checks fixtures, image batching, cross-entropy and L2)
are the C++ ones through SWIG (pydatamunge.nn_components); the checks and the
experiment workflow below mirror examples/cpp/source/nn/.../mobilenet_v3.cpp line by line.
"""
import math
import struct
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import check, example_runtime, image_training, mobile_checks, mobilenet, paths


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def f32(value):
    """Rounds a double to float32, as the C++ float arithmetic does."""
    return struct.unpack("f", struct.pack("f", value))[0]


def copy_config(cfg):
    """ImageTrainingConfig is passed by value to each size in the C++; copy every field."""
    copy = image_training.ImageTrainingConfig()
    for field in ("name", "classes", "epochs", "batch", "validation_rows", "maximum_fit_rows", "learning_rate",
                  "l2", "label_smoothing", "augment", "minimum_test_accuracy", "seed"):
        setattr(copy, field, getattr(cfg, field))
    return copy


def parse_cifar_options(arguments, cfg):
    """image_training::ParseCifarOption over key=value arguments; returns (data directory, explicit epochs)."""
    directory, explicit_epochs = paths.Dataset("cifar10"), False
    for argument in arguments:
        key, separator, value = argument.partition("=")
        if not separator:
            raise ValueError("expected key=value option")
        explicit_epochs = explicit_epochs or key == "epochs"
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
            raise ValueError("unknown option: " + argument)
    return directory, explicit_epochs


def activation_checks():
    # values[i] = -8 + .01f * i in float arithmetic.
    values = [f32(-8 + f32(f32(.01) * i)) for i in range(1601)]
    x = nn.from_vector(values, [1601])
    hs, hsig = mobilenet.HardSwish(x).values(), mobilenet.HardSigmoid(x).values()
    sw, sig = nn.mul(x, nn.sigmoid(x)).values(), nn.sigmoid(x).values()
    swish = sigmoid = saturation = 0.
    for i in range(1601):
        swish = max(swish, abs(f32(hs[i] - sw[i])))
        sigmoid = max(sigmoid, abs(f32(hsig[i] - sig[i])))
        if values[i] <= -3:
            saturation = max(saturation, abs(hs[i]), abs(hsig[i]))
        if values[i] >= 3:
            saturation = max(saturation, abs(f32(hs[i] - values[i])), abs(f32(hsig[i] - 1)))
    check.Near(saturation, 0, 0, "hard activations saturate exactly")
    check.Require(swish < .15 and sigmoid < .1, "hard activation approximation")
    print(f"hard activation approximation errors {g(swish)} {g(sigmoid)}", flush=True)


def squeeze_excite_checks():
    p = nn.ParameterStore(17)
    x = mobile_checks.Pattern([2, 5, 5, 8], 1, .4)
    mobilenet.SqueezeExcite(p, "se", x, 8)
    p.set("se/expand/W", nn.zeros(list(p.at("se/expand/W").shape())))
    bias = [float(-4 + i) for i in range(8)]
    p.set("se/expand/b", nn.from_vector(bias, [8]))
    out = mobilenet.SqueezeExcite(p, "se", x, 8).output.values()
    inp = x.values()
    error = 0.
    for i in range(len(out)):
        error = max(error, abs(out[i] - min(max(-4 + i % 8 + 3., 0.), 6.) / 6 * inp[i]))
    check.Near(error, 0, 1e-6, "squeeze-excitation channel scaling")


def cost_checks():
    for large, expected_params, expected_state, expected_macs in (
            (True, 5483032, 24400, 216589760), (False, 2542856, 12112, 56510400)):
        with nn.no_grad():
            p, state = nn.ParameterStore(17), nn.ParameterStore(17)
            cfg = mobilenet.MobileNetV3Config()
            cfg.large = large
            net = mobilenet.MobileNetV3(p, state, nn.zeros([1, 224, 224, 3]), cfg)
            macs = net.classifier_macs
            for layer in net.layers:
                macs += layer.Macs()
            check.Require(p.count() == expected_params and state.count() == expected_state and macs == expected_macs,
                          "MobileNet v3 parameters and MACs")
            print(f"MobileNet v3 large={int(large)} parameters={p.count()} MACs={macs}", flush=True)


def gradient_checks(large):
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    cfg = mobilenet.MobileNetV3Config()
    cfg.update_stats = False
    cfg.large = large
    x = mobile_checks.Pattern([2, 224, 224, 3], 1, 0)
    with nn.no_grad():
        mobilenet.MobileNetV3(p, state, x, cfg, True)
        mobile_checks.ShiftBetas(p, True)
    net = mobilenet.MobileNetV3(p, state, x, cfg, True)
    check.Require(list(net.logits.shape()) == [2, 1000], "MobileNet v3 output dimensions")
    norms = list(mobile_checks.GradientNorms(p, nn.mean(nn.square(net.logits))))
    maximum, minimum, removed, count = max(norms), math.inf, 0., 0
    names = p.names()
    for i in range(len(norms)):
        check.Require(math.isfinite(norms[i]), "finite MobileNet v3 derivative")
        if mobile_checks.EndsWith(names[i], "/project/bn/beta"):
            removed = max(removed, norms[i])
            count += 1
        else:
            minimum = min(minimum, norms[i])
    check.Require(count == len(mobilenet.MobileNetV3Blocks(large)) and removed < 1e-6 * maximum,
                  "vanishing projection BN shifts")
    check.Require(minimum > 1e-5 * maximum, "nonzero MobileNet v3 derivatives with shifted BN")
    print(f"gradient range {g(minimum)} to {g(maximum)} removed shifts {g(removed)}", flush=True)


def train_compact_mobilenet_v3(large, development, test, cfg, explicit_epochs):
    """Fits one CIFAR MobileNet v3 size: fixed development split, Adam on
    cross-entropy plus L2, validation-NLL checkpoint selection, one test pass."""
    cfg = copy_config(cfg)
    cfg.name = "mobilenet_v3_large_cifar10" if large else "mobilenet_v3_small_cifar10"

    def model(params, state, image, training):
        options = mobilenet.MobileNetV3Config()
        options.large = large
        options.classes = 10
        options.small_stem = True
        net = mobilenet.MobileNetV3(params, state, image, options, training)
        check.Require(params.count() == (4214842 if large else 1528106), "CIFAR MobileNet v3 parameter count")
        return net.logits

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
    nn.set_num_threads(2)
    activation_checks()
    squeeze_excite_checks()
    cost_checks()
    gradient_checks(True)
    gradient_checks(False)
    cfg = image_training.ImageTrainingConfig()
    cfg.maximum_fit_rows = 12000
    cfg.epochs = 12
    cfg.minimum_test_accuracy = .4
    directory, explicit_epochs = parse_cifar_options(arguments, cfg)
    development, test = image_training.LoadCifar10(str(directory))
    for large in (True, False):
        train_compact_mobilenet_v3(large, development, test, cfg, explicit_epochs)
    print("All MobileNet v3 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
