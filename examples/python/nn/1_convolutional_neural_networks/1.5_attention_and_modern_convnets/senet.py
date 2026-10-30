# Native Python NN example. senet: the C++ checks and training workflow over SWIG components.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.5_attention_and_modern_convnets/senet.py
import math
import os
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import check, example_runtime, image_training, paths, attention_cnn, mobile_checks


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


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


def module_checks():
    channels = 32
    p = nn.ParameterStore(17)
    x = mobile_checks.Pattern([2, 3, 3, channels], 1.0, 0.2)
    hidden = channels // 4
    with nn.no_grad():
        attention_cnn.SqueezeExcitationModule(p, "se", x, 4)
        for name in p.names():
            if mobile_checks.EndsWith(name, "/b"):
                p.set(name, mobile_checks.Pattern(p.at(name).shape(), .3, float(len(name))))
    net = attention_cnn.SqueezeExcitationModule(p, "se", x, 4)
    w1, b1 = p.at("se/reduce/W").values(), p.at("se/reduce/b").values()
    w2, b2 = p.at("se/expand/W").values(), p.at("se/expand/b").values()
    output = net.output.values()
    values = x.values()
    error = 0.0
    low, high = 1.0, 0.0
    for n in range(2):
        means = [sum(values[(n * 9 + i) * channels + c] / 9 for i in range(9))
                 for c in range(channels)]
        h = [max(0.0, b1[j] + sum(means[c] * w1[c * hidden + j] for c in range(channels))) for j in range(hidden)]
        gates = [1 / (1 + math.exp(-(b2[c] + sum(h[j] * w2[j * channels + c] for j in range(hidden)))))
                 for c in range(channels)]
        for c, gate in enumerate(gates):
            low, high = min(low, gate), max(high, gate)
            for i in range(9):
                k = (n * 9 + i) * channels + c
                error = max(error, abs(output[k] - gate * values[k]))
    check.Near(error, 0, 1e-6, "squeeze-and-excitation recalibration")
    print(f"  SE module (32 channels, reduction 4): output equals the host-computed gate times the input (error "
          f"{g(error)}); gates in [{g(low)},{g(high)}]", flush=True)


def full_checks():
    for name, cardinality, base, parameters in (("SE-ResNet-50", 1, 64, 28088024), ("SE-ResNeXt-50 32x4d", 32, 4, 27559896)):
        p, state = nn.ParameterStore(17), nn.ParameterStore(17)
        cfg = attention_cnn.AttentionResNetConfig()
        cfg.attention = attention_cnn.ChannelAttention_kSqueezeExcitation
        cfg.cardinality, cfg.base_width = cardinality, base
        cfg.update_stats = False
        with nn.no_grad():
            attention_cnn.AttentionResNet(p, state, nn.zeros([1, 64, 64, 3]), cfg)
        check.Require(p.count() == parameters, "SE-ResNet parameter count")
        if cardinality != 1:
            print(f"  {name}: {p.count()} parameters", flush=True)
            continue
        mobile_checks.ShiftBetas(p, False)
        net = attention_cnn.AttentionResNet(p, state, mobile_checks.Pattern([2, 224, 224, 3], 1.0, 0.0), cfg, True)
        norms = list(mobile_checks.GradientNorms(p, nn.mean(nn.square(net.logits))))
        minimum, maximum, smallest = math.inf, 0.0, ""
        for parameter_name, norm in zip(p.names(), norms):
            check.Require(math.isfinite(norm), "finite SE-ResNet derivative")
            if norm < minimum:
                minimum, smallest = norm, parameter_name
            maximum = max(maximum, norm)
        check.Require(minimum > 1e-5 * maximum, "clearly nonzero SE-ResNet derivatives")
        print(f"  {name}: {p.count()} parameters; training-mode gradient norms with shifted BN betas [{g(minimum)},{g(maximum)}] (smallest {smallest})", flush=True)


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    for argument in arguments:
        if argument.startswith("threads="):
            nn.set_num_threads(int(argument[8:]))
    module_checks()
    full_checks()
    cfg = image_training.ImageTrainingConfig()
    cfg.name = "se_resnet50_compact_cifar10"
    cfg.maximum_fit_rows, cfg.epochs, cfg.minimum_test_accuracy = 12000, 12, .4
    directory, explicit_epochs = parse_cifar_options(arguments, cfg)
    development, test = image_training.LoadCifar10(str(directory))
    model_cfg = attention_cnn.AttentionResNetConfig()
    model_cfg.classes, model_cfg.width_divisor = 10, 4
    model_cfg.small_stem, model_cfg.stride_on_3x3 = True, False
    model_cfg.attention = attention_cnn.ChannelAttention_kSqueezeExcitation


    def model(params, state, image, training):
        result = attention_cnn.AttentionResNet(params, state, image, model_cfg, training)
        check.Require(params.count() == 1645382, "compact SE-ResNet-50 parameter count")
        return result.logits

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
    print(f"  best checkpoint and history: {directory}\nAll {cfg.name} checks passed.")

    print("All SENet checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
