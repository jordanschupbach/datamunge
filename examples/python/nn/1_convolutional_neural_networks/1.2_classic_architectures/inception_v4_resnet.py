# Native Python NN example. inception_v4_resnet: residual-scale and full v4/Inception-ResNet-v2 checks, then compact CIFAR-10 training of both variants with an auxiliary head over SWIG components.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.2_classic_architectures/inception_v4_resnet.py
"""Inception v4 and Inception-ResNet-v2: full depth and scaled residual branches.

Port of tflow's inception_v4_resnet. Implement the complete TF-Slim layouts:
v4's 4/7/3 A/B/C blocks and the residual model's 10/20/10 blocks.
Inception-ResNet uses linear projections without BN; scales are
.17/.10/.20, except the last C block is unscaled and has no final ReLU. Both
include the reference auxiliary head. Validate every canonical stage,
independent main/auxiliary/state parameter counts, and gradients through all
learned tensors at 299 pixels and 1000 classes. Architecture and adaptation
details: common/inception4.hpp.

For each residual block type, compare shared-weight evaluations at scales
zero, s and 2s. Check the identity path, output scaling, parameter-gradient
scaling and the identity contribution to input gradients. This isolates the
residual mechanism without making an unsupported empirical stability claim.

Train compact CIFAR-10 demonstrations with every block retained: 107-pixel
input, one-eighth widths, 12k fit / 5k validation rows, twelve epochs,
Adam(.001), crop/flip augmentation, BN momentum .9, .8 dropout retention,
.1 label smoothing and .4-weight auxiliary CE. Select by unsmoothed primary
validation NLL and evaluate all 10k test rows once. These short CPU runs
are not full-model ImageNet benchmarks or a controlled residual ablation.
CIFAR-10 is read from paths::Dataset("cifar10"). BN moving averages update
eagerly in training passes; inference passes leave the state store unchanged.
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

The model components (inception4::Inception4, inception4::InceptionResidual,
inception23::InceptionBuilder, the architecture helpers, image batching,
cross-entropy and L2) are the C++ ones through SWIG (pydatamunge.nn_components);
the checks and the experiment workflow below mirror
examples/cpp/source/nn/.../inception_v4_resnet.cpp line by line.
"""
import math
import struct
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import architecture, check, example_runtime, image_training, inception23, inception4, paths


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def f32(value):
    """Rounds to IEEE single precision (the C++ fixtures are float arithmetic)."""
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


def residual_scale_checks(kind, channels, scale):
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    cfg = inception23.Inception23Config()
    cfg.width_divisor = 8
    result = inception23.Inception23Outputs()
    builder = inception23.InceptionBuilder(p, state, result, cfg, False)
    # .4f + .2f * std::sin(float(i) * .17f): single-precision fixture values.
    values = [f32(.4 + f32(.2 * f32(math.sin(f32(i * f32(.17)))))) for i in range(81 * channels)]
    input = nn.from_vector(values, [1, 9, 9, channels])
    input.requires_grad_(True)
    x = inception23.ConvFeature()
    x.value, x.channels = input, channels
    first = inception4.InceptionResidual(builder, "block", x, kind, scale, False).value
    parameters = p.count()
    twice = inception4.InceptionResidual(builder, "block", x, kind, 2 * scale, False).value
    zero = inception4.InceptionResidual(builder, "block", x, kind, 0, False).value
    check.Require(p.count() == parameters, "residual fixture shares parameters across scale choices")
    variables = [input] + list(p.parameters())
    g1, g2 = nn.grad(nn.sum(first), variables), nn.grad(nn.sum(twice), variables)
    a, b, z = first.values(), twice.values(), zero.values()
    error = 0.0
    for i in range(len(values)):
        check.Near(z[i], values[i], 0, "zero residual scale preserves identity")
        error = max(error, abs((b[i] - values[i]) - 2 * (a[i] - values[i])))
    for parameter in range(len(g1)):
        ga, gb = g1[parameter].values(), g2[parameter].values()
        skip = 1 if parameter == 0 else 0
        for i in range(len(ga)):
            error = max(error, abs((gb[i] - skip) - 2 * (ga[i] - skip)) / max(1., abs(gb[i])))
    check.Near(error, 0, 2e-5, "residual output, input and parameter derivative scaling")
    print(f"  residual {kind} at scale {g(scale)}: identity and scaling checks; max normalized error={g(error)}", flush=True)


def full_architecture_checks(residual):
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    cfg = inception4.Inception4Config()
    cfg.residual = residual
    net = inception4.Inception4(p, state, architecture.SinePixels([1, 299, 299, 3], .1, .13), cfg, False, True)
    aux_count = architecture.CountWithPrefix(p, "aux/")
    main_count = p.count() - aux_count
    check.Require(main_count == (55813192 if residual else 42648232) and aux_count == (3366760 if residual else 3358568),
                  "reference main/auxiliary parameter counts")
    check.Require(state.count() == (62336 if residual else 64960), "reference BN state count")
    check.Require(len(net.auxiliary) == 1, "canonical auxiliary head")
    initial_state = state.flatten()
    expected = [(149, 32), (147, 32), (147, 64)]
    if residual:
        expected += [(73, 64), (73, 80), (71, 192), (35, 192), (35, 320)]
        expected += [(35, 320)] * 10
        expected.append((17, 1088))
        expected += [(17, 1088)] * 20
        expected.append((8, 2080))
        expected += [(8, 2080)] * 10
        expected.append((8, 1536))
    else:
        expected.append((73, 160))
        expected.append((71, 192))
        expected.append((35, 384))
        expected += [(35, 384)] * 4
        expected.append((17, 1024))
        expected += [(17, 1024)] * 7
        expected.append((8, 1536))
        expected += [(8, 1536)] * 3
    stages = net.stages  # (name, tensor) pairs in construction order
    check.Require(len(stages) == len(expected), "complete architecture block count")
    for (spatial, channels), (name, stage) in zip(expected, stages):
        architecture.RequireShape(stage, [1, spatial, spatial, channels], "reference stage shape: " + name)
    check.Require(list(net.logits.shape()) == [1, 1000] and list(net.auxiliary[0].shape()) == [1, 1000],
                  "main and auxiliary classifier shapes")
    objective = nn.add(nn.mean(nn.square(net.logits)), nn.mul(nn.mean(nn.square(net.auxiliary[0])), .4))
    minimum, maximum = architecture.GradientNormRange(p, objective, "advanced Inception")
    check.Require(state.flatten() == initial_state, "inference does not update BN running statistics")
    print(f"  {'Inception-ResNet-v2' if residual else 'Inception v4'}: main={main_count}, aux={aux_count}, "
          f"state={state.count()}; {len(expected)} stage shapes; gradient norms [{g(minimum)},{g(maximum)}]", flush=True)


def compact_config(residual):
    cfg = inception4.Inception4Config()
    cfg.residual = residual
    cfg.image_size = 107
    cfg.classes = 10
    cfg.width_divisor = 8
    cfg.momentum = .9
    return cfg


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    residual_scale_checks('A', 40, .17)
    residual_scale_checks('B', 136, .1)
    residual_scale_checks('C', 260, .2)
    full_architecture_checks(False)
    full_architecture_checks(True)
    for residual in (False, True):
        compact = compact_config(residual)

        def model(p, state, x, training):
            return inception4.Inception4(p, state, nn.resize_bilinear(x, [107, 107], False, True), compact, training).logits

        # Training path: the same network with its auxiliary head (tflow's
        # ClassifierNetwork::auxiliary); returns (logits, [(auxiliary logits, weight)]).
        def training(p, state, x):
            net = inception4.Inception4(p, state, nn.resize_bilinear(x, [107, 107], False, True), compact, True, True)
            return net.logits, [(net.auxiliary[0], .4)]

        cfg = image_training.ImageTrainingConfig()
        cfg.name = "inception_resnet_v2_compact_cifar10" if residual else "inception_v4_compact_cifar10"
        cfg.maximum_fit_rows = 12000
        cfg.epochs = 12
        cfg.minimum_test_accuracy = .35
        cfg.label_smoothing = .1
        data_directory, explicit_epochs = parse_cifar_options(arguments, cfg)
        check.Require(cfg.classes == 10, "CIFAR-10 requires ten output classes")
        development, test = image_training.LoadCifar10(str(data_directory))
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
            x = image_training.ImageBatch(development, [fit[0], fit[1]])[0]
            model(params, state, x, False)
            if training:
                scratch = nn.ParameterStore(cfg.seed)
                training(params, scratch, x)
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
                # Augmented batch; training-mode forward; smoothed cross-entropy plus the weighted
                # auxiliary cross-entropy plus 0.5 * l2 * sum(W^2); one Adam step.
                x, y = image_training.ImageBatch(development, rows, augmentation if cfg.augment else None)
                logits, auxiliary = training(params, state, x) if training else (model(params, state, x, True), [])
                objective = image_training.TrainingCrossEntropy(logits, y, cfg.classes, cfg.label_smoothing)
                for aux, weight in auxiliary:
                    check.Require(math.isfinite(weight) and weight > 0, "positive finite auxiliary loss weight")
                    objective = nn.add(objective, nn.mul(
                        image_training.TrainingCrossEntropy(aux, y, cfg.classes, cfg.label_smoothing), weight))
                objective = image_training.WithL2(objective, params, cfg.l2)
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
        check.TrainingQuality(test_accuracy >= cfg.minimum_test_accuracy,
                              "image classifier demonstration accuracy threshold")
        train_total = sum(epoch_seconds)
        print(f"  training time {train_total:.1f} s ({train_total / cfg.epochs:.2f} s/epoch)")
        print(f"  best checkpoint and history: {directory}\nAll {cfg.name} checks passed.")
    print("All Inception v4 / Inception-ResNet-v2 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
