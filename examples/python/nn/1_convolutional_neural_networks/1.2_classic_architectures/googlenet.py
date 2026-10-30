# Native Python NN example. googlenet: Inception v1 branch/auxiliary/full-architecture checks and compact CIFAR-10 training with auxiliary heads over SWIG components.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.2_classic_architectures/googlenet.py
"""GoogLeNet / Inception v1: parallel receptive fields and auxiliary supervision.

Port of tflow's googlenet. Implement the nine modules and 224-pixel/1000-class
widths in Table 1 of https://arxiv.org/abs/1409.4842. Check all stage
dimensions, 6,998,552 main parameters and 6,379,728 auxiliary parameters,
and finite nonzero gradients through every learned tensor. Independently
check branch filtering and concatenation against host arithmetic, including
padded image boundaries. Measure how the 1x1 reductions change
spatial-kernel multiply-add counts.

The two auxiliary heads follow modules 4a/4d: 5x5 stride-3 average pooling,
1x1 projection, a dense hidden layer, 70% dropout, and class logits. Their
cross entropies each have weight .3 during training. Verify weighted losses
against host log-sum-exp, auxiliary parameter updates with L2 enabled, and
unchanged inference after perturbing auxiliary parameters. Main dropout is
40%. See common/googlenet.hpp for padding, LRN and initialization conventions.

Train a separate CPU demonstration on CIFAR-10: resize to 96 pixels, quarter
convolution widths, 128-unit auxiliary hidden layers, all nine modules,
12k fit / 5k validation rows, crop/flip augmentation, Adam(.001), twelve
epochs. Select by primary validation NLL and evaluate the complete 10k test
set. This compact demonstration is not an ImageNet benchmark.
CIFAR-10 is read from paths::Dataset("cifar10"); auxiliary heads train through
the written-out training objective (tflow's ClassifierNetwork::auxiliary).
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

The model components (googlenet::GoogLeNet, inception::InceptionV1, the
architecture helpers, image batching, cross-entropy and L2) are the C++ ones
through SWIG (pydatamunge.nn_components); the checks and the experiment
workflow below mirror examples/cpp/source/nn/.../googlenet.cpp line by line.
"""
import math
import struct
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import architecture, check, example_runtime, googlenet, image_training, inception, paths


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


def inception_widths(one, reduce_three, three, reduce_five, five, pool):
    """inception::InceptionWidths{...}."""
    widths = inception.InceptionWidths()
    widths.one, widths.reduce_three, widths.three = one, reduce_three, three
    widths.reduce_five, widths.five, widths.pool = reduce_five, five, pool
    return widths


# googlenet::GoogLeNetWidths()[0] (the std::array accessor is not bound; Table 1, module 3a).
GOOGLENET_WIDTHS_3A = (64, 96, 128, 16, 32, 32)


def branch_checks():
    p = nn.ParameterStore(0)
    pixels = [0.0] * 50
    for r in range(5):
        for c in range(5):
            pixels[(r * 5 + c) * 2] = f32(f32(1.0 + f32(r * f32(.2))) + f32(c * f32(.1)))
            pixels[(r * 5 + c) * 2 + 1] = f32(f32(2.0 + f32(r * f32(.1))) + f32(c * f32(.3)))
    input = nn.from_vector(pixels, [1, 5, 5, 2])
    widths = inception_widths(2, 1, 1, 1, 1, 1)
    with nn.no_grad():
        inception.InceptionV1(p, "block", input, widths)
        names, parameters = p.names(), p.parameters()
        for i in range(p.size()):
            name = names[i]
            value = [0.0] * parameters[i].numel()
            if name == "block/one/W":
                value[0], value[3] = 1, 1
            elif name == "block/three_reduce/W":
                value[0], value[1] = 1, 2
            elif name == "block/five_reduce/W":
                value[0], value[1] = 3, 1
            elif name in ("block/three/W", "block/five/W"):
                value = [.1] * len(value)
            elif name == "block/pool/W":
                value[0], value[1] = 5, 6
            p.set(name, nn.from_vector(value, parameters[i].shape()))
    net = inception.InceptionV1(p, "block", input, widths)
    architecture.RequireShape(net.concatenated, [1, 5, 5, 5], "Inception branch channel sum")
    out = net.concatenated.values()
    error = 0.0

    def at(y, x, c):
        return pixels[(y * 5 + x) * 2 + c]

    for r in range(5):
        for c in range(5):
            expected = [at(r, c, 0), at(r, c, 1), 0.0, 0.0, 0.0]
            m0 = m1 = 0.0
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    y, x = r + dr, c + dc
                    if y < 0 or y >= 5 or x < 0 or x >= 5:
                        continue
                    a, b = at(y, x, 0), at(y, x, 1)
                    expected[3] += .1 * (3 * a + b)
                    if abs(dr) <= 1 and abs(dc) <= 1:
                        expected[2] += .1 * (a + 2 * b)
                        m0 = max(m0, a)
                        m1 = max(m1, b)
            expected[4] = 5 * m0 + 6 * m1
            for channel in range(5):
                error = max(error, abs(out[(r * 5 + c) * 5 + channel] - expected[channel]))
    check.Near(error, 0, 1e-5, "four branches and concatenation match independent boundary-aware arithmetic")
    w = inception_widths(*GOOGLENET_WIDTHS_3A)
    reduced = w.KernelWeights(192)
    direct = 192 * (w.one + 9 * w.three + 25 * w.five + w.pool)
    check.Require(reduced == 163328 and direct == 393216, "first-module kernel MACs per spatial position")
    print(f"  branch arithmetic max error={g(error)}; Inception 3a MACs/position: reduced={reduced}, "
          f"direct spatial convolutions={direct} (pool comparisons excluded)")


def cross_entropy(scores, labels):
    """Host mean log-sum-exp cross-entropy of [rows, columns] scores against int64 labels."""
    values = scores.values()
    rows, columns = scores.shape()[0], scores.shape()[1]
    total = 0.0
    for r in range(rows):
        largest, z = -math.inf, 0.0
        for c in range(columns):
            largest = max(largest, values[r * columns + c])
        for c in range(columns):
            z += math.exp(values[r * columns + c] - largest)
        total += largest + math.log(z) - values[r * columns + labels[r]]
    return total / rows


def training_objective(params, state, model, x, y, cfg, training=None):
    """image_training::TrainingObjective written out over the bound components: (smoothed)
    cross-entropy of the training logits, plus weight * cross-entropy of every auxiliary head,
    plus 0.5 * l2 * sum(W^2) over parameters of rank >= 2."""
    logits, auxiliary = training(params, state, x) if training else (model(params, state, x, True), [])
    objective = image_training.TrainingCrossEntropy(logits, y, cfg.classes, cfg.label_smoothing)
    for aux, weight in auxiliary:
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("auxiliary loss weight must be positive and finite")
        objective = nn.add(objective, nn.mul(image_training.TrainingCrossEntropy(aux, y, cfg.classes, cfg.label_smoothing), weight))
    return image_training.WithL2(objective, params, cfg.l2)


def auxiliary_loss_checks():
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)

    def model(q, _state, x, _training):
        return nn.dense(q, "main", x, 3)

    def training(q, _state, x):
        # TrainingNetwork{main logits, {{aux logits, .3}}}: "main" parameters are created before "aux".
        logits = nn.dense(q, "main", x, 3)
        return logits, [(nn.dense(q, "aux", x, 3), .3)]

    x = nn.from_vector([1, -.2, -.4, .7], [2, 2])
    labels = [0, 2]
    y = nn.from_vector_int64(labels, [2])
    cfg = image_training.ImageTrainingConfig()
    cfg.classes = 3
    cfg.l2 = 0  # tflow's training_loss excludes the L2 term
    with nn.no_grad():
        training(p, state, x)

    def losses():
        with nn.no_grad():
            main, aux = nn.dense(p, "main", x, 3), nn.dense(p, "aux", x, 3)
            loss = training_objective(p, state, model, x, y, cfg, training).item()
            check.Near(loss, cross_entropy(main, labels) + .3 * cross_entropy(aux, labels), 2e-6,
                       "primary plus weighted auxiliary cross entropy")
            return loss

    def inference():
        with nn.no_grad():
            return nn.softmax(model(p, state, x, False), 1).values()

    before = inference()
    old_loss = losses()
    p.set("aux/b", nn.from_vector([3, -2, 1], [3]))
    after = inference()
    new_loss = losses()
    for i in range(6):
        check.Near(before[i], after[i], 0, "auxiliary head cannot change inference")
    check.Require(abs(old_loss - new_loss) > .01, "auxiliary head changes the training objective")
    # Biases are excluded from L2: their movement demonstrates auxiliary CE
    # connectivity even when regularization is enabled.
    cfg.l2 = .1
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .01, config)
    optimizer.minimize(training_objective(p, state, model, x, y, cfg, training))
    check.Require(abs(p.at("aux/b").values()[0] - 3.0) > .001,
                  "auxiliary cross entropy updates its parameters with L2 enabled")
    print(f"  auxiliary loss: {g(old_loss)} -> {g(new_loss)} after aux-only perturbation; "
          f"inference unchanged; training updates aux bias")


def full_architecture_checks():
    p = nn.ParameterStore(17)
    net = googlenet.GoogLeNet(p, architecture.SinePixels([1, 224, 224, 3], .1, .13), googlenet.GoogLeNetConfig(), False, True)
    aux_count = architecture.CountWithPrefix(p, "aux")
    main_count = p.count() - aux_count
    check.Require(main_count == 6998552 and aux_count == 6379728, "canonical main and auxiliary parameter counts")
    objective = nn.mean(nn.square(net.logits))
    for aux in net.auxiliary:
        objective = nn.add(objective, nn.mul(nn.mean(nn.square(aux)), .3))
    stages = [net.stem(i) for i in range(3)] + [net.inception(i) for i in range(9)] + [net.logits] + list(net.auxiliary)
    shapes = [
        [1, 112, 112, 64],
        [1, 56, 56, 64],
        [1, 56, 56, 192],
        [1, 28, 28, 256],
        [1, 28, 28, 480],
        [1, 14, 14, 512],
        [1, 14, 14, 512],
        [1, 14, 14, 512],
        [1, 14, 14, 528],
        [1, 14, 14, 832],
        [1, 7, 7, 832],
        [1, 7, 7, 1024],
        [1, 1000],
        [1, 1000],
        [1, 1000]]
    check.Require(len(stages) == len(shapes), "GoogLeNet stage count")
    for stage, shape in zip(stages, shapes):
        architecture.RequireShape(stage, shape, "GoogLeNet stage shape")
    minimum, maximum = architecture.GradientNormRange(p, objective, "GoogLeNet parameter tensor")
    print(f"  full GoogLeNet: main={main_count}, auxiliary={aux_count}, total={p.count()}; "
          f"all 15 stage shapes passed; gradient norms [{g(minimum)},{g(maximum)}]")


def compact_config():
    cfg = googlenet.GoogLeNetConfig()
    cfg.image_size = 96
    cfg.classes = 10
    cfg.width_divisor = 4
    cfg.auxiliary_hidden = 128
    return cfg


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    branch_checks()
    auxiliary_loss_checks()
    full_architecture_checks()
    cfg = image_training.ImageTrainingConfig()
    cfg.name = "googlenet_compact_cifar10"
    cfg.maximum_fit_rows = 12000
    cfg.epochs = 12
    cfg.minimum_test_accuracy = .35
    data, explicit_epochs = parse_cifar_options(arguments, cfg)
    development, test = image_training.LoadCifar10(str(data))

    # Compact GoogLeNet on CIFAR-10: fixed development split, Adam on the primary
    # cross-entropy plus the two .3-weight auxiliary heads plus L2, validation-NLL
    # checkpoint selection on the primary logits, one test pass. The whole workflow
    # is written out here, line by line with the C++ example, over the same bound components.
    def model(p, _state, x, training):
        return googlenet.GoogLeNet(p, nn.resize_bilinear(x, [96, 96], False, True), compact_config(), training).logits

    # Training path: the same network with its auxiliary heads (tflow's
    # ClassifierNetwork::auxiliary); returns (logits, [(auxiliary logits, weight), ...]).
    def training(p, _state, x):
        net = googlenet.GoogLeNet(p, nn.resize_bilinear(x, [96, 96], False, True), compact_config(), True, True)
        return net.logits, [(net.auxiliary[0], .3), (net.auxiliary[1], .3)]

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
        # Training-only parameters (auxiliary heads); their batch-norm statistics
        # go to a scratch store so this pass updates no real moving averages.
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
            # Augmented batch; training-path forward; cross-entropy of the primary logits plus the
            # weighted auxiliary cross-entropies, plus 0.5 * l2 * sum(W^2); one Adam step.
            x, y = image_training.ImageBatch(development, rows, augmentation if cfg.augment else None)
            logits, auxiliary = training(params, state, x)
            objective = image_training.TrainingCrossEntropy(logits, y, cfg.classes, cfg.label_smoothing)
            for aux, weight in auxiliary:
                check.Require(math.isfinite(weight) and weight > 0, "auxiliary loss weight must be positive and finite")
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
    check.TrainingQuality(test_accuracy >= cfg.minimum_test_accuracy, "image classifier demonstration accuracy threshold")
    train_total = sum(epoch_seconds)
    print(f"  training time {train_total:.1f} s ({train_total / cfg.epochs:.2f} s/epoch)")
    print(f"  best checkpoint and history: {directory}\nAll {cfg.name} checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
