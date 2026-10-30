# Native Python NN example. inception_v2_v3: factorization/BN/label-smoothing checks, full v2/v3 architecture checks and compact CIFAR-10 training of both versions over SWIG components.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.2_classic_architectures/inception_v2_v3.py
"""Inception v2 and v3: batch normalization, factorization and label smoothing.

Port of tflow's inception_v2_v3. Implement full TF-Slim reference layouts:
v2 at 224 pixels, v3 at 299 pixels, with 1000 classes. Check every stage
shape, independently counted learned parameters and running statistics, and
finite nonzero parameter derivatives. v2 includes the separable 7x7 stem and
ten ordinary/reduction modules. v3 includes asymmetric 1x7/7x1 convolutions,
split 1x3/3x1 branches and an auxiliary head. Reference definitions and
initialization differences are documented in common/inception23.hpp.

Demonstrate linear spatial factorization with an outer-product 7x7 kernel:
two rectangular convolutions match the composed convolution and its input
and factor gradients. This equivalence needs a factorizable kernel and no
intervening nonlinearities; real Inception modules include BN/ReLU. Verify
beta-only BN against host moments, running-statistic updates, and smoothed
training cross entropy separately from unsmoothed evaluation NLL.

Train compact CIFAR-10 versions, retaining every block: quarter widths,
64-pixel v2 / 107-pixel v3 inputs, 12k fit / 5k validation rows, twelve
epochs, Adam(.001), crop/flip augmentation and .8 dropout retention. Use
momentum .9 for BN on these short runs; canonical configuration uses .9997.
v3 adds .1 label smoothing and .4-weight auxiliary cross entropy. Validation
selection and the final 10k-row test use only unsmoothed primary NLL.
These demonstrations are not full-model ImageNet benchmark results.
CIFAR-10 is read from paths::Dataset("cifar10"). Batch-norm moving averages
update eagerly during training passes (tflow runs collected update ops), so
"inference requests no updates" becomes "an inference pass leaves the state
store unchanged".
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

The model components (inception23::Inception23, normalized_conv::NormalizedConv,
the architecture helpers, image batching, cross-entropy and L2) are the C++ ones
through SWIG (pydatamunge.nn_components); the checks and the experiment workflow
below mirror examples/cpp/source/nn/.../inception_v2_v3.cpp line by line.
"""
import math
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import architecture, check, example_runtime, image_training, inception23, normalized_conv, paths


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


def factorization_checks():
    x = architecture.SinePixels([1, 11, 11, 1], 1, .31)
    vertical = nn.from_vector([.1, .2, -.1, .3, .1, -.2, .2], [7])
    horizontal = nn.from_vector([.3, -.1, .2, .1, .2, -.3, .1], [7])
    for t in (x, vertical, horizontal):
        t.requires_grad_(True)
    v, h = nn.reshape(vertical, [7, 1, 1, 1]), nn.reshape(horizontal, [1, 7, 1, 1])
    valid = nn.conv_options(1, nn.Padding.Valid)
    factored = nn.conv2d(nn.conv2d(x, h, valid), v, valid)
    composed = nn.conv2d(x, nn.mul(v, h), valid)
    keep = nn.GradOptions()
    keep.retain_graph = True  # both graphs share the reshaped factors
    a = nn.grad(nn.mean(nn.square(factored)), [x, vertical, horizontal], keep)
    b = nn.grad(nn.mean(nn.square(composed)), [x, vertical, horizontal])
    error = 0.0

    def compare(p, q):
        nonlocal error
        for pv, qv in zip(p.values(), q.values()):
            error = max(error, abs(pv - qv))

    compare(factored, composed)
    for i in range(len(a)):
        compare(a[i], b[i])
    check.Near(error, 0, 2e-6, "factorized output and input/factor gradients")
    print(f"  1x7 then 7x1 vs outer-product 7x7: output/gradient max error={g(error)}; "
          f"14 vs 49 coefficients per channel pair at equal intermediate width")


def normalization_checks():
    p, state = nn.ParameterStore(5), nn.ParameterStore(5)
    cfg = normalized_conv.ConvNormConfig()
    cfg.training = True
    cfg.momentum = .9
    input = architecture.SinePixels([2, 4, 4, 1], 1, .27)
    output = normalized_conv.NormalizedConv(p, state, "rect", input, 2, 1, 3, cfg, 1, "VALID", False).values()
    convolution = nn.conv2d(input, p.at("rect/W"), nn.conv_options(1, nn.Padding.Valid)).values()
    check.Require(p.count() == 8 and state.count() == 4, "beta-only BN parameters and running state")
    mean_state, var_state = state.at("rect/bn/moving_mean").values(), state.at("rect/bn/moving_var").values()
    rows = len(output) // 2
    for c in range(2):
        mean = variance = 0.0
        for i in range(rows):
            mean += convolution[2 * i + c] / rows
        for i in range(rows):
            variance += (convolution[2 * i + c] - mean) ** 2 / rows
        for i in range(rows):
            check.Near(output[2 * i + c], (convolution[2 * i + c] - mean) / math.sqrt(variance + .001), 2e-6,
                       "fixed-gamma BN output")
        check.Near(mean_state[c], .1 * mean, 2e-6, "one BN mean update")
        check.Near(var_state[c], .9 + .1 * variance * rows / (rows - 1), 2e-6, "one unbiased-variance update")
    print("  rectangular convolution and beta-only BN: host output/moments and one EMA update passed")


def smoothing_checks():
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    p.set("W", nn.from_vector([10, 0, 0], [1, 3]))

    def model(q, _state, x, _training):
        return nn.matmul(x, q.at("W"))

    cfg = image_training.ImageTrainingConfig()
    cfg.classes = 3
    cfg.l2 = 0
    cfg.label_smoothing = .1
    x, y = nn.ones([1, 1]), nn.from_vector_int64([0], [1])
    with nn.no_grad():
        evaluation = nn.sparse_softmax_cross_entropy(model(p, state, x, False), y).item()
        training = training_objective(p, state, model, x, y, cfg).item()
    nll = math.log1p(2 * math.exp(-10.))
    check.Near(evaluation, nll, 2e-7, "evaluation retains unsmoothed NLL")
    check.Near(training, nll + 2. / 3, 2e-6, "uniform label smoothing cross entropy")
    config = nn.OptimizerConfig()
    config.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, .01, config)
    optimizer.minimize(training_objective(p, state, model, x, y, cfg))
    check.Require(p.at("W").values()[0] < 10, "smoothing reduces an overconfident correct-class score")
    print(f"  .1 label smoothing: evaluation NLL={g(nll)}; training CE={g(nll + 2. / 3)}; correct-class score decreases")


def full_architecture_checks(version):
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    cfg = inception23.Inception23Config()
    cfg.version = version
    cfg.image_size = 224 if version == 2 else 299
    pixels = architecture.SinePixels([1, cfg.image_size, cfg.image_size, 3], .1, .13)
    net = inception23.Inception23(p, state, pixels, cfg, False, True)
    aux_count = architecture.CountWithPrefix(p, "aux/")
    main_count = p.count() - aux_count
    check.Require(main_count == (11178336 if version == 2 else 23817352) and aux_count == (0 if version == 2 else 3325800),
                  "reference learned parameter counts")
    check.Require(state.count() == (19776 if version == 2 else 36224), "reference running-state count")
    initial_state = state.flatten()
    if version == 2:
        expected = [(112, 64), (56, 64), (56, 64), (56, 192), (28, 192), (28, 256), (28, 320), (14, 576), (14, 576),
                    (14, 576), (14, 576), (14, 576), (7, 1024), (7, 1024), (7, 1024)]
    else:
        expected = [
            (149, 32),
            (147, 32),
            (147, 64),
            (73, 64),
            (73, 80),
            (71, 192),
            (35, 192),
            (35, 256),
            (35, 288),
            (35, 288),
            (17, 768),
            (17, 768),
            (17, 768),
            (17, 768),
            (17, 768),
            (8, 1280),
            (8, 2048),
            (8, 2048)]
    stages = net.stages  # (name, tensor) pairs in construction order
    check.Require(len(stages) == len(expected), "reference stage count")
    for (spatial, channels), (name, stage) in zip(expected, stages):
        architecture.RequireShape(stage, [1, spatial, spatial, channels], "reference Inception stage shape: " + name)
    objective = nn.mean(nn.square(net.logits))
    for aux in net.auxiliary:
        objective = nn.add(objective, nn.mul(nn.mean(nn.square(aux)), .4))
    minimum, maximum = architecture.GradientNormRange(p, objective, "Inception")
    check.Require(state.flatten() == initial_state, "inference does not update BN running statistics")
    print(f"  full Inception v{version}: main={main_count}, aux={aux_count}, running state={state.count()}; "
          f"{len(expected)} stage shapes; gradient norms [{g(minimum)},{g(maximum)}]", flush=True)


def compact_config(version):
    cfg = inception23.Inception23Config()
    cfg.version = version
    cfg.image_size = 64 if version == 2 else 107
    cfg.classes = 10
    cfg.width_divisor = 4
    cfg.momentum = .9
    return cfg


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    factorization_checks()
    normalization_checks()
    smoothing_checks()
    full_architecture_checks(2)
    full_architecture_checks(3)
    for version in (2, 3):
        cfg_model = compact_config(version)

        def model(p, state, x, training):
            image = nn.resize_bilinear(x, [cfg_model.image_size, cfg_model.image_size], False, True)
            return inception23.Inception23(p, state, image, cfg_model, training).logits

        training = None
        if version == 3:
            # Training path: the same network with its auxiliary head (tflow's
            # ClassifierNetwork::auxiliary); returns (logits, [(auxiliary logits, weight)]).
            def training(p, state, x):
                image = nn.resize_bilinear(x, [cfg_model.image_size, cfg_model.image_size], False, True)
                net = inception23.Inception23(p, state, image, cfg_model, True, True)
                return net.logits, [(net.auxiliary[0], .4)]
        cfg = image_training.ImageTrainingConfig()
        cfg.name = f"inception_v{version}_compact_cifar10"
        cfg.maximum_fit_rows = 12000
        cfg.epochs = 12
        cfg.minimum_test_accuracy = .35
        cfg.label_smoothing = .1 if version == 3 else 0.
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
                # Augmented batch; training-mode forward; (smoothed) cross-entropy plus the weighted
                # auxiliary cross-entropies plus 0.5 * l2 * sum(W^2); one Adam step.
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
    print("All Inception v2/v3 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
