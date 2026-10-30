# Native Python NN example. Wide ResNet dropout/full-model checks and three WRN-16 CIFAR-10 trainings; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.3_residual_and_dense_families/wide_resnet.py
"""Wide ResNet: width scaling and dropout inside the residual branch.

Port of tflow's wide_resnet. Implement the authors' CIFAR 6n+4 layout with
preactivated basic blocks, fixed 16-channel stem and 16k/32k/64k stages.
Dropout follows the second BN/ReLU and precedes the second convolution;
shortcuts remain undropped. Check WRN-16-1, WRN-16-2 and the full
36,479,194-parameter WRN-28-10, including stage shapes and derivatives
through every learned parameter. Count convolution/classifier MACs for one
image; BN, ReLU, additions and pooling are excluded. These operation counts
are not measured latency.
https://arxiv.org/abs/1605.07146
https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua

Controlled checks verify inverted dropout values and derivatives, repeatable
inference and signed identity paths even with training dropout enabled.
Train three WRN-16 configurations on CIFAR-10: width 1 without dropout,
width 2 without dropout, and width 2 with dropout .3. Each uses the same
12k fit / 5k validation split, seed, twelve epochs, crop/flip, Adam(.001)
and kernel L2(.0001). Restore each lowest validation-NLL checkpoint and
evaluate 10k test images. These single-seed short runs illustrate the
comparison procedure; they do not establish an architecture ranking or
reproduce the paper's long SGD schedule. Initialization uses He-uniform.
Deviations: dropout masks come from one nn::Generator per model (tflow
seeds each block); weights come from nn's generator (seed=), so trained
numbers match TF only statistically. Fixture and full-model checks run
training mode with the moving-average updates disabled (tflow does not run
its update ops there).
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

The model components (wide_resnet::WideBasicBlock/WideResNet, preact
normalization, the sine fixtures, image batching, cross-entropy and L2) are
the C++ ones through SWIG (pydatamunge.nn_components); the checks and the
experiment workflow below mirror examples/cpp/source/nn/.../wide_resnet.cpp line by line.
"""
import math
import os
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import architecture, check, example_runtime, image_training, paths, preact, wide_resnet


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


def norm_config(training, update_stats, momentum, epsilon):
    """preact::NormConfig{training, update_stats, momentum, epsilon}."""
    norm = preact.NormConfig()
    norm.training, norm.update_stats, norm.momentum, norm.epsilon = training, update_stats, momentum, epsilon
    return norm


def dropout_checks():
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    gen = nn.Generator(17)
    # sin(.17f * i) in float, i = 0..2047 (architecture::SinePixels computes exactly that).
    x = architecture.SinePixels([8, 8, 8, 4], 1, .17)
    x.requires_grad_(True)
    values = list(x.values())
    train_norm, infer_norm = norm_config(True, False, .9, 1e-5), norm_config(False, False, .9, 1e-5)

    def block(norm, dropout):
        return wide_resnet.WideBasicBlock(p, state, "block", x, 4, 1, norm, dropout, gen)

    train = block(train_norm, .3)
    infer = list(block(infer_norm, .3).output.values())
    no_dropout = list(block(infer_norm, 0).output.values())
    mask_gradient = list(nn.grad(nn.sum(train.after_dropout), [train.before_dropout])[0].values())
    before, after = list(train.before_dropout.values()), list(train.after_dropout.values())
    shortcut = list(train.shortcut.values())
    positive = dropped = 0
    for i in range(len(values)):
        check.Near(infer[i], no_dropout[i], 0, "dropout rate does not affect inference")
        check.Near(shortcut[i], values[i], 0, "dropout leaves raw shortcut untouched")
        if before[i] <= 1e-6:
            continue
        positive += 1
        removed = after[i] == 0
        dropped += removed
        check.Near(after[i], 0 if removed else before[i] / .7, 2e-6, "inverted dropout values")
        check.Near(mask_gradient[i], 0 if removed else 1 / .7, 2e-6, "inverted dropout derivatives")
    check.Require(positive > 500 and abs(dropped / positive - .3) < .06, "wide dropout empirical mask rate")
    again_train = list(block(train_norm, .3).after_dropout.values())
    again_infer = list(block(infer_norm, .3).output.values())
    difference = 0
    for i in range(len(values)):
        check.Near(again_infer[i], infer[i], 0, "repeatable wide inference")
        difference += abs(again_train[i] - after[i])
    check.Require(difference > 1, "training dropout draws fresh masks")
    p.set("block/conv2/W", nn.zeros([3, 3, 4, 4]))  # zero residual
    train = block(train_norm, .3)
    output = list(train.output.values())
    dx = list(nn.grad(nn.sum(train.output), [x])[0].values())
    for i in range(len(values)):
        check.Near(output[i], values[i], 0, "signed identity survives residual dropout")
        check.Near(dx[i], 1, 2e-6, "unit identity derivative survives residual dropout")
    print("  dropout values/derivatives, changing masks, deterministic inference and signed identity passed; "
          f"removed fraction={g(dropped / positive)}", flush=True)


def multiply_accumulates(depth, width):
    total = 32 * 32 * 9 * 3 * 16
    in_channels, spatial = 16, 32
    for stage in range(3):
        out = (16 << stage) * width
        if stage:
            spatial //= 2
        for _block in range((depth - 4) // 6):
            total += spatial * spatial * (9 * in_channels * out + 9 * out * out + (in_channels * out if in_channels != out else 0))
            in_channels = out
    return total + in_channels * 10


def full_checks(depth, width, parameters, state_parameters):
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    gen = nn.Generator(17)
    cfg = wide_resnet.WideResNetConfig()
    cfg.depth = depth
    cfg.widen_factor = width
    cfg.dropout = 0
    cfg.update_stats = False
    # .1f * sin(i * .13f) in float (architecture::SinePixels computes exactly that).
    net = wide_resnet.WideResNet(p, state, architecture.SinePixels([2, 32, 32, 3], .1, .13), cfg, True, gen)
    check.Require(p.count() == parameters and state.count() == state_parameters, "canonical wide parameter/state counts")
    check.Require(len(net.blocks) == (depth - 4) // 2, "wide residual block count")
    for stage in range(4):
        spatial = 32 if stage < 2 else 32 // (1 << (stage - 1))
        channels = 16 if stage == 0 else (16 << (stage - 1)) * width
        shape = list((net.stem if stage == 0 else net.stages(stage - 1)).shape())
        check.Require(shape == [2, spatial, spatial, channels], "canonical wide stage dimensions")
    check.Require(list(net.logits.shape()) == [2, 10], "wide classifier dimensions")
    gradients = nn.grad(nn.mean(nn.square(net.logits)), list(p.parameters()))
    names = list(p.names())
    minimum, maximum = math.inf, 0
    for i in range(len(gradients)):
        norm = nn.sqrt(nn.sum(nn.square(gradients[i]))).item()
        check.Require(math.isfinite(norm) and norm > 0, "finite nonzero wide derivative: " + names[i])
        minimum, maximum = min(minimum, norm), max(maximum, norm)
    print(f"  WRN-{depth}-{width}: parameters={parameters}, state={state_parameters}, "
          f"MACs/image={multiply_accumulates(depth, width)}; gradient norms [{g(minimum)},{g(maximum)}]", flush=True)


def train_wide_resnet(width, dropout, arguments):
    """One WRN-16 setting (the body of the C++ settings loop): options, CIFAR-10 split,
    Adam on cross-entropy plus L2, validation-NLL checkpoint selection, one test pass."""
    cfg = image_training.ImageTrainingConfig()
    cfg.name = f"wide_resnet16_{width}{'_dropout' if dropout > 0 else ''}_cifar10"
    cfg.maximum_fit_rows = 12000
    cfg.epochs = 12
    cfg.minimum_test_accuracy = .45
    gen = nn.Generator(17)  # one dropout stream per model, persistent across steps

    def model(params, state, image, training):
        options = wide_resnet.WideResNetConfig()
        options.depth = 16
        options.widen_factor = width
        options.dropout = dropout
        return wide_resnet.WideResNet(params, state, image, options, training, gen).logits

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


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    for argument in arguments:  # threads= also applies to the checks
        if argument.startswith("threads="):
            nn.set_num_threads(int(argument[8:]))
    dropout_checks()
    full_checks(16, 1, 175066, 928)
    full_checks(16, 2, 691674, 1824)
    full_checks(28, 10, 36479194, 17952)
    for width, dropout in ((1, 0), (2, 0), (2, .3)):
        train_wide_resnet(width, dropout, arguments)
    print("All Wide ResNet checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
