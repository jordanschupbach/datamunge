# Native Python NN example. Stochastic depth: DropPath mask/schedule/shortcut checks, then constant-depth versus stochastic-depth 20-layer CIFAR-10 training; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.3_residual_and_dense_families/stochastic_depth.py
"""Stochastic depth: survival schedules, masking conventions and learning.

Port of tflow's stochastic_depth. Verify original train=bF/test=pF and
modern inverted train=bF/p/test=F conventions, whole-batch and per-example
masks, exact input/weight derivatives, fresh masks, empirical survival rates
and p=0/1 boundaries. The schedule p_l=1-l/L*(1-p_last) has expected active
depth (3L-1)/4 when p_last=.5. Check the 110-layer CIFAR architecture with
all gates open, including its average-pool/zero-pad shortcuts and all
parameter gradients.
https://arxiv.org/abs/1603.09382
https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py

Compare two 20-layer, 269,722-parameter CIFAR models: constant depth and
inverted whole-batch DropPath with final survival .5. Use the same seed,
12k fit / 5k validation split, twelve epochs, crop/flip, Adam(.001), kernel
L2(.0001), and independent best-validation checkpoints before test scoring.
Every residual is computed and every BN statistic updated; this does not
implement the original conditional execution or claim speedup.
Bias-free convolutions and the short optimizer schedule are adaptations.
One paired run illustrates the procedure, not a general accuracy claim.
Deviations: masks are drawn from one nn::Generator per model (tflow seeds
each block's RandomUniform with 71+index); weights come from nn's generator
(seed=), so trained numbers match TF only statistically. The 110-layer
gradient check disables the moving-average updates.
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

The components (stochastic_depth::DropPath, SurvivalProbabilities,
CifarIdentityShortcut and StochasticCifarResNet, image batching, cross-entropy
and L2) are the C++ ones through SWIG (pydatamunge.nn_components); the checks
and the experiment workflow below mirror examples/cpp/source/nn/.../stochastic_depth.cpp
line by line.
"""
import math
import struct
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import check, example_runtime, image_training, paths, stochastic_depth as sd


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def f32(value):
    """The C++ `float` literal: round a double to single precision."""
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


def mask_checks(per_example, scaling, survival):
    """Whole-batch or per-row DropPath masks: Bernoulli gates, exact input/weight
    derivatives, deterministic inference scaling and the empirical survival rate."""
    p = nn.ParameterStore(17)
    gen = nn.Generator(73)
    rows = 8192 if per_example else 4
    repeats = 1 if per_example or survival == 0 or survival == 1 else 512
    # .2f + .1f * sin(.17f * i), evaluated in single precision as the C++ does.
    values = [f32(f32(.2) + f32(f32(.1) * f32(math.sin(f32(f32(.17) * i))))) for i in range(rows * 8)]
    x = nn.from_vector(values, [rows, 2, 2, 2])
    x.requires_grad_(True)
    p.set("weight", nn.scalar(1.5))
    weight = p.at("weight")
    inverted = scaling == sd.DropPathScaling_kInverted
    alive, samples = 0, 0
    error = 0.0
    for _repeat in range(repeats):
        branch = nn.mul(x, weight)
        train = sd.DropPath(branch, survival, True, per_example, scaling, gen)
        infer = sd.DropPath(branch, survival, False, per_example, scaling, gen)
        output = nn.add(x, train.value)
        gradients = nn.grad(nn.sum(output), [x, weight])
        out, gate = output.values(), train.gate.values()
        dx, inference = gradients[0].values(), infer.value.values()
        scalar_gate = len(gate) == 1
        weight_derivative = 0.0
        for row in range(rows):
            gate_value = gate[0 if scalar_gate else row]
            check.Require(gate_value == 0 or gate_value == 1, "Bernoulli gate")
            if per_example or row == 0:
                alive += int(gate_value)
                samples += 1
            multiplier = gate_value / survival if inverted and survival > 0 else gate_value
            for item in range(8):
                index = row * 8 + item
                value = values[index]
                check.Near(out[index], value * (1 + 1.5 * multiplier), 3e-6,
                           "identity plus uniformly masked row/whole batch")
                check.Near(dx[index], 1 + 1.5 * multiplier, 3e-6, "identity plus masked branch input derivative")
                check.Near(inference[index], 1.5 * value * (1 if inverted else survival), 3e-6,
                           "deterministic inference scaling")
                weight_derivative += multiplier * value
        error = max(error, abs(gradients[1].item() - weight_derivative) / max(1., abs(weight_derivative)))
    check.Near(error, 0, 3e-5, "masked parameter derivative including zero branch gradient")
    empirical = alive / samples
    check.Near(empirical, survival, 4 * math.sqrt(survival * (1 - survival) / samples) + .002,
               "empirical survival and expected branch multiplier")
    if 0 < survival < 1:
        check.Require(0 < alive < samples, "both active and inactive masks observed")
    print(f"  {'row' if per_example else 'batch'}/{'inverted' if inverted else 'original'}, p={g(survival)}: "
          f"empirical={g(empirical)}, derivative error={g(error)}", flush=True)


def schedule_and_shortcut_checks():
    """The linear survival schedule's expected active depth and the option-A
    (average-pool, zero-padded) CIFAR shortcut with its derivative."""
    survival = sd.SurvivalProbabilities(54, .5)
    check.Near(survival[0], 1 - .5 / 54, 1e-7, "first scheduled survival")
    check.Near(survival[-1], .5, 0, "last scheduled survival")
    expected = sum(survival, 0.)
    check.Near(expected, 40.25, 2e-6, "expected active blocks in 110-layer model")
    values = [float(i + 1) for i in range(32)]
    x = nn.from_vector(values, [1, 4, 4, 2])
    x.requires_grad_(True)
    shortcut = sd.CifarIdentityShortcut(x, 4, 2)
    gradient = nn.grad(nn.sum(shortcut), [x])[0].values()
    check.Require(tuple(shortcut.shape()) == (1, 2, 2, 4), "CIFAR shortcut spatial/channel dimensions")
    out = shortcut.values()
    for r in range(2):
        for c in range(2):
            for channel in range(4):
                value = 0.0
                if channel < 2:
                    for dr in range(2):
                        for dc in range(2):
                            value += .25 * values[((2 * r + dr) * 4 + 2 * c + dc) * 2 + channel]
                check.Near(out[(r * 2 + c) * 4 + channel], value, 0, "host average-pool and channel padding")
    for i in range(32):
        check.Near(gradient[i], .25, 0, "average-pool shortcut derivative")
    print(f"  schedule: 54 possible blocks, {g(expected)} expected active; option-A shortcut checks passed", flush=True)


def full_checks():
    """The 110-layer CIFAR model with every gate open: counts, stage shapes and
    finite nonzero derivatives through every parameter (moving averages frozen)."""
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    gen = nn.Generator(71)
    cfg = sd.StochasticCifarConfig()
    cfg.last_survival = 1
    cfg.update_stats = False
    # .1f * sin(i * .13f), evaluated in single precision as the C++ does.
    pixels = [f32(f32(.1) * f32(math.sin(f32(i * f32(.13))))) for i in range(2 * 32 * 32 * 3)]
    net = sd.StochasticCifarResNet(p, state, nn.from_vector(pixels, [2, 32, 32, 3]), cfg, True, gen)
    check.Require(p.count() == 1727962 and state.count() == 8096 and len(net.gates) == 54,
                  "110-layer CIFAR parameter/state/block counts")
    for stage in range(3):
        check.Require(tuple(net.stages[stage].shape()) == (2, 32 >> stage, 32 >> stage, 16 << stage),
                      "CIFAR residual stage dimensions")
    check.Require(tuple(net.logits.shape()) == (2, 10), "CIFAR residual classifier dimensions")
    gradients = nn.grad(nn.mean(nn.square(net.logits)), p.parameters())
    minimum, maximum = math.inf, 0.0
    names = p.names()
    for i, gradient in enumerate(gradients):
        norm = nn.sqrt(nn.sum(nn.square(gradient))).item()
        check.Require(math.isfinite(norm) and norm > 0, "finite nonzero residual derivative: " + names[i])
        minimum = min(minimum, norm)
        maximum = max(maximum, norm)
    print(f"  110-layer CIFAR model with open gates: parameters={p.count()}, gradient norms [{g(minimum)},{g(maximum)}]",
          flush=True)


def train_depth20(survival, arguments):
    """One 20-layer comparison model (constant depth when survival is 1, inverted
    whole-batch DropPath otherwise): CIFAR-10 options, fixed development split,
    Adam on cross-entropy plus L2, validation-NLL checkpoint selection, one test pass."""
    cfg = image_training.ImageTrainingConfig()
    cfg.name = "constant_depth20_cifar10" if survival == 1 else "stochastic_depth20_cifar10"
    cfg.maximum_fit_rows = 12000
    cfg.epochs = 12
    cfg.minimum_test_accuracy = .45
    gen = nn.Generator(71)  # one mask stream per model, persistent across steps

    def model(p, state, image, training):
        model_cfg = sd.StochasticCifarConfig()
        model_cfg.depth = 20
        model_cfg.last_survival = survival
        logits = sd.StochasticCifarResNet(p, state, image, model_cfg, training, gen).logits
        check.Require(p.count() == 269722 and state.count() == 1376, "20-layer comparison model parameter/state counts")
        return logits

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
    print(f"  best checkpoint and history: {directory}\nAll {cfg.name} checks passed.", flush=True)


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    for argument in arguments:  # threads= also applies to the checks
        if argument.startswith("threads="):
            nn.set_num_threads(int(argument[8:]))
    for scaling in (sd.DropPathScaling_kOriginal, sd.DropPathScaling_kInverted):
        mask_checks(False, scaling, f32(.6))
        mask_checks(True, scaling, f32(.6))
        mask_checks(False, scaling, 0)
        mask_checks(True, scaling, 1)
    schedule_and_shortcut_checks()
    full_checks()
    for survival in (1., .5):
        train_depth20(survival, arguments)
    print("All stochastic-depth checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
