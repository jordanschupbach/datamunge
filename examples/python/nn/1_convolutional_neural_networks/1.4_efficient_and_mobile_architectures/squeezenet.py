# Native Python NN example. SqueezeNet fire-module/geometry checks and three CIFAR-10 trainings (v1.0, v1.0 bypass, v1.1); the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/squeezenet.py
"""SqueezeNet: AlexNet-level design with 50x fewer parameters via fire modules.

Check that a fire module's concatenated 1x1 and 3x3 expand layers equal a
single 3x3 convolution whose first filters are the 1x1 kernels placed at
the centre tap, including input derivatives, and compare its parameters
with a plain 3x3 convolution of the same width (the paper's strategies 1
and 2). Check the released spatial geometry, which depends on Caffe's
ceil-rounded pooling (54/27/13 for v1.0 and 55/27/13 for v1.1 at 224),
the 1,248,424 / 1,235,496 learned parameters of v1.0 / v1.1 (torchvision's
counts; simple bypasses add none) and finite nonzero derivatives.
https://arxiv.org/abs/1602.07360
https://github.com/forresti/SqueezeNet

Train v1.0, v1.0 with simple bypasses and v1.1 at full width on CIFAR-10
with a 3x3 stride-one first convolution, keeping all fire modules, pools,
dropout and the convolutional classifier. Use 12k fit / 5k validation,
twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and the
best-validation checkpoint before evaluating all 10k test images. These are
CIFAR adaptations, not the paper's ImageNet benchmark.
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1.

Port: datamunge eager autograd; seeded native initialization gives statistical
training parity. Dataset cache and training options use image_training.hpp.
Port default lr=.0001: with native initialization the original .001 kills
the final ReLU classifier (chance predictions and zero data gradients).
The original learning rate remains available with lr=.001.

The model components (squeezenet::Fire/SqueezeNet, the mobile_checks sine
fixtures, architecture::GradientNormRange, image batching, cross-entropy and
L2) are the C++ ones through SWIG (pydatamunge.nn_components); the checks and
the training workflow below mirror examples/cpp/source/nn/.../squeezenet.cpp line by line.
"""
import math
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import architecture, check, example_runtime, image_training, mobile_checks, paths, squeezenet

Pattern = mobile_checks.Pattern


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


def fire_checks():
    p = nn.ParameterStore(17)
    x = Pattern([2, 6, 6, 5], 1, .1).requires_grad_()
    squeezenet.Fire(p, "fire", x, 3, 4, 6)
    for name in p.names():
        if mobile_checks.EndsWith(name, "/b"):
            p.set(name, Pattern(p.at(name).shape(), .2, len(name)))
    fire = squeezenet.Fire(p, "fire", x, 3, 4, 6)
    centred = nn.pad(p.at("fire/expand1x1/W"), [(1, 1), (1, 1), (0, 0), (0, 0)])
    kernel = nn.concat([centred, p.at("fire/expand3x3/W")], 3)
    bias = nn.concat([p.at("fire/expand1x1/b"), p.at("fire/expand3x3/b")], 0)
    o = nn.Conv2DOptions()
    o.padding = nn.Padding.Same
    merged = nn.relu(nn.add(nn.conv2d(fire.squeeze, kernel, o), bias))
    weight = Pattern([2, 6, 6, 10], 1, .3)
    ga = nn.grad(nn.sum(nn.mul(fire.output, weight)), [x])
    # The first backward releases its graph, so recompute the merged branch.
    fire = squeezenet.Fire(p, "fire", x, 3, 4, 6)
    merged = nn.relu(nn.add(nn.conv2d(fire.squeeze, kernel, o), bias))
    gb = nn.grad(nn.sum(nn.mul(merged, weight)), [x])
    check.Near(nn.max(nn.abs(nn.sub(fire.output, merged))).item(), 0, 1e-5,
               "fire module equals centre-embedded convolution")
    check.Near(nn.max(nn.abs(nn.sub(ga[0], gb[0]))).item(), 0, 1e-5, "fire input derivatives")
    fire2 = 96 * 16 + 16 + 16 * 64 + 64 + 9 * 16 * 64 + 64
    plain = 9 * 96 * 128 + 128
    print(f"fire2 parameters {fire2} versus plain {plain}", flush=True)


def full_checks():
    cases = [
        (10, False, 1248424, [54, 54, 54, 27, 27, 27, 27, 13]),
        (10, True, 1248424, [54, 54, 54, 27, 27, 27, 27, 13]),
        (11, False, 1235496, [55, 55, 27, 27, 13, 13, 13, 13]),
    ]
    for version, bypass, params, sizes in cases:
        p = nn.ParameterStore(17)
        gen = nn.Generator(101)
        cfg = squeezenet.SqueezeNetConfig()
        cfg.version = version
        cfg.simple_bypass = bypass
        net = squeezenet.SqueezeNet(p, Pattern([2, 224, 224, 3], 1, 0), cfg, True, gen)
        check.Require(p.count() == params and list(net.sizes) == sizes, "SqueezeNet count and ceil geometry")
        stem = 109 if version == 10 else 111
        fires = list(net.fires)
        check.Require(net.stem.size(1) == stem and fires[-1].size(1) == 13 and fires[-1].size(3) == 512
                      and tuple(net.logits.shape()) == (2, 1000), "SqueezeNet stem and classifier dimensions")
        norms = architecture.GradientNormRange(p, nn.mean(nn.square(net.logits)), "SqueezeNet")
        print(f"SqueezeNet {version} bypass={int(bypass)} parameters={p.count()} smallest gradient={g(norms[0])}",
              flush=True)


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    nn.set_num_threads(2)
    fire_checks()
    full_checks()
    runs = [
        ("squeezenet10_cifar10", 10, False, 729034),
        ("squeezenet10_bypass_cifar10", 10, True, 729034),
        ("squeezenet11_cifar10", 11, False, 727626),
    ]
    for run_name, run_version, run_bypass, run_params in runs:
        cfg = image_training.ImageTrainingConfig()
        cfg.name = run_name
        cfg.maximum_fit_rows = 12000
        cfg.learning_rate = .0001
        cfg.epochs = 12
        cfg.minimum_test_accuracy = .4
        generator = nn.Generator(101)

        def model(p, state, image, training):
            """image_training::LogitsBuilder: the CIFAR SqueezeNet variant of this run (one dropout generator per run)."""
            model_cfg = squeezenet.SqueezeNetConfig()
            model_cfg.version = run_version
            model_cfg.simple_bypass = run_bypass
            model_cfg.classes = 10
            model_cfg.image_size = 32
            model_cfg.cifar_stem = True
            net = squeezenet.SqueezeNet(p, image, model_cfg, training, generator)
            check.Require(p.count() == run_params and state.count() == 0, "CIFAR SqueezeNet parameter count")
            return net.logits

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
                      and cfg.validation_rows < development_rows and cfg.epochs > 0 and cfg.batch > 0
                      and cfg.classes > 1, "valid image training dimensions/config")
        for d in range(1, 4):
            check.Require(development.images.shape()[d] == test.images.shape()[d],
                          "training/test image dimensions match")
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
                # Augmented batch; training-mode forward; cross-entropy plus 0.5 * l2 * sum(W^2); one
                # Adam step.
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
        check.TrainingQuality(test_accuracy >= cfg.minimum_test_accuracy,
                              "image classifier demonstration accuracy threshold")
        train_total = sum(epoch_seconds)
        print(f"  training time {train_total:.1f} s ({train_total / cfg.epochs:.2f} s/epoch)")
        print(f"  best checkpoint and history: {directory}\nAll {cfg.name} checks passed.", flush=True)
    print("All SqueezeNet checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
