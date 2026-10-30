# Native Python NN example. CondenseNet learned-group-convolution/cost/gradient checks and CondenseNet-86 CIFAR-10 training; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/condensenet.py
"""CondenseNet: learned group convolutions condensed during training.

Port of tflow's condensenet. Drive one learned group convolution (16 -> 16
channels, G = C = 4) through the release's schedule: at progress 1/6, 1/3
and 1/2 every group keeps 12, 8 and finally 4 input channels, and each time
the dropped channels are exactly those with the smallest summed |w| among
the group's still-active inputs. Convert the condensed layer to index
selection (with BN statistics copied), BN/ReLU, a grouped 1x1 convolution
and a channel shuffle, and check that it reproduces the masked layer
exactly. Count CondenseNet-86 on CIFAR-10 (1,451,594 parameters while
training; 516,202 after conversion with the classifier halved, the README's
0.52M; 62.4M multiply-adds against its 65.8M FLOPs) and the ImageNet
CondenseNet-74 with G = C = 4 / 8 (4.77M / 2.94M parameters, 517M / 262M
multiply-adds; the README's 4.8M / 2.9M and 529M / 274M). Check every
derivative of CondenseNet-86 in training mode (moving averages and
condensing state not updated).
https://arxiv.org/abs/1711.09224
https://github.com/ShichenLiu/CondenseNet

Train CondenseNet-86 (14-14-14 layers, growth 8-16-32, G = C = 4, group
lasso 1e-5) on CIFAR-10 with the schedule advancing on every training step
(tflow runs it in the graph; here each training-mode pass updates the masks
in the state store), so the network is fully condensed from the first step
of epoch seven. Use 12k fit / 5k validation, twelve epochs, crop/flip,
Adam(.001), kernel L2(.0001) and the best-validation checkpoint; the masked
network it evaluates is the converted one, but with the full classifier
(the release also drops half of the classifier's inputs at conversion).
Initialization uses nn's generator, so trained numbers match TF only
statistically. CIFAR-10 is read from the tflow dataset cache
(common/paths.hpp) or data=.
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

The model components (condensenet::LearnedGroupConv/Condense/CondenseNet, the
shufflenet channel shuffle, mobile_checks fixtures, image batching, cross-entropy
and L2) are the C++ ones through SWIG (pydatamunge.nn_components); the checks and
the training workflow below mirror examples/cpp/source/nn/.../condensenet.cpp line by line.
"""
import math
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import check, condensenet, example_runtime, image_training, mobile_checks, paths, shufflenet

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


def layer_checks():
    inputs, outputs, groups = 16, 16, 4
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    x = Pattern([2, 5, 5, inputs], 1, .1)
    # condensenet::CondenseContext{p, state, layers, false}: the handle owns the layer list.
    infer = condensenet.CondenseContextHandle(p, state, False)
    with nn.no_grad():
        condensenet.LearnedGroupConvHandle(infer, "lgc", x, outputs, groups, 4)
    weight = list(p.at("lgc/W").values())
    active = [[True] * inputs for _ in range(groups)]
    schedule = ""
    for value in (0., .1, .17, .2, .34, .4, .5, .9):
        # Expected drops from the masked weights before this step.
        expected = [list(a) for a in active]
        stage = min(3, int(math.floor(value * 6)))
        kept = inputs - stage * inputs // 4
        current = sum(active[0])
        if kept < current:
            for group in range(groups):
                scores = []
                for i in range(inputs):
                    if active[group][i]:
                        total = 0.0
                        for k in range(outputs // groups):
                            total += abs(weight[i * outputs + k * groups + group])
                        scores.append((total, i))
                scores.sort()
                for d in range(inputs // 4):
                    expected[group][scores[d][1]] = False
        # One training-mode pass at this progress (its BN update is harmless here).
        train = condensenet.CondenseContextHandle(p, state, True, .9, value)
        with nn.no_grad():
            condensenet.LearnedGroupConvHandle(train, "lgc", x, outputs, groups, 4)
        mask = list(state.at("lgc/mask").values())
        for group in range(groups):
            count = 0
            for i in range(inputs):
                on = mask[i * outputs + group] > .5
                count += on
                for k in range(1, outputs // groups):
                    check.Require((mask[i * outputs + k * groups + group] > .5) == on, "group-consistent mask")
                check.Require(on == expected[group][i], "dropped channels have the smallest summed weight magnitude")
            check.Require(count == kept, "channels kept per condensing stage")
        active = expected
        # std::to_string(value).substr(0, 4): two decimals.
        schedule += ("" if not schedule else ", ") + f"{value:.2f} -> {kept}"
    # Conversion: gather, BN with gathered statistics, ReLU, grouped 1x1,
    # shuffle, built from the condensed layer's values.
    with nn.no_grad():
        for name in state.names():
            if "/bn/" in name:
                values = Pattern(state.at(name).shape(), .3, len(name))
                if "moving_var" in name:
                    values = nn.add(values, .8)
                state.set(name, values)
        value = Pattern([2, 5, 5, inputs], 1, .3)
        reference = list(condensenet.LearnedGroupConvHandle(infer, "lgc", value, outputs, groups, 4).output.values())
        condensed = condensenet.Condense(p.at("lgc/W"), state.at("lgc/mask"), groups)
        condensed_index = list(condensed.index)
        width = len(condensed_index) // groups
        index = nn.from_vector_int64(condensed_index, [len(condensed_index)])

        def pick(name, store):
            return nn.gather(store.at(name), index, 0)

        gathered = nn.gather(value, index, 3)
        normalized = nn.relu(nn.add(
            nn.mul(nn.mul(nn.sub(gathered, pick("lgc/bn/moving_mean", state)),
                          nn.div(1.0, nn.sqrt(nn.add(pick("lgc/bn/moving_var", state), 1e-5)))),
                   pick("lgc/bn/gamma", p)),
            pick("lgc/bn/beta", p)))
        parts = nn.split(normalized, groups, 3)
        convolved = []
        for group in range(groups):
            convolved.append(nn.conv2d(
                parts[group],
                nn.slice(condensed.weight, [0, 0, 0, group * outputs // groups], [1, 1, width, outputs // groups]),
                nn.conv_options(1, nn.Padding.Valid)))
        converted = list(shufflenet.PaperChannelShuffle(nn.concat(convolved, 3), outputs, groups).values())
        error = 0.0
        for i in range(len(reference)):
            error = max(error, abs(reference[i] - converted[i]))
        distinct = len(set(condensed_index))
        check.Near(error, 0, 1e-5, "converted group convolution equals the masked layer")
        print(f"  learned group convolution 16->16, G=C=4, progress -> kept inputs per group: {schedule}; "
              f"drops follow summed |w|; converted: {groups} groups of {width} gathered channels ({distinct} distinct "
              f"inputs), BN/ReLU, grouped 1x1, shuffle; error {g(error)}", flush=True)


def converted(cfg, resolution):
    """Inference-form cost: per LGC in*out/C weights over G groups (with G = C the
    gathered width equals the input width), 3x3 group convolutions, final BN and
    a classifier on half the features. Returns (params, macs, features)."""
    stages, growth = list(cfg.stages), list(cfg.growth)
    params, macs = 0, 0
    channels, size = 2 * growth[0], resolution // 2 if cfg.imagenet else resolution
    params += 27 * channels
    macs += size * size * 27 * channels
    for stage in range(len(stages)):
        k, wide = growth[stage], cfg.bottleneck * growth[stage]
        for _layer in range(stages[stage]):
            gathered = channels * cfg.groups // cfg.condense
            params += 2 * gathered + (gathered // cfg.groups) * wide + 2 * wide + (wide // cfg.groups) * k * 9
            macs += size * size * ((gathered // cfg.groups) * wide + (wide // cfg.groups) * k * 9)
            channels += k
        if stage + 1 < len(stages):
            size //= 2
    params += 2 * channels + (channels // 2) * cfg.classes + cfg.classes
    macs += (channels // 2) * cfg.classes
    return params, macs, channels


def cost_checks():
    cifar = condensenet.CondenseNetConfig()
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    with nn.no_grad():
        net = condensenet.CondenseNet(p, state, nn.zeros([1, 32, 32, 3]), cifar)
        check.Require(p.count() == 1451594 and net.channels == 800 and state.count() == 1273850,
                      "CondenseNet-86 training parameters (and state)")
        params, macs, _features = converted(cifar, 32)
        check.Require(params == 516202 and macs == 62377888, "CondenseNet-86 converted cost")
        print(f"  CondenseNet-86 (CIFAR): {p.count()} parameters while training (masks and stages in {state.count()} "
              f"state values), {params} after conversion, {macs / 1e6:.1f}M mult-adds; {net.channels} final features",
              flush=True)
    for groups in (4, 8):
        imagenet = condensenet.CondenseNetConfig()
        imagenet.stages = [4, 6, 8, 10, 8]
        imagenet.growth = [8, 16, 32, 64, 128]
        imagenet.groups = imagenet.condense = groups
        imagenet.classes = 1000
        imagenet.imagenet = True
        params, macs, _features = converted(imagenet, 224)
        check.Require(params == (4773944 if groups == 4 else 2935416)
                      and macs == (516640576 if groups == 4 else 261545792), "ImageNet CondenseNet cost")
        print(f"  CondenseNet-74 (ImageNet, G=C={groups}): {params} parameters, {macs / 1e6:.1f}M mult-adds "
              "after conversion", flush=True)


def gradient_checks():
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    cfg = condensenet.CondenseNetConfig()
    cfg.update_stats = False
    net = condensenet.CondenseNet(p, state, Pattern([2, 32, 32, 3], 1, 0), cfg, True, 100)
    norms = list(mobile_checks.GradientNorms(p, nn.mean(nn.square(net.logits))))
    names = list(p.names())
    minimum, maximum, smallest = math.inf, 0.0, ""
    for i, norm in enumerate(norms):
        check.Require(math.isfinite(norm), "finite CondenseNet derivative")
        if norm < minimum:
            minimum, smallest = norm, names[i]
        maximum = max(maximum, norm)
    check.Require(minimum > 1e-5 * maximum, "clearly nonzero CondenseNet derivatives")
    print(f"  CondenseNet-86 training mode: {len(names)} parameter derivatives in [{g(minimum)},{g(maximum)}] "
          f"(smallest {smallest})", flush=True)


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    for argument in arguments:  # threads= also applies to the checks
        if argument.startswith("threads="):
            nn.set_num_threads(int(argument[8:]))
    layer_checks()
    cost_checks()
    gradient_checks()
    cfg = image_training.ImageTrainingConfig()
    cfg.name = "condensenet86_cifar10"
    cfg.maximum_fit_rows = 12000
    cfg.epochs = 12
    cfg.minimum_test_accuracy = .4
    directory, explicit_epochs = parse_cifar_options(arguments, cfg)
    development, test = image_training.LoadCifar10(str(directory))

    # The training workflow is written out below, mirroring the C++. Quick mode
    # bounds the data and batch; an explicit epoch budget is honored.
    if not explicit_epochs:
        cfg.epochs = example_runtime.Limit(cfg.epochs, 1)
    cfg.batch = example_runtime.Limit(cfg.batch, 8)
    if not example_runtime.Full():
        cfg.maximum_fit_rows = min(cfg.maximum_fit_rows, 32) if cfg.maximum_fit_rows else 32
    # The condensing schedule needs the run's total number of training steps.
    fit_rows = cfg.maximum_fit_rows or int(development.images.shape()[0]) - cfg.validation_rows
    total_steps = cfg.epochs * ((fit_rows + cfg.batch - 1) // cfg.batch)

    def model(p, state, image, training):
        """Inference form (evaluation and checkpoint selection): the converted, masked network with the
        full classifier."""
        if training:
            raise RuntimeError("CondenseNet training passes go through the training function")
        logits = condensenet.CondenseNet(p, state, image, condensenet.CondenseNetConfig()).logits
        check.Require(p.count() == 1451594, "CondenseNet-86 parameter count")
        return logits

    def training(p, state, image):
        """Training form: one pass advances the masks in `state` by 1/total_steps and returns the logits
        and the group-lasso penalties (image_training::TrainingNetwork{logits, {}, penalties})."""
        net = condensenet.CondenseNet(p, state, image, condensenet.CondenseNetConfig(), True, total_steps)
        return net.logits, list(net.penalties)

    print(f"  condensing schedule over {total_steps} training steps; fully condensed from step {total_steps // 2}",
          flush=True)
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
    with nn.no_grad():  # create every parameter (and state entry) before building the optimizer
        x = image_training.ImageBatch(development, [fit[0], fit[1]])[0]
        model(params, state, x, False)
        # Any training-only parameters are created by one training pass whose masks
        # and batch-norm statistics go to a scratch store, so it advances neither
        # the real condensing schedule nor the moving averages.
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
            # Augmented batch; training-mode pass (advances the condensing masks in `state`);
            # cross-entropy plus the group-lasso penalties plus 0.5 * l2 * sum(W^2); one Adam step.
            # (The training network has no custom loss and no auxiliary heads.)
            x, y = image_training.ImageBatch(development, rows, augmentation if cfg.augment else None)
            logits, penalties = training(params, state, x)
            objective = image_training.TrainingCrossEntropy(logits, y, cfg.classes, cfg.label_smoothing)
            for penalty in penalties:
                objective = nn.add(objective, penalty)
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
    print("All CondenseNet checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
