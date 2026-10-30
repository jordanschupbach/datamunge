# Native Python NN example. ShuffleNet shuffle/unit/cost/gradient checks and v1 (3 groups) / v2 CIFAR-10 training; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/shufflenet.py
"""ShuffleNet v1 and v2: grouped pointwise convolutions with channel shuffle, and channel split.

Check the released channel shuffle and the paper's transpose exactly, and
that they are inverse permutations. Reproduce the paper's Figure 1
argument with derivative supports: through two stacked group
convolutions, an output channel depends on one input group, and on every
group once a shuffle sits between them. Check that v1's strided shortcut
counts zero padding in its average and that a v2 unit passes its even
channels through unchanged. Count parameters and multiply-adds for all
released sizes: v1 (3 groups) 38/137/292/524M, v1 (8 groups)
40/138/290/522M and v2 41/146/299/591M at 0.5x/1x/1.5x/2x, matching the
release's table. Check every derivative with shifted BN betas: all are
clearly nonzero except the depthwise BN shifts, which only reach 1x1
convolutions followed by training-mode BN.
https://arxiv.org/abs/1707.01083
https://arxiv.org/abs/1807.11164
https://github.com/megvii-model/ShuffleNet-Series

Train v1 1x (3 groups) and v2 1x on CIFAR-10 without the stem's stride and
max-pool (all sixteen units per network). Use 12k fit / 5k validation,
twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and the
best-validation checkpoint before evaluating all 10k test images.
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1.

The model components (shufflenet::ShuffleNet, the shuffles and units, the
grouped convolutions, mobile_checks fixtures, image batching, cross-entropy
and L2) are the C++ ones through SWIG (pydatamunge.nn_components); the checks
and the experiment workflow below mirror examples/cpp/source/nn/.../shufflenet.cpp
line by line.
"""
import math
import struct
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import check, example_runtime, grouped, image_training, mobile_checks, paths, shufflenet


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def f32(value):
    """Rounds a double to float32, as the C++ float arithmetic does."""
    return struct.unpack("f", struct.pack("f", value))[0]


def copy_config(cfg):
    """ImageTrainingConfig is passed by value to each version in the C++; copy every field."""
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


def shuffle_checks():
    C, G = 12, 3
    n = C // G
    values = [float(i % C) for i in range(48)]
    x = nn.from_vector(values, [1, 2, 2, C]).requires_grad_()
    release = shufflenet.ChannelShuffle(x, C, G).values()
    paper = shufflenet.PaperChannelShuffle(x, C, G)
    pv, composed = paper.values(), shufflenet.ChannelShuffle(paper, C, G).values()
    for c in range(C):
        check.Require(release[(c % G) * n + c // G] == c, "release shuffle order")
        check.Require(pv[(c % n) * G + c // n] == c, "paper shuffle order")
        check.Require(composed[c] == c, "inverse shuffles")
    p = nn.ParameterStore(17)
    o = nn.ConvLayerOptions()
    o.bias = False
    o.init = nn.init.he_uniform()

    def build(shuffled):
        first = grouped.GroupedConv2D(p, "first", x, C, 1, G, o)
        if shuffled:
            first = shufflenet.ChannelShuffle(first, C, G)
        y = grouped.GroupedConv2D(p, "second", first, C, 1, G, o)
        return nn.grad(nn.sum(nn.slice(y, [0, 0, 0, 0], [1, 2, 2, 1])), [x])[0].values()

    plain, shuffled = build(False), build(True)
    support = shuffled_support = 0
    reached = set()
    for c in range(C):
        a = b = 0.
        for pixel in range(4):
            a += abs(plain[pixel * C + c])
            b += abs(shuffled[pixel * C + c])
        support += a > 0
        shuffled_support += b > 0
        if b > 0:
            reached.add(c // n)
    check.Require(support == n and len(reached) == G, "shuffle connects every group")
    print(f"derivative supports plain={support} shuffled={shuffled_support}", flush=True)


def unit_checks():
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    c = shufflenet.ShuffleContextHandle(p, state, False)
    x = mobile_checks.Pattern([1, 6, 6, 8], 1, .3)
    v = x.values()
    pooled = shufflenet.AveragePoolWithPadding(x).values()
    v2 = shufflenet.ShuffleV2Unit(c.get(), "v2", x, 8, 1).values()
    corner = passed = 0.
    for ch in range(8):
        # (v[ch] + v[8 + ch] + v[48 + ch] + v[56 + ch]) is summed in float.
        total = f32(f32(f32(v[ch] + v[8 + ch]) + v[48 + ch]) + v[56 + ch])
        corner = max(corner, abs(pooled[ch] - total / 9.))
    for pixel in range(36):
        for ch in range(4):
            passed = max(passed, abs(f32(v2[pixel * 8 + ch] - v[pixel * 8 + 2 * ch])))
    check.Near(corner, 0, 1e-6, "average shortcut includes zero padding")
    check.Near(passed, 0, 0, "v2 passes even channels unchanged")


def cost_checks():
    cases = [
        (1, 3, "0.5x", 717324, 12240, 37589856),
        (1, 3, "1.0x", 1864728, 24480, 137460672),
        (1, 3, "1.5x", 3441168, 36720, 292443552),
        (1, 3, "2.0x", 5449776, 48960, 524045184),
        (1, 8, "0.5x", 1010448, 19584, 40469760),
        (1, 8, "1.0x", 2433768, 39168, 138365952),
        (1, 8, "1.5x", 4268952, 58752, 290188800),
        (1, 8, "2.0x", 6521232, 78336, 521942016),
        (2, 0, "0.5x", 1365792, 7952, 40476448),
        (2, 0, "1.0x", 2278760, 16180, 145814296),
        (2, 0, "1.5x", 3506720, 23440, 298970656),
        (2, 0, "2.0x", 7402600, 33716, 590783000),
    ]
    for version, groups, size, expected_params, expected_state, expected_macs in cases:
        with nn.no_grad():
            p, state = nn.ParameterStore(17), nn.ParameterStore(17)
            cfg = shufflenet.ShuffleNetConfig()
            cfg.version = version
            cfg.groups = groups if groups else 1
            cfg.size = size
            net = shufflenet.ShuffleNet(p, state, nn.zeros([1, 224, 224, 3]), cfg)
            macs = net.classifier_macs
            for i in range(net.layer_count()):
                shape = net.layer_shape(i)
                macs += shape[1] * shape[2] * net.layer_cost(i)
            check.Require(net.features.size(1) == 7 and p.count() == expected_params and state.count() == expected_state
                          and macs == expected_macs, "ShuffleNet parameters, MACs and 7x7 features")
            print(f"ShuffleNet v{version} {size} groups={groups} params={p.count()} MACs={macs}", flush=True)


def gradient_checks(version):
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    cfg = shufflenet.ShuffleNetConfig()
    cfg.update_stats = False
    cfg.version = version
    x = mobile_checks.Pattern([2, 224, 224, 3], 1, 0)
    with nn.no_grad():
        shufflenet.ShuffleNet(p, state, x, cfg, True)
        mobile_checks.ShiftBetas(p, False)
    net = shufflenet.ShuffleNet(p, state, x, cfg, True)
    check.Require(list(net.logits.shape()) == [2, 1000], "ShuffleNet output dimensions")
    norms = list(mobile_checks.GradientNorms(p, nn.mean(nn.square(net.logits))))
    maximum, minimum, removed, count = max(norms), math.inf, 0., 0
    names = p.names()
    for i in range(len(norms)):
        check.Require(math.isfinite(norms[i]), "finite ShuffleNet derivative")
        if mobile_checks.EndsWith(names[i], "/depthwise/bn/beta"):
            removed = max(removed, norms[i])
            count += 1
        else:
            minimum = min(minimum, norms[i])
    check.Require(count == (16 if version == 1 else 19) and removed < 1e-6 * maximum, "vanishing projection BN shifts")
    check.Require(minimum > 1e-5 * maximum, "nonzero ShuffleNet derivatives with shifted BN")
    print(f"gradient range {g(minimum)} to {g(maximum)} removed shifts {g(removed)}", flush=True)


def train_compact_shufflenet(version, development, test, cfg, explicit_epochs):
    """Fits one CIFAR ShuffleNet version (v1 1x with 3 groups, or v2 1x): fixed
    development split, Adam on cross-entropy plus L2, validation-NLL checkpoint
    selection, one test pass."""
    cfg = copy_config(cfg)
    cfg.name = "shufflenet_v1_g3_cifar10" if version == 1 else "shufflenet_v2_cifar10"

    def model(params, state, image, training):
        options = shufflenet.ShuffleNetConfig()
        options.version = version
        options.classes = 10
        options.small_stem = True
        net = shufflenet.ShuffleNet(params, state, image, options, training)
        check.Require(params.count() == (914328 if version == 1 else 1265000), "CIFAR ShuffleNet parameter count")
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
    shuffle_checks()
    unit_checks()
    cost_checks()
    gradient_checks(1)
    gradient_checks(2)
    cfg = image_training.ImageTrainingConfig()
    cfg.maximum_fit_rows = 12000
    cfg.epochs = 12
    cfg.minimum_test_accuracy = .4
    directory, explicit_epochs = parse_cifar_options(arguments, cfg)
    development, test = image_training.LoadCifar10(str(directory))
    for version in (1, 2):
        train_compact_shufflenet(version, development, test, cfg, explicit_epochs)
    print("All ShuffleNet checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
