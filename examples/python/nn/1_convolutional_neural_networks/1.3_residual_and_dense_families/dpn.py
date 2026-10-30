# Native Python NN example. DPN unit/path/count checks and compact DPN-92 CIFAR-10 training; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.3_residual_and_dense_families/dpn.py
"""Dual path networks: a shared bottleneck feeding residual and dense paths.

Port of tflow's dpn. Follow the released DPN units: pre-activation 1x1 /
grouped 3x3 / 1x1 whose output is split into a residual update (added) and
inc new dense channels (concatenated); each stage starts with a projection
of the joined input. With the last 1x1 zeroed, ordinary units must pass the
residual path and old dense channels through unchanged with identity
derivatives, and strided projection units must reproduce an explicit
BN/ReLU/1x1 reference. Across every stage, the residual path must equal the
projection plus the sum of all residual updates (bitwise, accumulating
float32 in the same order), and the dense path the concatenation of the
projection's dense part and every unit's new channels, in order.
Full DPN-68/92/98/107/131 have 12,611,602 / 37,668,392 / 61,570,728 /
86,917,800 / 79,254,504 learned parameters (an independent count of the
reference layout); DPN-92 is also checked for stage dimensions and every
parameter derivative at 224 pixels in training mode (moving averages are
not updated). Parameter counts do not depend on the image size, so the
count-only models are built on 64-pixel images (tflow only builds graphs).
https://arxiv.org/abs/1707.01629
https://github.com/rwightman/pytorch-dpn-pretrained/blob/master/dpn.py

Train all 92 layers on CIFAR-10 with quarter widths and increments,
cardinality 8 (quarter-width bottlenecks of 24 channels are not divisible
by 32) and a 3x3 stride-one stem without max pooling. Use 12k fit / 5k
validation, twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and the
best-validation checkpoint before evaluating all 10k test images. This is a
compact CPU adaptation, not an ImageNet reproduction. The grouped 3x3s use
one grouped filter per layer (see common/dpn.hpp). CIFAR-10 is read from the
tflow dataset cache (common/paths.hpp) or data=.
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

The model components (dpn::DPNBlock/DPN/DPNJoin, the sine fixtures,
architecture::GradientNormRange, image batching, cross-entropy and L2) are the
C++ ones through SWIG (pydatamunge.nn_components); the checks and the training
workflow below mirror examples/cpp/source/nn/.../dpn.cpp line by line.
"""
import math
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import architecture, check, dpn, example_runtime, image_training, mobile_checks, paths

# dpn.cpp's Pattern: scale * sin(.37f * i + phase) in float (mobile_checks::Pattern computes exactly that).
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


def dpn_paths(residual, dense, residual_channels, dense_channels):
    """dpn::DPNPaths{residual, dense, residual_channels, dense_channels}."""
    x = dpn.DPNPaths()
    x.residual, x.dense = residual, dense
    x.residual_channels, x.dense_channels = residual_channels, dense_channels
    return x


def max_difference(a, b):
    check.Require(len(a) == len(b), "matching DPN comparison shapes")
    error = 0.0
    for i in range(len(a)):
        error = max(error, abs(a[i] - b[i]) / max(1.0, abs(b[i])))
    return error


def block_checks():
    """Zeroed final 1x1: identity paths for ordinary units, explicit projection for
    strided first units, and the corresponding input derivatives."""
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    r, d = Pattern([2, 5, 5, 4], .8, .3), Pattern([2, 5, 5, 3], .8, 1.3)

    def run(residual, dense):
        x = dpn_paths(residual, dense, 4, 3)
        return (dpn.DPNBlock(p, state, "normal", x, 4, 4, 2, 2, False),
                dpn.DPNBlock(p, state, "down", x, 6, 4, 2, 2, True, 2))

    with nn.no_grad():
        run(r, d)
        for name in ("normal/conv3/W", "down/conv3/W"):
            p.set(name, nn.zeros_like(p.at(name)))
        for name in state.names():
            variance = "moving_var" in name
            shape = list(state.at(name).shape())
            # PatternValues(shape[0], variance ? .2f : .3f, variance ? 0.f : 1.f), plus .8f for variances (float).
            values = Pattern([shape[0]], .2 if variance else .3, 0 if variance else 1)
            if variance:
                values = nn.add(values, .8)
            state.set(name, nn.reshape(values, shape))
    residual, dense = r.clone(), d.clone()
    residual.requires_grad_(True)
    dense.requires_grad_(True)
    normal, down = run(residual, dense)
    # Explicit projection: inference BN, ReLU, strided 1x1, split 6 + 4.
    with nn.no_grad():
        joined = nn.concat([r, d], 3)
        n = "down/projection/bn"
        bn = nn.add(nn.mul(nn.mul(nn.sub(joined, state.at(n + "/moving_mean")),
                                  nn.rsqrt(nn.add(state.at(n + "/moving_var"), 1e-3))),
                           p.at(n + "/gamma")), p.at(n + "/beta"))
        projection = nn.conv2d(nn.relu(bn), p.at("down/projection/W"), nn.conv_options(2, nn.Padding.Valid))
    # Weighted sums of each output; their input derivatives must be the weights.
    wr, wd = Pattern([2, 5, 5, 4], 1, .1), Pattern([2, 5, 5, 5], 1, .7)
    loss = nn.add(nn.sum(nn.mul(normal.output.residual, wr)), nn.sum(nn.mul(normal.output.dense, wd)))
    gradients = nn.grad(loss, [residual, dense])
    check.Require(tuple(normal.output.residual.shape()) == (2, 5, 5, 4)
                  and tuple(normal.output.dense.shape()) == (2, 5, 5, 5), "ordinary DPN unit widths")
    check.Require(tuple(down.output.residual.shape()) == (2, 3, 3, 6)
                  and tuple(down.output.dense.shape()) == (2, 3, 3, 6), "strided DPN projection widths")
    nr, nd = list(normal.output.residual.values()), list(normal.output.dense.values())
    dr, dd = list(down.output.residual.values()), list(down.output.dense.values())
    pr, gr, gd = list(projection.values()), list(gradients[0].values()), list(gradients[1].values())
    rv, dv, wrv, wdv = list(r.values()), list(d.values()), list(wr.values()), list(wd.values())
    identity, gradient, projected, zeros = max_difference(nr, rv), max_difference(gr, wrv), 0.0, 0.0
    for i in range(len(nd)):
        channel, pixel = i % 5, i // 5
        if channel < 3:
            identity = max(identity, abs(nd[i] - dv[pixel * 3 + channel]))
            gradient = max(gradient, abs(gd[pixel * 3 + channel] - wdv[i]))
        else:
            zeros = max(zeros, abs(nd[i]))
    for i in range(len(dr)):
        channel, pixel = i % 6, i // 6
        projected = max(projected, abs(dr[i] - pr[pixel * 10 + channel]))
        if channel < 4:
            projected = max(projected, abs(dd[i] - pr[pixel * 10 + 6 + channel]))
        else:
            zeros = max(zeros, abs(dd[i]))
    check.Near(identity, 0, 0, "ordinary unit passes residual and old dense channels unchanged")
    check.Near(zeros, 0, 0, "zeroed final 1x1 appends zero dense channels")
    check.Near(gradient, 0, 0, "identity derivatives through both paths")
    check.Near(projected, 0, 1e-6, "strided projection splits into residual and dense paths")
    print(f"  zeroed updates: identity paths and derivatives exact; strided projection error={g(projected)}",
          flush=True)


def stage_checks():
    """Whole-network path unrolling in a compact model with nontrivial weights."""
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    cfg = dpn.DPNConfig()
    cfg.width_divisor = 4
    cfg.groups = 8
    cfg.classes = 10
    cfg.small_stem = True
    with nn.no_grad():
        net = dpn.DPN(p, state, Pattern([2, 32, 32, 3], 1, 0), cfg)
        sum_error, concat_error = 0.0, 0.0
        widths = ""
        for stage, blocks in enumerate(net.blocks):
            blocks = list(blocks)
            res, den = list(blocks[-1].output.residual.values()), list(blocks[-1].output.dense.values())
            den0 = list(blocks[0].input.dense.values())
            shape = blocks[-1].output.residual.shape()
            rc, pixels = shape[3], len(res) // shape[3]
            dc, dc0 = blocks[-1].output.dense.shape()[3], blocks[0].input.dense.shape()[3]
            total = blocks[0].input.residual  # float32 tensor sums, in the same addition order
            offset = dc0
            for block in blocks:
                du = list(block.dense_update.values())
                inc = block.dense_update.shape()[3]
                total = nn.add(total, block.residual_update)
                for q in range(pixels):
                    for c in range(inc):
                        concat_error = max(concat_error, abs(den[q * dc + offset + c] - du[q * inc + c]))
                offset += inc
            for q in range(pixels):
                for c in range(dc0):
                    concat_error = max(concat_error, abs(den[q * dc + c] - den0[q * dc0 + c]))
            check.Require(offset == dc, "dense path holds exactly the projection and every unit's new channels")
            total_values = list(total.values())
            for i in range(len(res)):
                sum_error = max(sum_error, abs(res[i] - total_values[i]))
            widths += (", " if stage else "") + f"{rc}+{dc} at {shape[1]}"
    check.Near(sum_error, 0, 0, "residual path is the projection plus the sum of all residual updates")
    check.Near(concat_error, 0, 0, "dense path is the ordered concatenation of all new features")
    print(f"  compact DPN-92 paths (residual+dense channels at pixels): {widths}; "
          "residual sum and dense concatenation exact", flush=True)


def full_checks():
    counts = [(68, 12611602, 65116), (92, 37668392, 125216), (98, 61570728, 164896),
              (107, 86917800, 213824), (131, 79254504, 223360)]
    for depth, parameters, state_count in counts:
        p, state = nn.ParameterStore(17), nn.ParameterStore(17)
        cfg = dpn.DPNConfig()
        cfg.depth = depth
        with nn.no_grad():
            dpn.DPN(p, state, Pattern([1, 64, 64, 3], .1, 0), cfg)
        check.Require(p.count() == parameters and state.count() == state_count,
                      f"DPN-{depth} parameter/state counts")
        print(f"  DPN-{depth}: parameters={p.count()}, state={state.count()}", flush=True)
    p, state = nn.ParameterStore(17), nn.ParameterStore(17)
    cfg = dpn.DPNConfig()
    cfg.update_stats = False
    net = dpn.DPN(p, state, Pattern([2, 224, 224, 3], .1, 0), cfg, True)
    spatial, channels = [56, 56, 28, 14, 7], [64, 336, 704, 1552, 2688]
    for stage in range(5):
        shape = tuple(net.stem.shape() if stage == 0 else dpn.DPNJoin(net.stages(stage - 1)).shape())
        check.Require(shape == (2, spatial[stage], spatial[stage], channels[stage]), "canonical DPN-92 stage dimensions")
    check.Require(tuple(net.logits.shape()) == (2, 1000), "DPN classifier dimensions")
    minimum, maximum = architecture.GradientNormRange(p, nn.mean(nn.square(net.logits)), "DPN")
    print("  DPN-92 at 224: joined channels 64/336/704/1552/2688 at 56/56/28/14/7; "
          f"training-mode gradient norms [{g(minimum)},{g(maximum)}]", flush=True)


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    for argument in arguments:  # threads= also applies to the checks
        if argument.startswith("threads="):
            nn.set_num_threads(int(argument[8:]))
    block_checks()
    stage_checks()
    full_checks()
    cfg = image_training.ImageTrainingConfig()
    cfg.name = "dpn92_compact_cifar10"
    cfg.maximum_fit_rows = 12000
    cfg.epochs = 12
    cfg.minimum_test_accuracy = .4
    directory, explicit_epochs = parse_cifar_options(arguments, cfg)
    development, test = image_training.LoadCifar10(str(directory))

    def model(p, state, image, training):
        """Quarter-width DPN-92 with a small stem for CIFAR-10; the count check runs on every forward pass."""
        arch = dpn.DPNConfig()
        arch.width_divisor = 4
        arch.groups = 8
        arch.classes = 10
        arch.small_stem = True
        logits = dpn.DPN(p, state, image, arch, training).logits
        check.Require(p.count() == 2474330 and state.count() == 31304, "compact DPN parameter/state counts")
        return logits

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
    print("All DPN checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
