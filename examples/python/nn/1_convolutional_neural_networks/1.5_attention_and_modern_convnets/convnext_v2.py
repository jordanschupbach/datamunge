# Native Python NN example. ConvNeXt V2: GRN formula/invariance/derivative checks, the eight released sizes' costs, V2-Tiny derivatives and compact V2-Atto CIFAR-10 training; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.5_attention_and_modern_convnets/convnext_v2.py
"""ConvNeXt V2: global response normalization in place of layer scale.

Port of tflow's convnext_v2 (model: common/convnext.hpp). Check global
response normalization against its formula (per-channel spatial L2 norms
divided by their channel mean, gamma * x * N + beta + x), that it is the
identity at initialization (gamma = beta = 0), that its normalized
responses are invariant to a global rescaling of the input, and that an
all-zero input or a dead channel keeps finite, identity input derivatives.
Measure all eight released sizes at 224 pixels: Atto 3,708,400 learned
parameters (0.55G multiply-adds), Femto 5,233,240 (0.78G), Pico 9,066,280
(1.37G), Nano 15,623,800 (2.45G), Tiny 28,635,496 (4.46G), Base 88,717,800
(15.35G), Large 197,956,840 (34.36G) and Huge 660,289,640 (114.92G), the
paper's 3.7/5.2/9.1/15.6/28.6/89/198/660M and 0.55/0.78/1.37/2.45/4.47/
15.4/34.4/115G. As in tflow the multiply-adds follow from the fixed stage
sizes 56/28/14/7; the parameters are counted by building each model (here
eagerly on a 32-pixel input, since counts do not depend on the image size;
Huge allocates its 2.6 GB of weights briefly). Check every derivative of
V2-Tiny in training mode with active GRN (nonzero gamma and beta). The
masked-autoencoder (FCMAE) pretraining that motivated GRN is not
implemented here.
https://arxiv.org/abs/2301.00808
https://github.com/facebookresearch/ConvNeXt-V2

Train V2-Atto on CIFAR-10 exactly as convnext trains V1-Atto (2x2/2 stem,
stochastic depth 0.1), so the two runs compare layer scale with GRN. Use
12k fit / 5k validation, twelve epochs, crop/flip, Adam(.001), kernel
L2(.0001) and the best-validation checkpoint before evaluating all 10k
test images. Weights and drop-path masks come from nn's generator, so
trained numbers match TF only statistically. CIFAR-10 is read from the
tflow dataset cache (common/paths.hpp) or data=.
Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.

The components (convnext::GlobalResponseNorm, ConvNeXtLayout and ConvNeXt,
mobile_checks::Pattern and GradientNorms, image batching, cross-entropy and
L2) are the C++ ones through SWIG (pydatamunge.nn_components); the checks and
the experiment workflow below mirror examples/cpp/source/nn/.../convnext_v2.cpp line
by line.
"""
import math
import sys
import time
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import check, convnext, example_runtime, image_training, mobile_checks, paths

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


def grn_checks():
    """Global response normalization: identity at initialization, the formula with
    active gamma/beta, scale invariance of the responses, and finite identity
    input derivatives for an all-zero input and a dead channel."""
    size, channels = 5, 12
    p = nn.ParameterStore(17)
    value = Pattern([2, size, size, channels], 1.0, .2)
    with nn.no_grad():
        identity = nn.max(nn.abs(nn.sub(convnext.GlobalResponseNorm(p, "grn", value).output, value))).item()
        p.set("grn/gamma", Pattern([channels], .8, 1.0))
        p.set("grn/beta", Pattern([channels], .3, 2.0))
    grn = convnext.GlobalResponseNorm(p, "grn", value)
    out, response = grn.output.values(), grn.response.values()
    scaled = convnext.GlobalResponseNorm(p, "grn", nn.mul(value, 7.0)).response.values()
    gamma, beta = p.at("grn/gamma").values(), p.at("grn/beta").values()
    v = value.values()
    error, invariance = 0.0, 0.0
    for n in range(2):
        norm = [0.0] * channels
        mean = 0.0
        for i in range(size * size):
            for c in range(channels):
                norm[c] += v[(n * size * size + i) * channels + c] ** 2
        for c in range(channels):
            norm[c] = math.sqrt(norm[c])
            mean += norm[c] / channels
        for i in range(size * size):
            for c in range(channels):
                k = (n * size * size + i) * channels + c
                expected = gamma[c] * v[k] * norm[c] / (mean + 1e-6) + beta[c] + v[k]
                error = max(error, abs(out[k] - expected))
        for c in range(channels):
            invariance = max(invariance, abs(response[n * channels + c] - scaled[n * channels + c]))
    check.Near(identity, 0, 0, "GRN is the identity at initialization")
    check.Near(error, 0, 1e-5, "global response normalization")
    check.Near(invariance, 0, 1e-5, "GRN responses are invariant to input scale")

    def input_gradient(tensor):
        x = tensor.detach().requires_grad_(True)
        return nn.grad(nn.sum(convnext.GlobalResponseNorm(p, "grn", x).output), [x])[0].values()

    zero = input_gradient(nn.zeros([2, size, size, channels]))
    for gradient in zero:
        check.Near(gradient, 1, 0, "all-zero GRN has finite identity input derivative")
    mixed = list(v)
    for i in range(2 * size * size):
        mixed[i * channels] = 0.0
    dead = input_gradient(nn.from_vector(mixed, [2, size, size, channels]))
    for gradient in dead:
        check.Require(math.isfinite(gradient), "finite derivatives with a dead channel")
    for i in range(2 * size * size):
        check.Near(dead[i * channels], 1, 0, "dead channel retains identity derivative")
    print("  all-zero and single-zero-channel GRN gradients remain finite with identity derivative", flush=True)
    print(f"  GRN (12 channels): identity at initialization; output matches gamma*x*N+beta+x (error {g(error)}); "
          f"responses unchanged when the input is scaled by 7 (difference {g(invariance)})", flush=True)


def cost_checks():
    """Parameters (built eagerly on a 32-pixel input) and multiply-adds at 224
    pixels (fixed stage sizes 56/28/14/7) of the eight released V2 sizes."""
    cases = [("atto", 3708400, 547332480), ("femto", 5233240, 779629824), ("pico", 9066280, 1367055360),
             ("nano", 15623800, 2445528320), ("tiny", 28635496, 4455531264), ("base", 88717800, 15354729472),
             ("large", 197956840, 34361433600), ("huge", 660289640, 114922100480)]
    for size, expected_params, expected_macs in cases:
        layout = convnext.ConvNeXtLayout(size)
        p = nn.ParameterStore(17)
        with nn.no_grad():
            cfg = convnext.ConvNeXtConfig()
            cfg.size = size
            cfg.v2 = True
            convnext.ConvNeXt(p, nn.zeros([1, 32, 32, 3]), cfg)
            params = p.count()
        del p  # Huge's 2.6 GB of weights live only for the count
        # Stage sizes are fixed by the stem and stride-two joins: 56/28/14/7.
        depths, dims = list(layout.depths), list(layout.dims)
        macs, n = dims[-1] * 1000, 56
        for stage in range(len(depths)):
            d = dims[stage]
            macs += depths[stage] * n * n * (49 * d + 8 * d * d) + (n * n * 4 * dims[stage - 1] * d if stage else n * n * 48 * d)
            n //= 2
        check.Require(params == expected_params and macs == expected_macs, "ConvNeXt V2 parameters and multiply-adds")
        print(f"  ConvNeXt V2-{size[0].upper()}{size[1:]}: {params} parameters, {macs / 1e9:.2f}G mult-adds", flush=True)


def gradient_checks():
    """Every derivative of V2-Tiny in training mode with active GRN (nonzero gamma and beta)."""
    p = nn.ParameterStore(17)
    cfg = convnext.ConvNeXtConfig()
    cfg.v2 = True
    with nn.no_grad():
        convnext.ConvNeXt(p, nn.zeros([1, 32, 32, 3]), cfg)  # create the parameters
        for name in p.names():
            if "/grn/" in name:
                p.set(name, Pattern(list(p.at(name).shape()), .5, float(len(name))))
    net = convnext.ConvNeXt(p, Pattern([2, 224, 224, 3], 1.0, 0.0), cfg, True)
    check.Require(net.features.shape()[1] == 7 and net.features.shape()[3] == 768, "ConvNeXt V2-Tiny 7x7x768 features")
    norms = mobile_checks.GradientNorms(p, nn.mean(nn.square(net.logits)))
    minimum, maximum, smallest = math.inf, 0.0, ""
    names = p.names()
    for i, norm in enumerate(norms):
        check.Require(math.isfinite(norm), "finite ConvNeXt V2 derivative")
        if norm < minimum:
            minimum, smallest = norm, names[i]
        maximum = max(maximum, norm)
    check.Require(minimum > 1e-6 * maximum, "clearly nonzero ConvNeXt V2 derivatives")
    print(f"  ConvNeXt V2-Tiny with active GRN: {len(norms)} derivatives in [{g(minimum)},{g(maximum)}] "
          f"(smallest {smallest})", flush=True)


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    for argument in arguments:  # threads= also applies to the checks
        if argument.startswith("threads="):
            nn.set_num_threads(int(argument[8:]))
    grn_checks()
    cost_checks()
    gradient_checks()
    cfg = image_training.ImageTrainingConfig()
    cfg.name = "convnext_v2_atto_cifar10"
    cfg.maximum_fit_rows = 12000
    cfg.epochs = 12
    cfg.minimum_test_accuracy = .4
    directory, explicit_epochs = parse_cifar_options(arguments, cfg)
    development, test = image_training.LoadCifar10(str(directory))

    def model(p, state, image, training):
        architecture = convnext.ConvNeXtConfig()
        architecture.size = "atto"
        architecture.v2 = True
        architecture.classes = 10
        architecture.cifar_stem = True
        architecture.drop_path = .1
        logits = convnext.ConvNeXt(p, image, architecture, training).logits
        check.Require(p.count() == 3389170 and state.count() == 0, "CIFAR ConvNeXt V2-Atto parameter count")
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
    print("All ConvNeXt V2 checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
