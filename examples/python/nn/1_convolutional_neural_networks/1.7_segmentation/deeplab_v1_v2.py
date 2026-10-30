# Native Python NN example. DeepLab v1/v2 ASPP, CRF and dense-backbone checks plus compact shape-segmentation training; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.7_segmentation/deeplab_v1_v2.py
"""DeepLab v1/v2: atrous dense features, ASPP and dense Potts CRF refinement.
https://arxiv.org/abs/1412.7062
https://arxiv.org/abs/1606.00915
https://arxiv.org/abs/1210.5644

Port of tflow's deeplab_v1_v2. V1 full graph: VGG LargeFOV, outputstride8,
conv5rate2,fc6rate12/1024, 20505429parameters. V2: ResNet101 outputstride8
plus summed class-score ASPP rates6/12/18/24,44048532parameters. Check every
full derivative, independent host ASPP values/input gradients and all
compact derivatives. Verify standard ResNet stride32 geometry/counts are
unchanged by stride16/8 options.

Check fully connected Gaussian/bilateral Potts mean field against an
independent host iteration and finite-difference unary gradients. Remove
self edges, normalize kernels symmetrically and retain differentiability
through five iterations. This exact O(pixels^2) reference is bounded4096
pixels, not the original fast permutohedral implementation.

Train compact V1(widths/8,fc128/rate4,no dropout) and V2(ResNet18/8,ASPP
rates1/2/3/4) from scratch on the shared96pixel procedural shape dataset,
600fit/120validation/120test.3epochs,balanced CE,Adam(.001),batch16,L2 .0001,
best-validation mIoU. Apply a fixed CRF to32pixel samples of the first four
held-out images, report before/after mIoU and export image/truth/raw/CRF PNG.
Full native-resolution test mIoU is reported separately. No VOC claim.

TF -> datamunge: atrous convolutions use nn::conv2d's native dilation (TF
needed expanded "portable" kernels for CPU gradients). TF builds the full
models as graphs only; here they execute forward/backward on a 33x33 image
(same parameters), so counts and derivatives come from a real run. The
procedural shapes are bit-identical to TF's; weights come from datamunge's
generator, so trained numbers match TF statistically.
Options: model=v1|v2 (default both) epochs=3 accuracy=0|1 (default0) threads=N.
Short mode verifies learning; accuracy=1 retains the .55 mIoU target.
Artifacts: output/nn/models/deeplab_v{1,2}_shapes/.

The model components (deeplab::DeepLabV1/V2/DenseResNet/AtrousScorePyramid, the
dense CRF kernel and mean field, the procedural shapes, batching, balanced CE and
IoU/policy checks) are the C++ ones through SWIG (pydatamunge.nn_components); the
workflow below mirrors examples/cpp/source/nn/.../deeplab_v1_v2.cpp line by line.
"""
import math
import os
import struct
import sys
import time
from collections import Counter
from pathlib import Path

from pydatamunge import nn
from pydatamunge.nn_components import check, deeplab, example_runtime, resnet, seg


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def f32(value):
    """Round to the nearest float32 (the C++ fixtures use float arithmetic)."""
    return struct.unpack("f", struct.pack("f", value))[0]


def wave(n, f):
    return [f(i) for i in range(n)]


def gradient_norms(p, objective, nonzero, what):
    """Returns (lowest, highest) L2 norm over the derivatives of every parameter."""
    params = p.parameters()
    grads = nn.grad(objective, params)
    names = p.names()
    low, high = math.inf, 0.0
    for k in range(len(grads)):
        norm = math.sqrt(nn.sum(nn.square(grads[k].to(nn.DType.Float64))).item())
        check.Require(math.isfinite(norm) and (not nonzero or norm > 0),
                      "finite nonzero " + what + " derivative: " + names[k])
        low = min(low, norm)
        high = max(high, norm)
    return low, high


def asp_checks():
    p = nn.ParameterStore(17)
    rates = [1, 2, 3, 4]
    value = wave(13 * 15, lambda i: f32(f32(math.sin(f32(f32(.13) * i))) - f32(.3)))
    weights = wave(13 * 15, lambda i: f32(math.cos(f32(f32(.21) * i))))
    image = nn.from_vector(value, [1, 13, 15, 1])
    image.requires_grad_(True)
    with nn.no_grad():
        deeplab.AtrousScorePyramid(p, "aspp", image, 1, rates)
    for i in range(4):
        name = "aspp/rate" + str(rates[i])
        p.set(name + "/W", nn.full([3, 3, 1, 1], f32(f32(.1) * (i + 1))))
        p.set(name + "/b", nn.full([1], f32(f32(.01) * (i + 1))))
    branches = deeplab.AtrousScorePyramid(p, "aspp", image, 1, rates)
    output = branches[0]
    for i in range(1, 4):
        output = nn.add(output, branches[i])
    gradient = nn.grad(nn.sum(nn.mul(output, nn.from_vector(weights, [1, 13, 15, 1]))), [image])[0].values()
    out = output.values()
    expected, derivative = [0.0] * (13 * 15), [0.0] * (13 * 15)
    for r in range(13):
        for c in range(15):
            for branch in range(4):
                expected[r * 15 + c] += f32(f32(.01) * (branch + 1))
                coefficient = f32(f32(.1) * (branch + 1))
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        y, x = r + dy * rates[branch], c + dx * rates[branch]
                        if y < 0 or x < 0 or y >= 13 or x >= 15:
                            continue
                        expected[r * 15 + c] += coefficient * value[y * 15 + x]
                        derivative[y * 15 + x] += coefficient * weights[r * 15 + c]
    error = 0.0
    for i in range(13 * 15):
        error = max(error, abs(out[i] - expected[i]))
        error = max(error, abs(gradient[i] - derivative[i]))
    check.Near(error, 0, 3e-6, "independent four-rate atrous score sum and all input derivatives")
    print(f"  ASPP rates1/2/3/4 sum and input derivatives match host taps; error={g(error)}", flush=True)


def crf_reference(logits, kernel, iterations):
    def softmax(x):
        p = [0.0] * len(x)
        for i in range(3):
            maximum = max(x[2 * i], x[2 * i + 1])
            a, b = math.exp(x[2 * i] - maximum), math.exp(x[2 * i + 1] - maximum)
            p[2 * i] = a / (a + b)
            p[2 * i + 1] = b / (a + b)
        return p

    q = softmax(logits)
    for _step in range(iterations):
        next_logits = list(logits)
        for i in range(3):
            for j in range(3):
                for c in range(2):
                    next_logits[2 * i + c] += kernel[i * 3 + j] * q[2 * j + c]
        q = softmax(next_logits)
    return q


def crf_checks():
    pixels = [0.0] * 9
    for i in range(3):
        for c in range(3):
            pixels[i * 3 + c] = .9 if i == 2 else .1
    cfg = seg.DenseCrfConfig()
    cfg.spatial_weight = .1
    cfg.appearance_weight = 4
    cfg.appearance_sigma = 2
    kernel = seg.DenseCrfKernel(nn.from_vector(pixels, [1, 3, 3]), cfg)
    k = kernel.values()
    for i in range(3):
        for j in range(3):
            v = k[i * 3 + j]
            check.Require(math.isfinite(v) and v >= 0, "nonnegative finite CRF kernel")
            check.Near(v, k[j * 3 + i], 0, "symmetric normalization")
            if i == j:
                check.Near(v, 0, 0, "no self interaction")
    unary = [-3, 3, f32(.2), 0, 3, -3]
    weights = [f32(.1), f32(.9), f32(.3), f32(-.4), f32(.8), f32(-.2)]
    logits = nn.from_vector(unary, [3, 2])
    logits.requires_grad_(True)
    result = seg.DenseCrfMeanField(logits, kernel, 5)
    gradient = nn.grad(nn.sum(nn.mul(result, nn.from_vector(weights, [3, 2]))), [logits])[0].values()
    out = result.values()
    raw = [float(u) for u in unary]
    reference = crf_reference(raw, k, 5)

    def objective(values):
        probabilities = crf_reference(values, k, 5)
        return sum(probabilities[i] * weights[i] for i in range(6))

    error, gradient_error = 0.0, 0.0
    for i in range(6):
        error = max(error, abs(out[i] - reference[i]))
        original, step = raw[i], 1e-4
        raw[i] = original + step
        plus = objective(raw)
        raw[i] = original - step
        minus = objective(raw)
        raw[i] = original
        gradient_error = max(gradient_error, abs(gradient[i] - (plus - minus) / (2 * step)))
    check.Near(error, 0, 2e-7, "independent host CRF updates")
    check.Near(gradient_error, 0, 2e-6, "CRF input central differences")
    check.Require(out[1] > out[0] and out[3] > out[2] and out[4] > out[5],
                  "bilateral affinity corrects ambiguous same-color middle pixel and preserves color boundary")
    print(f"  Gaussian/bilateral Potts mean field matches host; forward error={g(error)}"
          f", unary derivative error={g(gradient_error)}", flush=True)


def build(p, state, images, v2, compact, training):
    """deeplab::DeepLabV1 / DeepLabV2 with the full (paper) or compact configuration."""
    if v2:
        c = deeplab.DeepLabV2Config()
        if compact:
            c.classes = 4
            c.backbone_depth = 18
            c.width_divisor = 8
            for i, rate in enumerate([1, 2, 3, 4]):
                c.set_rates(i, rate)
        return deeplab.DeepLabV2(p, state, images, c, training)
    c = deeplab.DeepLabV1Config()
    if compact:
        c.classes = 4
        c.width_divisor = 8
        c.fc_width = 128
        c.fc_rate = 4
        c.keep_probability = 1
    return deeplab.DeepLabV1(p, images, c, training)


def backbone_checks():
    for stride in (32, 16, 8):
        p, state = nn.ParameterStore(17), nn.ParameterStore(18)
        c = resnet.ResNetConfig()
        c.depth = 18
        c.width_divisor = 8
        c.include_classifier = False
        net = deeplab.DenseResNet(p, state, nn.full([1, 96, 96, 3], .5), c, stride, True)
        check.Require(p.count() == 176712 and state.count() == 1200,
                      "output-stride change preserves basic ResNet parameters and default counts")
        for stage in range(4):
            side = 96 // min(stride, 4 << stage)
            shape = net.stage(stage).shape()
            check.Require(shape[1] == side and shape[2] == side, "unchanged standard strides or requested dense stages")
    print("  default stride32 and atrous stride16/8 backbones preserve parameter/state counts "
          "and expected stage geometry", flush=True)


def model_checks():
    for v2 in (False, True):
        p, state = nn.ParameterStore(17), nn.ParameterStore(18)
        net = build(p, state, nn.from_vector(wave(33 * 33 * 3, lambda i: f32(math.sin(f32(f32(.13) * i)))),
                                             [1, 33, 33, 3]), v2, False, True)
        check.Require(p.count() == (44048532 if v2 else 20505429) and state.count() == (105344 if v2 else 0),
                      "full DeepLab trainable/state counts")
        gradient_norms(p, nn.mean(nn.square(net.logits)), False, "full DeepLab")
        check.Require(p.size() == (320 if v2 else 32), "every full DeepLab parameter connected")
        print(f"  full DeepLab v{2 if v2 else 1}: {p.count()} parameters, {p.size()} derivatives "
              "(executed on a 33x33 image)", flush=True)
    for v2 in (False, True):
        p, state = nn.ParameterStore(17), nn.ParameterStore(18)
        net = build(p, state, nn.from_vector(wave(2 * 96 * 96 * 3, lambda i: f32(math.sin(f32(f32(.13) * i)))),
                                             [2, 96, 96, 3]), v2, True, True)
        check.Require(p.count() == (185944 if v2 else 321452), "compact DeepLab parameter counts")
        features = net.features.shape()
        check.Require(tuple(net.logits.shape()) == (2, 96, 96, 4) and features[1] == 12 and features[2] == 12,
                      "stride-eight features and resized logits")
        norms = gradient_norms(p, nn.mean(nn.square(net.logits)), True, "DeepLab")
        print(f"  compact DeepLab v{2 if v2 else 1}: {p.count()} parameters; all derivative norms "
              f"[{g(norms[0])},{g(norms[1])}]", flush=True)


# ---- seg::SegmentationScore / SegmentationPreview / DenseCrfPreview take a LogitsFn and are
# ---- not bound; the loops are written out here exactly as the headers do them.

PALETTE = [[24, 24, 32], [230, 70, 70], [70, 210, 110], [80, 120, 240]]  # seg::kSegmentationPalette


def segmentation_score(infer, data):
    """Confusion (truth rows), mean IoU and pixel accuracy over `data` in batches of 16."""
    confusion = [0] * (data.classes * data.classes)
    with nn.no_grad():
        for start in range(0, data.count, 16):
            rows = list(range(start, min(start + 16, data.count)))
            images, labels = seg.SegmentationBatch(data, rows)
            for i, v in enumerate(seg.SegmentationConfusion(infer(images), labels, data.classes)):
                confusion[i] += v
    correct = sum(confusion[c * data.classes + c] for c in range(data.classes))
    total = sum(confusion)
    metrics = seg.SegmentationMetrics()
    metrics.mean_iou = seg.ConfusionMeanIou(confusion, data.classes)
    metrics.pixel_accuracy = correct / total
    metrics.confusion = confusion
    return metrics


def segmentation_preview(infer, data, filename):
    """First four images: image | truth | prediction rows."""
    with nn.no_grad():
        images, _labels = seg.SegmentationBatch(data, list(range(min(4, data.count))))
        seg.SegmentationPreviewLogits(infer(images), data, str(filename))


def dense_crf_preview(infer, data, directory):
    """Fixed CRF on 32x32 nearest samples of the first four held-out images:
    before/after mIoU and an image | truth | raw | CRF preview (seg::DenseCrfPreview)."""
    size = 32
    count, classes = min(4, data.count), data.classes
    before, after = [0] * (classes * classes), [0] * (classes * classes)
    preview = [0.0] * (count * size * 4 * size * 3)

    def put(n, r, column, k, value):
        preview[((n * size + r) * 4 * size + column) * 3 + k] = value

    with nn.no_grad():
        for n in range(count):
            batch_images, batch_labels = seg.SegmentationBatch(data, [n])
            logits = infer(batch_images).values()
            image = batch_images.values()
            labels = [int(v) for v in batch_labels.values()]
            small, unary = [0.0] * (size * size * 3), [0.0] * (size * size * classes)
            truth, raw = [0] * (size * size), [0] * (size * size)
            for r in range(size):
                for c in range(size):
                    y = min(data.height - 1, int((r + .5) * data.height / size))
                    x = min(data.width - 1, int((c + .5) * data.width / size))
                    index = r * size + c
                    source = y * data.width + x
                    truth[index] = labels[source]
                    for k in range(3):
                        small[index * 3 + k] = image[source * 3 + k]
                    for k in range(classes):
                        unary[index * classes + k] = logits[source * classes + k]
                    u = unary[index * classes:(index + 1) * classes]
                    raw[index] = max(range(classes), key=lambda k, u=u: u[k])
            kernel = seg.DenseCrfKernel(nn.from_vector(small, [size, size, 3]))
            refined = seg.DenseCrfMeanField(nn.from_vector(unary, [size * size, classes]), kernel).values()
            for r in range(size):
                for c in range(size):
                    index = r * size + c
                    value = refined[index * classes:(index + 1) * classes]
                    prediction = max(range(classes), key=lambda k, value=value: value[k])
                    before[truth[index] * classes + raw[index]] += 1
                    after[truth[index] * classes + prediction] += 1
                    for k in range(3):
                        put(n, r, c, k, float(min(max(math.floor(f32(small[index * 3 + k] * 255) + .5), 0), 255)))
                        put(n, r, size + c, k, PALETTE[truth[index] % 4][k])
                        put(n, r, 2 * size + c, k, PALETTE[raw[index] % 4][k])
                        put(n, r, 3 * size + c, k, PALETTE[prediction % 4][k])
    path = str(Path(directory) / "crf_preview.png")
    nn.save_png(nn.from_vector(preview, [count * size, 4 * size, 3]), path, 0, 255)
    print(f"  fixed CRF on first{count} held-out scenes sampled32x32: mIoU "
          f"{g(seg.ConfusionMeanIou(before, classes))} -> {g(seg.ConfusionMeanIou(after, classes))}"
          f"; image/truth/raw/CRF preview: {path}", flush=True)


def subset(data, count):
    result = seg.SegmentationData()
    result.count = min(data.count, count)
    result.height = data.height
    result.width = data.width
    result.classes = data.classes
    pixels = result.count * data.height * data.width
    result.images = data.images[:3 * pixels]
    result.labels = data.labels[:pixels]
    return result


def train_segmentation(training_cfg, source_fit, source_validation, source_test, builder, explicit_epochs, inspect):
    """The segmentation training block of the C++ main: balanced CE + L2 with Adam,
    validation-mIoU checkpoint selection, restore check, held-out diagnostics and report."""
    if not explicit_epochs:
        training_cfg.epochs = example_runtime.Limit(training_cfg.epochs, 1)
    training_cfg.batch = example_runtime.Limit(training_cfg.batch, 2)
    if not example_runtime.Full():
        quick_fit = subset(source_fit, 8)
        quick_validation = subset(source_validation, 4)
        quick_test = subset(source_test, 4)
    fit = source_fit if example_runtime.Full() else quick_fit
    validation = source_validation if example_runtime.Full() else quick_validation
    test = source_test if example_runtime.Full() else quick_test
    example_runtime.Describe()
    check.Require(training_cfg.epochs > 0 and training_cfg.batch > 0 and fit.classes == validation.classes
                  and fit.classes == test.classes, "valid segmentation configuration")
    check.Require(not example_runtime.Full() or training_cfg.accuracy or training_cfg.epochs >= 2,
                  "short segmentation demonstration needs at least two epochs")
    p, state = nn.ParameterStore(training_cfg.seed), nn.ParameterStore(training_cfg.seed)
    with nn.no_grad():  # create every parameter (no statistics update) before the optimizer
        probe = seg.SegmentationBatch(fit, [0])
        builder(p, state, probe[0], False)

    def infer(images):
        return builder(p, state, images, False).logits

    frequencies = [1] * fit.classes
    for label, n in Counter(fit.labels).items():
        frequencies[label] += n
    weights = [f32(1.0 / f32(math.sqrt(f32(f32(frequencies[c]) / f32(len(fit.labels)))))) for c in range(fit.classes)]
    class_weights = nn.from_vector(weights, [fit.classes])
    kernels = [t for t in p.parameters() if t.dim() > 1]
    oc = nn.OptimizerConfig()
    oc.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, training_cfg.learning_rate, oc)
    directory = Path(seg.SegmentationDirectory(training_cfg.name))
    prefix = str(directory / "best")
    history = (directory / "history.tsv").open("w")
    history.write("epoch\tloss\tvalidation_mean_iou\n")
    print(f"== {training_cfg.name}: {fit.count} fit / {validation.count} validation / {test.count} test; "
          f"{p.count()} parameters; {fit.height}x{fit.width}, classes={fit.classes}", flush=True)
    batches = nn.BatchOrder(fit.count, training_cfg.batch, 29)
    best, best_epoch = -1.0, -1
    epoch_losses = []
    if training_cfg.warm_start:
        check.Require(os.path.normpath(os.path.abspath(training_cfg.warm_start)) != os.path.normpath(os.path.abspath(prefix)),
                      "segmentation warm start requires a separate output tag")
        p.load(training_cfg.warm_start + "_params.npz")
        state.load(training_cfg.warm_start + "_state.npz")
        best = segmentation_score(infer, validation).mean_iou
        best_epoch = 0
        p.save(prefix + "_params.npz")
        state.save(prefix + "_state.npz")
        history.write(f"0\tnan\t{g(best)}\n")
        history.flush()
        print(f"  warm start: {training_cfg.warm_start}; fresh Adam moments; restored validation mIoU={g(best)}", flush=True)
    train_seconds = 0.0
    for epoch in range(1, training_cfg.epochs + 1):
        started = time.perf_counter()
        batches.begin_epoch()
        total, seen = 0.0, 0
        while True:
            rows = [int(r) for r in batches.next()]
            if not rows:
                break
            images, labels = seg.SegmentationBatch(fit, rows)
            model = builder(p, state, images, True)
            loss = seg.SegmentationCrossEntropy(model.logits, labels, fit.classes, class_weights)
            if len(model.auxiliary):
                total_loss = loss
                for output in model.auxiliary:
                    total_loss = nn.add(total_loss, nn.mul(
                        seg.SegmentationCrossEntropy(output, labels, fit.classes, class_weights), float(model.auxiliary_weight)))
                loss = nn.div(total_loss, 1.0 + float(model.auxiliary_weight) * float(len(model.auxiliary)))
            value = loss.item()
            check.Require(math.isfinite(value), "finite segmentation training loss")
            objective = loss
            if training_cfg.l2 > 0 and kernels:
                penalty = nn.sum(nn.square(kernels[0]))
                for kernel in kernels[1:]:
                    penalty = nn.add(penalty, nn.sum(nn.square(kernel)))
                objective = nn.add(objective, nn.mul(penalty, .5 * float(training_cfg.l2)))
            optimizer.minimize(objective)
            total += value * len(rows)
            seen += len(rows)
        train_seconds += time.perf_counter() - started
        epoch_losses.append(total / seen)
        check.Require(math.isfinite(epoch_losses[-1]), "finite segmentation epoch mean")
        score = segmentation_score(infer, validation)
        check.Require(math.isfinite(score.mean_iou), "finite validation mIoU")
        if score.mean_iou > best:
            best, best_epoch = score.mean_iou, epoch
            p.save(prefix + "_params.npz")
            state.save(prefix + "_state.npz")
        history.write(f"{epoch}\t{g(total / seen)}\t{g(score.mean_iou)}\n")
        history.flush()
        print(f"  epoch {epoch}: loss={g(total / seen)}, validation mIoU={g(score.mean_iou)}"
              f", pixel accuracy={g(score.pixel_accuracy)}", flush=True)
    history.close()
    p.load(prefix + "_params.npz")
    state.load(prefix + "_state.npz")
    check.Near(segmentation_score(infer, validation).mean_iou, best, 1e-10,
               "restored validation score matches selected checkpoint")
    print(f"  restored validation mIoU={g(best)}; checkpoint agreement verified", flush=True)
    if inspect:
        inspect(infer, test, directory)
    result = segmentation_score(infer, test)
    print(f"  selected epoch={best_epoch}; test mIoU={g(result.mean_iou)}, pixel accuracy={g(result.pixel_accuracy)}"
          "\n  test confusion (truth rows, predicted columns):")
    for c in range(fit.classes):
        print("   " + "".join(f" {result.confusion[c * fit.classes + k]}" for k in range(fit.classes)))
    segmentation_preview(infer, test, directory / "preview.png")
    print(f"  checkpoint, history and image/truth/prediction preview: {directory}", flush=True)
    print(f"  training time {train_seconds:.1f} s ({train_seconds / training_cfg.epochs:.2f} s/epoch, excluding validation)",
          flush=True)
    if example_runtime.Full():
        seg.CheckSegmentationDemonstration(training_cfg, epoch_losses, result)
    else:
        check.Require(math.isfinite(result.mean_iou) and 0 <= result.mean_iou <= 1, "finite quick segmentation evaluation")
    if example_runtime.Full() and not training_cfg.accuracy:
        print(f"  short learning demonstration passed: mean loss {g(epoch_losses[0])} -> {g(epoch_losses[-1])}"
              f"; accuracy=1 enables mIoU target {g(training_cfg.minimum_test_iou)}", flush=True)


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    epochs, accuracy, models = 3, False, [False, True]
    for arg in arguments:
        if arg.startswith("epochs="):
            epochs = int(arg[7:])
        elif arg in ("accuracy=0", "accuracy=1"):
            accuracy = arg == "accuracy=1"
        elif arg == "model=v1":
            models = [False]
        elif arg == "model=v2":
            models = [True]
        elif arg.startswith("threads="):
            nn.set_num_threads(int(arg[8:]))
        else:
            raise ValueError("expected epochs=N, model=v1|v2, accuracy=0|1 or threads=N")
    seg.SegmentationTrainingChecks()
    seg.SegmentationLossChecks()
    asp_checks()
    crf_checks()
    backbone_checks()
    model_checks()
    fit = seg.SegmentationShapes(600, 96, 11)
    validation = seg.SegmentationShapes(120, 96, 22)
    test = seg.SegmentationShapes(120, 96, 33)
    explicit_epochs = any(arg.startswith("epochs=") for arg in arguments)
    for v2 in models:
        cfg = seg.SegmentationConfig()
        cfg.name = "deeplab_v2_shapes" if v2 else "deeplab_v1_shapes"
        cfg.epochs = epochs
        cfg.accuracy = accuracy

        def builder(p, state, images, training, v2=v2):
            model = seg.SegmentationModel()
            model.logits = build(p, state, images, v2, True, training).logits
            return model

        train_segmentation(cfg, fit, validation, test, builder, explicit_epochs, dense_crf_preview)
    print("All DeepLab v1/v2 and CRF checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
