# Native Python NN example. FCN-32s/16s/8s checks and compact procedural-shape segmentation; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.7_segmentation/fcn.py
"""FCN-32s/16s/8s: dense VGG predictions, bilinear deconvolution and skip fusion.
https://arxiv.org/abs/1411.4038
https://github.com/shelhamer/fcn.berkeleyvision.org

Port of tflow's fcn. Construct the original 21-class VGG graphs, including
pad100/ceil pooling, valid7x7 fc6, pool4/pool3 crop offsets5/9, and final
crop19/27/31. Check independent trainable parameter counts
134346581/134357354/134362751 and every derivative. Bilinear filters are
fixed diagonal constants, matching the release's frozen upsampling; learned
skip projections retain signed scores. Check deconvolution/skip/crop values
and input derivatives against independent host scatter calculations,
including class isolation. Check every compact derivative and output sizes
95/96/97.

Compare compact versions on procedural circles/rectangles/triangles with
class-independent intensities/tints, occlusion and noise, requiring spatial
shape recognition. This is explicitly a teaching dataset, not Pascal VOC.
All thirteen VGG convolutions remain; widths/8, fc6 3x3/64, SAME padding,
dropout0 and crop0 reduce computation. Train from scratch with inverse-sqrt
fit-frequency pixel CE, Adam(.001), batch16,L2 .0001,twenty epochs and best
validation mean IoU; 600 fit/120 validation/120 test images,96x96. Report
confusion and export image/truth/prediction PNGs. No pretrained weights.

TF -> datamunge: TF builds the full models as graphs only; here the full
models execute forward and backward on an 8x8 image (pad 100 keeps every
layer valid), so the counts and derivatives come from a real run. The
procedural shapes are generated bit-identically to TF's; weights come from
datamunge's generator, so trained numbers match TF statistically.
Options: stride=32|16|8 (default all) epochs=20 threads=N.
Artifacts: output/nn/models/fcn{32,16,8}_shapes/.

The model components (fcn::Fcn, FcnFuseSkip, BilinearKernel, the seg::
procedural shapes, batching, balanced pixel CE, confusion and preview helpers)
are the C++ ones through SWIG (pydatamunge.nn_components); the checks and the
training workflow below mirror examples/cpp/source/nn/.../fcn.cpp line by line.
"""
import math
import os
import struct
import sys
import time

from pydatamunge import nn
from pydatamunge.nn_components import check, example_runtime, fcn, seg


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def f32(value):
    """Rounds a double to float32, as storing into a std::vector<float> does."""
    return struct.unpack("f", struct.pack("f", value))[0]


def wave(n, scale, offset=0, cosine=False):
    """std::vector<float> of float(offset + sin/cos(scale * i))."""
    return [f32(offset + (math.cos(scale * i) if cosine else math.sin(scale * i))) for i in range(n)]


def gradient_norms(p, objective, nonzero, what):
    """Max/min gradient norm (and count) over every parameter; requires finite (and nonzero when `nonzero`)."""
    params = p.parameters()
    grads = nn.grad(objective, params)
    check.Require(len(grads) == len(params), what + ": every parameter has a derivative")
    names = p.names()
    low, high = math.inf, 0.0
    for k, gradient in enumerate(grads):
        norm = math.sqrt(nn.sum(nn.square(gradient.to(nn.DType.Float64))).item())
        check.Require(math.isfinite(norm) and (not nonzero or norm > 0),
                      "finite nonzero " + what + " derivative: " + names[k])
        low = min(low, norm)
        high = max(high, norm)
    return low, high


def fusion_checks():
    a = nn.from_vector(wave(24, .7), [1, 3, 4, 2])
    b = nn.from_vector(wave(160, .31, -.2, True), [1, 8, 10, 2])
    weights = wave(96, .13)
    a.requires_grad_(True)
    b.requires_grad_(True)
    fused = fcn.FcnFuseSkip(a, b, 2, False, 1)
    gradients = nn.grad(nn.sum(nn.mul(fused, nn.from_vector(weights, [1, 6, 8, 2]))), [a, b])
    av, bv, out = a.values(), b.values(), fused.values()
    ga_out, gb_out = gradients[0].values(), gradients[1].values()
    expected, ga, gb = [0.0] * (6 * 8 * 2), [0.0] * (3 * 4 * 2), [0.0] * (8 * 10 * 2)
    for r in range(6):
        for c in range(8):
            for k in range(2):
                index = (r * 8 + c) * 2 + k
                expected[index] = bv[((r + 1) * 10 + c + 1) * 2 + k]
                gb[((r + 1) * 10 + c + 1) * 2 + k] = weights[index]
    for r in range(3):
        for c in range(4):
            for dy in range(4):
                for dx in range(4):
                    y, x = 2 * r + dy - 1, 2 * c + dx - 1
                    if y < 0 or x < 0 or y >= 6 or x >= 8:
                        continue
                    factor = (1 - abs(dy - 1.5) / 2) * (1 - abs(dx - 1.5) / 2)
                    for k in range(2):
                        destination, source = (y * 8 + x) * 2 + k, (r * 4 + c) * 2 + k
                        expected[destination] += factor * av[source]
                        ga[source] += factor * weights[destination]
    check.Require(tuple(fused.shape()) == (1, 6, 8, 2), "SAME bilinear fusion shape")
    error = 0.0
    for i in range(len(out)):
        error = max(error, abs(out[i] - expected[i]))
    for i in range(len(ga_out)):
        error = max(error, abs(ga_out[i] - ga[i]))
    for i in range(len(gb_out)):
        error = max(error, abs(gb_out[i] - gb[i]))
    check.Near(error, 0, 1e-6, "host bilinear scatter, signed skip crop and every input gradient")
    kernel = fcn.BilinearKernel(2, 2).values()  # [4, 4, 2, 2]
    check.Near(kernel[((1 * 4 + 1) * 2 + 0) * 2 + 0], .5625, 0, "bilinear kernel centre")
    check.Near(kernel[((1 * 4 + 1) * 2 + 0) * 2 + 1], 0, 0, "no cross-class mixing")
    print("  signed bilinear/skip/crop values and every input derivative match host scatter, error=" + g(error))


def compact_config(stride):
    c = fcn.FcnConfig()
    c.stride = stride
    c.classes = 4
    c.width_divisor = 8
    c.fc_width = 64
    c.compact = True
    c.keep_probability = 1
    return c


def model_checks():
    strides = (32, 16, 8)
    full_counts = (134346581, 134357354, 134362751)
    compact_counts = (271916, 272176, 272308)
    for v in range(3):
        p = nn.ParameterStore(17)
        c = fcn.FcnConfig()
        c.stride = strides[v]
        net = fcn.Fcn(p, nn.from_vector(wave(8 * 8 * 3, .13), [1, 8, 8, 3]), c, True)
        check.Require(p.count() == full_counts[v], "full FCN trainable count excludes frozen bilinear filters")
        check.Require(tuple(net.logits.shape()) == (1, 8, 8, 21), "full FCN crops logits to the input grid")
        gradient_norms(p, nn.mean(nn.square(net.logits)), False, "full FCN")
        check.Require(p.size() == 32 + 2 * v, "full FCN parameter connections")
        print(f"  full FCN-{strides[v]}s: {p.count()} parameters, {p.size()} derivatives (executed on an 8x8 image)")
    for v in range(3):
        p = nn.ParameterStore(17)
        low, high = math.inf, 0.0
        for side in (95, 96, 97):
            net = fcn.Fcn(p, nn.from_vector(wave(2 * side * side * 3, .13), [2, side, side, 3]),
                          compact_config(strides[v]), True)
            check.Require(p.count() == compact_counts[v], "compact FCN parameter count")
            check.Require(tuple(net.logits.shape()) == (2, side, side, 4), "FCN logits align to original image dimensions")
            if side != 96:
                continue
            norms = gradient_norms(p, nn.mean(nn.square(net.logits)), True, "FCN")
            low = min(low, norms[0])
            high = max(high, norms[1])
        print(f"  compact FCN-{strides[v]}s: {p.count()} parameters; exact95/96/97 output grids; "
              f"gradient norms [{g(low)},{g(high)}]")


# ---- seg:: helpers that take a LogitsFn callback (not bound), written out over the bound pieces ----

def segmentation_model(logits, auxiliary=()):
    """seg::SegmentationModel{logits, auxiliary}."""
    model = seg.SegmentationModel()
    model.logits = logits
    model.auxiliary = list(auxiliary)
    return model


def copy_config(cfg):
    """Value copy of a seg::SegmentationConfig (the C++ `auto training_cfg = cfg`)."""
    copy = seg.SegmentationConfig()
    for field in ("name", "warm_start", "epochs", "batch", "learning_rate", "l2", "minimum_test_iou", "accuracy", "seed"):
        setattr(copy, field, getattr(cfg, field))
    return copy


def segmentation_score(infer, data):
    """seg::SegmentationScore: inference logits in batches of 16 rows, one confusion matrix (truth rows),
    mean IoU over present classes and pixel accuracy."""
    confusion = [0] * (data.classes * data.classes)
    with nn.no_grad():
        for start in range(0, data.count, 16):
            rows = list(range(start, min(start + 16, data.count)))
            x, y = seg.SegmentationBatch(data, rows)
            for i, count in enumerate(seg.SegmentationConfusion(infer(x), y, data.classes)):
                confusion[i] += count
    result = seg.SegmentationMetrics()
    result.confusion = confusion
    result.mean_iou = seg.ConfusionMeanIou(confusion, data.classes)
    result.pixel_accuracy = sum(confusion[c * data.classes + c] for c in range(data.classes)) / sum(confusion)
    return result


def segmentation_preview(infer, data, filename):
    """seg::SegmentationPreview: image | truth | prediction rows of the first four images."""
    with nn.no_grad():
        x, _y = seg.SegmentationBatch(data, list(range(min(4, data.count))))
        seg.SegmentationPreviewLogits(infer(x), data, filename)


def run_segmentation(builder, cfg, source_fit, source_validation, source_test, explicit_epochs):
    """The segmentation training block of the C++ example: quick-mode subsets, inverse-sqrt class
    weights, Adam on balanced pixel CE plus L2, validation-mIoU checkpoint selection, test confusion
    and the preview PNG."""
    training_cfg = copy_config(cfg)
    if not explicit_epochs:
        training_cfg.epochs = example_runtime.Limit(training_cfg.epochs, 1)
    training_cfg.batch = example_runtime.Limit(training_cfg.batch, 2)

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

    if not example_runtime.Full():
        fit, validation, test = subset(source_fit, 8), subset(source_validation, 4), subset(source_test, 4)
    else:
        fit, validation, test = source_fit, source_validation, source_test
    example_runtime.Describe()
    check.Require(training_cfg.epochs > 0 and training_cfg.batch > 0 and fit.classes == validation.classes
                  and fit.classes == test.classes, "valid segmentation configuration")
    check.Require(not example_runtime.Full() or training_cfg.accuracy or training_cfg.epochs >= 2,
                  "short segmentation demonstration needs at least two epochs")
    p, state = nn.ParameterStore(training_cfg.seed), nn.ParameterStore(training_cfg.seed)
    with nn.no_grad():  # create every parameter (no statistics update) before the optimizer
        builder(p, state, seg.SegmentationBatch(fit, [0])[0], False)

    def infer(images):
        return builder(p, state, images, False).logits

    labels = fit.labels
    frequencies = [1] * fit.classes
    for label in labels:
        frequencies[label] += 1
    # 1 / sqrt(float(frequency) / float(pixels)) in float32 arithmetic, through the tensor ops.
    class_weights = nn.div(1.0, nn.sqrt(nn.div(nn.from_vector([float(f) for f in frequencies], [fit.classes]),
                                               float(len(labels)))))
    kernels = [t for t in p.parameters() if t.dim() > 1]
    oc = nn.OptimizerConfig()
    oc.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(p, training_cfg.learning_rate, oc)
    directory = seg.SegmentationDirectory(training_cfg.name)
    prefix = os.path.join(directory, "best")
    history = open(os.path.join(directory, "history.tsv"), "w")
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
            rows = list(batches.next())
            if not rows:
                break
            x, y = seg.SegmentationBatch(fit, rows)
            model = builder(p, state, x, True)
            loss = seg.SegmentationCrossEntropy(model.logits, y, fit.classes, class_weights)
            if len(model.auxiliary):
                total_loss = loss
                for output in model.auxiliary:
                    total_loss = nn.add(total_loss, nn.mul(
                        seg.SegmentationCrossEntropy(output, y, fit.classes, class_weights), model.auxiliary_weight))
                loss = nn.div(total_loss, 1.0 + model.auxiliary_weight * len(model.auxiliary))
            value = loss.item()
            check.Require(math.isfinite(value), "finite segmentation training loss")
            objective = loss
            if training_cfg.l2 > 0 and kernels:
                penalty = nn.sum(nn.square(kernels[0]))
                for kernel in kernels[1:]:
                    penalty = nn.add(penalty, nn.sum(nn.square(kernel)))
                objective = nn.add(objective, nn.mul(penalty, .5 * training_cfg.l2))
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
        print(f"  epoch {epoch}: loss={g(total / seen)}, validation mIoU={g(score.mean_iou)}, "
              f"pixel accuracy={g(score.pixel_accuracy)}", flush=True)
    p.load(prefix + "_params.npz")
    state.load(prefix + "_state.npz")
    check.Near(segmentation_score(infer, validation).mean_iou, best, 1e-10,
               "restored validation score matches selected checkpoint")
    print(f"  restored validation mIoU={g(best)}; checkpoint agreement verified", flush=True)
    # training_cfg.inspect is never set by this example.
    result = segmentation_score(infer, test)
    print(f"  selected epoch={best_epoch}; test mIoU={g(result.mean_iou)}, pixel accuracy={g(result.pixel_accuracy)}\n"
          "  test confusion (truth rows, predicted columns):")
    for c in range(fit.classes):
        print("   " + "".join(f" {result.confusion[c * fit.classes + k]}" for k in range(fit.classes)))
    segmentation_preview(infer, test, os.path.join(directory, "preview.png"))
    print(f"  checkpoint, history and image/truth/prediction preview: {directory}", flush=True)
    print(f"  training time {train_seconds:.1f} s ({train_seconds / training_cfg.epochs:.2f} s/epoch, excluding validation)")
    if example_runtime.Full():
        seg.CheckSegmentationDemonstration(training_cfg, epoch_losses, result)
    else:
        check.Require(math.isfinite(result.mean_iou) and 0 <= result.mean_iou <= 1, "finite quick segmentation evaluation")
    if example_runtime.Full() and not training_cfg.accuracy:
        print(f"  short learning demonstration passed: mean loss {g(epoch_losses[0])} -> {g(epoch_losses[-1])}; "
              f"accuracy=1 enables mIoU target {g(training_cfg.minimum_test_iou)}")
    history.close()


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    epochs, strides = 20, [32, 16, 8]
    for arg in arguments:
        if arg.startswith("epochs="):
            epochs = int(arg[7:])
        elif arg.startswith("stride="):
            stride = int(arg[7:])
            if stride not in (32, 16, 8):
                raise ValueError("stride must be32,16or8")
            strides = [stride]
        elif arg.startswith("threads="):
            nn.set_num_threads(int(arg[8:]))
        else:
            raise ValueError("expected epochs=N, stride=32|16|8 or threads=N")
    seg.SegmentationLossChecks()
    fusion_checks()
    model_checks()
    fit = seg.SegmentationShapes(600, 96, 11)
    validation = seg.SegmentationShapes(120, 96, 22)
    test = seg.SegmentationShapes(120, 96, 33)
    explicit_epochs = any(arg.startswith("epochs=") for arg in arguments)
    for stride in strides:
        cfg = seg.SegmentationConfig()
        cfg.name = f"fcn{stride}_shapes"
        cfg.epochs = epochs

        def builder(p, state, images, training, stride=stride):
            return segmentation_model(fcn.Fcn(p, images, compact_config(stride), training).logits)

        run_segmentation(builder, cfg, fit, validation, test, explicit_epochs)
    print("All FCN-32s/16s/8s checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
