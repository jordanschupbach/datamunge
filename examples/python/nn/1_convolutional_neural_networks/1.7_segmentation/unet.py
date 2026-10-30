# Native Python NN example. U-Net geometry/crop/parameter checks and compact procedural-shape segmentation; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.7_segmentation/unet.py
"""U-Net: crop-and-concatenate skips and learned symmetric upsampling.
https://arxiv.org/abs/1505.04597

Port of tflow's unet. Construct the original valid-convolution U-Net:
base 64, four down/up levels, grayscale 572x572 -> 388x388, two classes,
31030658 trainable parameters. Check every full derivative, centre-crop
values/derivatives, canonical intermediate sizes and compact gradients.
The learned 2x2 transpose convolutions halve channels before concatenating
skips.

Train a compact base 8, SAME-padding, RGB/four-class network with no dropout
on the same independent procedural shape segmentation splits as FCN:
600 fit/120 validation/120 test, 96 pixels, occlusion/noise, class-independent
photometry. Preserve all four contracting/expanding levels. Use three
epochs, balanced pixel CE, Adam(.001), batch 16, L2 .0001, best validation
mIoU. Report test confusion and export input/truth/prediction PNG. This
teaching dataset does not imply biomedical or Pascal VOC benchmark
performance.

TF -> datamunge: TF builds the full 572x572 model as a graph only; here the
full model executes forward and backward on a 188x188 image (the smallest
valid input, 4x4 output), so the parameter census and derivative count come
from a real run; the 572 -> 388 geometry is checked with UnetOutputSide.
The shapes are generated bit-identically to TF's; weights come from
datamunge's generator, so trained numbers match TF statistically.
Options: epochs=3 accuracy=0|1 (default 0) warm_start=PREFIX threads=N.
Short mode checks learning; accuracy=1 retains the .55 test mIoU target.
Artifacts: output/nn/models/unet_shapes/.

The model components (unet::Unet, UnetOutputSide, CenterCropTo, the seg::
procedural shapes, batching, balanced pixel CE, policy checks, confusion and
preview helpers) are the C++ ones through SWIG (pydatamunge.nn_components);
the checks and the training workflow below mirror examples/cpp/source/nn/.../unet.cpp
line by line.
"""
import math
import os
import sys
import time

from pydatamunge import nn
from pydatamunge.nn_components import check, example_runtime, seg, unet


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def wave(shape, scale):
    """float(std::sin(scale * i)) over the shape's elements (float32 through from_vector)."""
    n = math.prod(shape)
    return nn.from_vector([math.sin(scale * i) for i in range(n)], shape)


def gradient_norms(p, objective, nonzero, what):
    """[min, max] gradient norm over every parameter; requires finite (and nonzero when `nonzero`)."""
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


def crop_checks():
    check.Require(unet.UnetOutputSide(572) == 388 and unet.UnetOutputSide(96, 4, False) == 96,
                  "canonical and SAME U-Net geometry")
    rejected = False
    try:
        unet.UnetOutputSide(64)
    except RuntimeError:  # std::invalid_argument
        rejected = True
    check.Require(rejected, "too-small valid input rejected")
    values = [float(i) - 100 for i in range(9 * 12 * 2)]
    x = nn.from_vector(values, [1, 9, 12, 2])
    x.requires_grad_(True)
    cropped = unet.CenterCropTo(x, nn.zeros([1, 4, 7, 3]))
    gradient = nn.grad(nn.sum(cropped), [x])[0].values()
    check.Require(tuple(cropped.shape()) == (1, 4, 7, 2), "centre crop preserves source channels")
    out = cropped.values()
    for r in range(4):
        for c in range(7):
            for k in range(2):
                check.Near(out[(r * 7 + c) * 2 + k], values[((r + 2) * 12 + c + 2) * 2 + k], 0,
                           "odd difference uses floor-centred crop")
    for r in range(9):
        for c in range(12):
            for k in range(2):
                check.Near(gradient[(r * 12 + c) * 2 + k], 1 if (2 <= r < 6 and 2 <= c < 9) else 0, 0,
                           "crop Jacobian masks exactly the retained region")
    print("  572->388 geometry, too-small rejection and signed rectangular centre-crop/Jacobian checks passed")


def compact_config():
    c = unet.UnetConfig()
    c.classes = 4
    c.input_channels = 3
    c.base_width = 8
    c.valid = False
    c.dropout = 0
    return c


def model_checks():
    p = nn.ParameterStore(17)
    side = 188
    check.Require(unet.UnetOutputSide(side) == 4, "smallest valid full U-Net input")
    net = unet.Unet(p, wave([1, side, side, 1], .13), unet.UnetConfig(), False)
    check.Require(p.count() == 31030658, "original grayscale/two-class U-Net parameter count")
    check.Require(tuple(net.logits.shape()) == (1, 4, 4, 2), "valid U-Net output geometry")
    check.Require(net.skips[0].shape()[1] == 184 and net.decoder[len(net.decoder) - 1].shape()[1] == 4,
                  "valid skip/decoder sizes")
    gradient_norms(p, nn.mean(nn.square(net.logits)), False, "full U-Net")
    check.Require(p.size() == 46, "full U-Net parameter connections")
    print("  original U-Net: 31030658 parameters, 46 derivatives (executed on 188x188 -> 4x4), four cropped skips")
    p = nn.ParameterStore(17)
    low, high = math.inf, 0.0
    for side in (96, 112, 128):
        net = unet.Unet(p, wave([2, side, side, 3], .13), compact_config(), True)
        check.Require(tuple(net.logits.shape()) == (2, side, side, 4), "SAME U-Net output alignment")
        if side != 96:
            continue
        low, high = gradient_norms(p, nn.mean(nn.square(net.logits)), True, "U-Net")
    print(f"  compact U-Net: {p.count()} parameters; 96/112/128 alignment; all gradient norms [{g(low)},{g(high)}]")


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
    cfg = seg.SegmentationConfig()
    cfg.name = "unet_shapes"
    cfg.epochs = 3
    cfg.accuracy = False
    for arg in arguments:
        if arg.startswith("epochs="):
            cfg.epochs = int(arg[7:])
        elif arg in ("accuracy=0", "accuracy=1"):
            cfg.accuracy = arg == "accuracy=1"
        elif arg.startswith("warm_start="):
            cfg.warm_start = arg[11:]
        elif arg.startswith("threads="):
            nn.set_num_threads(int(arg[8:]))
        else:
            raise ValueError("expected epochs=N, accuracy=0|1, warm_start=PREFIX or threads=N")
    seg.SegmentationTrainingChecks()
    seg.SegmentationLossChecks()
    crop_checks()
    model_checks()
    fit = seg.SegmentationShapes(600, 96, 11)
    validation = seg.SegmentationShapes(120, 96, 22)
    test = seg.SegmentationShapes(120, 96, 33)

    def builder(p, state, images, training):
        return segmentation_model(unet.Unet(p, images, compact_config(), training).logits)

    explicit_epochs = any(arg.startswith("epochs=") for arg in arguments)
    run_segmentation(builder, cfg, fit, validation, test, explicit_epochs)
    print("All U-Net checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
