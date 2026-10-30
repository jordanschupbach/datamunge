# Native Python NN example. HRNetV2 exchange-fusion/model checks and compact procedural-shapes segmentation training; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.7_segmentation/hrnet.py
"""HRNetV2: persistent high-resolution branches and repeated exchange fusion.
https://arxiv.org/abs/1908.07919
https://github.com/HRNet/HRNet-Semantic-Segmentation/blob/HRNet-OCR/lib/models/seg_hrnet.py

Port of tflow's hrnet. W32: two stride2 stem convolutions,4 bottlenecks/64,
then2/3/4 parallel streams of width32/64/128/256. Stage module counts1/4/3,
each4basic blocks per branch,all-to-all sum/ReLU fusion. Upsampling projects
with1x1 and aligned bilinear resize; downsampling uses3x3/stride2 chains.
Final streams concatenate at stride4,1x1/480 BN/ReLU and linear classes21,
then resize. Full-model derivatives, controlled identity-kernel odd-grid
exchange against independent host interpolation/subsampling and all pixel
gradients; compact forward/backward on96/97px checks every branch and
parameter derivative.

Compact W8,stem16,stage1width16/two bottlenecks,two basic blocks per branch,
one module per stage. Train600fit/120validation/120test procedural96px shape
scenes for3epochs,Adam(.001),batch16,balanced CE,L2 .0001. Select validation
mIoU,restore checkpoint,report test confusion and image/truth/prediction PNG.
Scratch teaching data; no pretrained medical/Cityscapes claims.

TF -> datamunge: TF builds the full W32 model as a graph only; here it
executes forward/backward on a 33x33 image (same parameters). The shapes
are bit-identical to TF's; weights come from datamunge's generator, so
trained numbers match TF statistically.
Options: epochs=3 accuracy=0|1 (default0; accuracy=1 retains .55 mIoU) threads=N.
Artifacts: output/nn/models/hrnet_shapes/.

The components (hrnet::HRNet/HRFuse, resnet::ConvNormConfig, the seg::
procedural shapes, batching, balanced cross-entropy, confusion/mIoU and
preview) are the C++ ones through SWIG (pydatamunge.nn_components); the checks
and the training workflow below mirror examples/cpp/source/nn/.../hrnet.cpp line by line.
"""
import math
import os
import struct
import sys
import time

from pydatamunge import nn
from pydatamunge.nn_components import check, example_runtime, hrnet, resnet, seg


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def f32(value):
    """Rounds to IEEE single precision (the C++ fixtures are float arithmetic)."""
    return struct.unpack("f", struct.pack("f", value))[0]


def fusion_checks():
    """Identity-kernel two-branch exchange on a 5x7/3x4 odd grid against host bilinear/subsample values and gradients."""
    p, state = nn.ParameterStore(17), nn.ParameterStore(18)
    norm = resnet.ConvNormConfig()
    norm.scale = True
    norm.epsilon = 1e-5
    h = [f32(1 + f32(f32(.03) * i)) for i in range(35)]  # 1 + .03f * i
    l = [f32(f32(.5) + f32(f32(.07) * i)) for i in range(12)]  # .5f + .07f * i
    high, low = nn.from_vector(h, [1, 5, 7, 1]), nn.from_vector(l, [1, 3, 4, 1])
    with nn.no_grad():
        hrnet.HRFuse(p, state, "fixture", [high, low], [1, 1], norm)
    for name in p.names():
        shape = list(p.at(name).shape())
        value = [0.0] * nn.shape_numel(shape)
        if name.endswith("/W"):
            value[len(value) // 2] = 1
        elif "/gamma" in name:
            value = [f32(math.sqrt(f32(1 + f32(norm.epsilon))))] * len(value)
        p.set(name, nn.from_vector(value, shape))
    high.requires_grad_(True)
    low.requires_grad_(True)
    fused = hrnet.HRFuse(p, state, "fixture", [high, low], [1, 1], norm)
    gradient = nn.grad(nn.add(nn.sum(fused[0]), nn.sum(fused[1])), [high, low])
    o0, o1 = fused[0].values(), fused[1].values()
    g0, g1 = gradient[0].values(), gradient[1].values()
    gh, gl = [1.0] * 35, [1.0] * 12
    error = 0.0
    for y in range(5):
        for x in range(7):
            sy, sx = y * .5, x * .5
            y0, x0 = int(sy), int(sx)
            y1, x1 = min(y0 + 1, 2), min(x0 + 1, 3)
            up = 0.0
            for dy in range(2):
                for dx in range(2):
                    index = (y1 if dy else y0) * 4 + (x1 if dx else x0)
                    weight = (sy - y0 if dy else 1 - (sy - y0)) * (sx - x0 if dx else 1 - (sx - x0))
                    up += weight * l[index]
                    gl[index] += weight
            error = max(error, abs(o0[y * 7 + x] - h[y * 7 + x] - up))
    for y in range(3):
        for x in range(4):
            index = y * 14 + x * 2
            gh[index] += 1
            error = max(error, abs(f32(f32(o1[y * 4 + x] - l[y * 4 + x]) - h[index])))
    for i in range(35):
        error = max(error, abs(gh[i] - g0[i]))
    for i in range(12):
        error = max(error, abs(gl[i] - g1[i]))
    check.Near(error, 0, 2e-6, "every high/low exchange value and input gradient")
    print("  odd-grid high/low fusion independent bilinear/subsample and gradient error=" + g(error), flush=True)


def config(compact):
    c = hrnet.HRNetConfig()
    if compact:
        c.classes = 4
        c.width = 8
        c.stem_width = 16
        c.stage1_width = 16
        c.stage1_blocks = 2
        c.blocks = 2
        for k in range(3):
            c.set_modules(k, 1)
    return c


def sine(n):
    """std::sin(.13f * float(i)) in single precision."""
    return [f32(math.sin(f32(f32(.13) * i))) for i in range(n)]


def model_checks():
    """Full W32 census/derivatives on 33x33, then compact 96/97 branch dimensions and gradient norms."""
    p, state = nn.ParameterStore(17), nn.ParameterStore(18)
    net = hrnet.HRNet(p, state, nn.from_vector(sine(33 * 33 * 3), [1, 33, 33, 3]), config(False), True)
    params = list(p.parameters())
    grads = nn.grad(nn.mean(nn.square(net.logits)), params)
    names = p.names()
    for k, gradient in enumerate(grads):
        check.Require(math.isfinite(nn.sum(nn.square(gradient)).item()), "finite full HRNet derivative: " + names[k])
    check.Require(net.fusion_modules == 8 and len(net.branches) == 4 and len(grads) == p.size(),
                  "full W32 eight exchange modules/four branches/all parameters")
    check.Require(tuple(p.at("head/project/W").shape()) == (1, 1, 480, 480), "full four streams concatenate480channels")
    print(f"  full HRNetV2 W32: {p.count()} parameters, {state.count()} BN state, {len(grads)} gradients "
          "(executed on a 33x33 image)", flush=True)
    del net, grads, params
    for size in (96, 97):
        p, state = nn.ParameterStore(17), nn.ParameterStore(18)
        net = hrnet.HRNet(p, state, nn.from_vector(sine(2 * size * size * 3), [2, size, size, 3]), config(True), True)
        check.Require(tuple(net.logits.shape()) == (2, size, size, 4), "HRNet logits restore odd/even input grid")
        resolution = (size + 3) // 4
        for k in range(4):
            shape = net.branches[k].shape()
            check.Require(shape[1] == resolution and shape[2] == resolution and shape[3] == 8 * (1 << k),
                          "HRNet persistent branch dimensions")
            resolution = (resolution + 1) // 2
        params = list(p.parameters())
        grads = nn.grad(nn.mean(nn.square(net.logits)), params)
        names = p.names()
        low, high = math.inf, 0.0
        for k, gradient in enumerate(grads):
            norm = math.sqrt(nn.sum(nn.square(gradient.to(nn.DType.Float64))).item())
            check.Require(math.isfinite(norm) and norm > 0, "finite nonzero HRNet derivative: " + names[k])
            low, high = min(low, norm), max(high, norm)
        print(f"  compact HRNet {size}px: {p.count()} parameters; all gradient norms [{g(low)},{g(high)}]", flush=True)


def subset(data, count):
    """The first `count` rows of a SegmentationData (quick-mode subset)."""
    result = seg.SegmentationData()
    result.count = min(data.count, count)
    result.height, result.width, result.classes = data.height, data.width, data.classes
    pixels = result.count * data.height * data.width
    result.images = data.images[:3 * pixels]
    result.labels = data.labels[:pixels]
    return result


def segmentation_score(infer, data):
    """seg::SegmentationScore: inference logits in batches of 16 rows, one confusion matrix, mIoU and pixel accuracy."""
    classes = data.classes
    confusion = [0] * (classes * classes)
    with nn.no_grad():
        for start in range(0, data.count, 16):
            rows = list(range(start, min(start + 16, data.count)))
            images, labels = seg.SegmentationBatch(data, rows)
            counts = seg.SegmentationConfusion(infer(images), labels, classes)
            confusion = [a + b for a, b in zip(confusion, counts)]
    result = seg.SegmentationMetrics()
    result.confusion = confusion
    result.mean_iou = seg.ConfusionMeanIou(confusion, classes)
    result.pixel_accuracy = sum(confusion[c * classes + c] for c in range(classes)) / sum(confusion)
    return result


def segmentation_preview(infer, data, filename):
    """seg::SegmentationPreview: image | truth | prediction rows of the first four images."""
    with nn.no_grad():
        images, _labels = seg.SegmentationBatch(data, list(range(min(4, data.count))))
        seg.SegmentationPreviewLogits(infer(images), data, filename)


def train_segmentation(training_cfg, builder, source_fit, source_validation, source_test, explicit_epochs):
    """The seg training block of the C++ main: quick-mode limits, balanced class weights, Adam on
    weighted pixel CE plus kernel L2, validation-mIoU checkpoint selection, restore, test confusion,
    preview and the demonstration policy."""
    if not explicit_epochs:
        training_cfg.epochs = example_runtime.Limit(training_cfg.epochs, 1)
    training_cfg.batch = example_runtime.Limit(training_cfg.batch, 2)
    if example_runtime.Full():
        fit, validation, test = source_fit, source_validation, source_test
    else:
        fit, validation, test = subset(source_fit, 8), subset(source_validation, 4), subset(source_test, 4)
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

    # Inverse-square-root class frequencies of the fit masks (float arithmetic, as the C++).
    fit_labels = fit.labels
    frequencies = [1] * fit.classes
    for label in fit_labels:
        frequencies[label] += 1
    weights = [f32(1 / f32(math.sqrt(f32(f32(frequencies[c]) / f32(len(fit_labels)))))) for c in range(fit.classes)]
    class_weights = nn.from_vector(weights, [fit.classes])
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
        check.Require(os.path.normpath(os.path.abspath(training_cfg.warm_start))
                      != os.path.normpath(os.path.abspath(prefix)),
                      "segmentation warm start requires a separate output tag")
        p.load(training_cfg.warm_start + "_params.npz")
        state.load(training_cfg.warm_start + "_state.npz")
        best = segmentation_score(infer, validation).mean_iou
        best_epoch = 0
        p.save(prefix + "_params.npz")
        state.save(prefix + "_state.npz")
        history.write(f"0\tnan\t{g(best)}\n")
        history.flush()
        print(f"  warm start: {training_cfg.warm_start}; fresh Adam moments; restored validation mIoU={g(best)}",
              flush=True)
    train_seconds = 0.0
    for epoch in range(1, training_cfg.epochs + 1):
        started = time.perf_counter()
        batches.begin_epoch()
        total, seen = 0.0, 0
        while True:
            rows = list(batches.next())
            if not rows:
                break
            batch = seg.SegmentationBatch(fit, rows)
            model = builder(p, state, batch[0], True)
            loss = seg.SegmentationCrossEntropy(model.logits, batch[1], fit.classes, class_weights)
            auxiliary = model.auxiliary
            if auxiliary:
                total_loss = loss
                for output in auxiliary:
                    total_loss = nn.add(total_loss, nn.mul(
                        seg.SegmentationCrossEntropy(output, batch[1], fit.classes, class_weights),
                        model.auxiliary_weight))
                loss = nn.div(total_loss, 1.0 + model.auxiliary_weight * len(auxiliary))
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
    history.close()
    p.load(prefix + "_params.npz")
    state.load(prefix + "_state.npz")
    check.Near(segmentation_score(infer, validation).mean_iou, best, 1e-10,
               "restored validation score matches selected checkpoint")
    print(f"  restored validation mIoU={g(best)}; checkpoint agreement verified", flush=True)
    # SegmentationConfig::inspect is never set by this example.
    result = segmentation_score(infer, test)
    print(f"  selected epoch={best_epoch}; test mIoU={g(result.mean_iou)}, pixel accuracy={g(result.pixel_accuracy)}")
    print("  test confusion (truth rows, predicted columns):")
    confusion = result.confusion
    for c in range(fit.classes):
        print("   " + "".join(f" {confusion[c * fit.classes + k]}" for k in range(fit.classes)))
    segmentation_preview(infer, test, os.path.join(directory, "preview.png"))
    print(f"  checkpoint, history and image/truth/prediction preview: {directory}", flush=True)
    print(f"  training time {train_seconds:.1f} s ({train_seconds / training_cfg.epochs:.2f} s/epoch, "
          "excluding validation)")
    if example_runtime.Full():
        seg.CheckSegmentationDemonstration(training_cfg, epoch_losses, result)
    else:
        check.Require(math.isfinite(result.mean_iou) and 0 <= result.mean_iou <= 1,
                      "finite quick segmentation evaluation")
    if example_runtime.Full() and not training_cfg.accuracy:
        print(f"  short learning demonstration passed: mean loss {g(epoch_losses[0])} -> {g(epoch_losses[-1])}; "
              f"accuracy=1 enables mIoU target {g(training_cfg.minimum_test_iou)}")


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    epochs, accuracy = 3, False
    for arg in arguments:
        if arg.startswith("epochs="):
            epochs = int(arg[7:])
        elif arg in ("accuracy=0", "accuracy=1"):
            accuracy = arg == "accuracy=1"
        elif arg.startswith("threads="):
            nn.set_num_threads(int(arg[8:]))
        else:
            raise ValueError("expected epochs=N, accuracy=0|1 or threads=N")
    seg.SegmentationTrainingChecks()
    seg.SegmentationLossChecks()
    fusion_checks()
    model_checks()
    fit = seg.SegmentationShapes(600, 96, 11)
    validation = seg.SegmentationShapes(120, 96, 22)
    test = seg.SegmentationShapes(120, 96, 33)
    cfg = seg.SegmentationConfig()
    cfg.name = "hrnet_shapes"
    cfg.epochs = epochs
    cfg.accuracy = accuracy

    def builder(p, state, images, training):
        model = seg.SegmentationModel()
        model.logits = hrnet.HRNet(p, state, images, config(True), training).logits
        return model

    explicit_epochs = any(arg.startswith("epochs=") for arg in arguments)
    train_segmentation(cfg, builder, fit, validation, test, explicit_epochs)
    print("All HRNet checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
