# Native Python NN example. UNet++ and attention U-Net gate/parameter checks and compact procedural-shape segmentation; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.7_segmentation/unet_variants.py
"""UNet++ dense nested skip paths and additive attention U-Net.
https://arxiv.org/abs/1807.10165
https://arxiv.org/abs/1804.03999

Port of tflow's unet_variants. Check additive gate values and analytic
skip/gate input derivatives with positive/negative projections and
saturated sigmoid. Build full base 32 UNet++ (fifteen nodes, four
deep-supervision heads) and base 64 attention U-Net (four additive gates);
verify all parameter connections. Check every compact derivative and dense
concatenation widths. Nested upsampling is bilinear with aligned corners and
actual skip size; attention U-Net uses learned 2x2 transposed convolution
then skip gating. Convolution pairs/gates omit BN in this adaptation.

Train both base 8, four-level, four-class compact models on procedural shape
scenes, 600 fit/120 validation/120 test, 96 pixels, occlusion/noise and
identical photometry across classes. UNet++ averages balanced CE over four
heads; evaluation uses the deepest. Use three epochs, Adam(.001), batch 16,
L2 .0001 and best-validation mIoU. Report confusion and colored
input/truth/prediction PNGs. No biomedical benchmark or pretrained-weight
claim is made.

TF -> datamunge: TF builds the full 256x256 models as graphs only; here they
execute forward and backward on 32x32 images (same parameter census and
derivative count). The shapes data are bit-identical to TF's; weights come
from datamunge's generator, so trained numbers match TF statistically.
Options: epochs=3 model=nested|attention (default both) accuracy=0|1 threads=N
seed=N (parameter initialization, default 17) tag=NAME (output directory suffix).
Default accuracy=0 checks learning; accuracy=1 retains the .55 mIoU target.
Artifacts: output/nn/models/{nested,attention}_unet_shapes/.

The model components (unet::NestedUnet, AttentionUnet, AttentionGate, the seg::
procedural shapes, batching, balanced pixel CE, policy checks, confusion and
preview helpers) are the C++ ones through SWIG (pydatamunge.nn_components);
the checks and the training workflow below mirror
examples/cpp/source/nn/.../unet_variants.cpp line by line.
"""
import math
import os
import struct
import sys
import time

from pydatamunge import nn
from pydatamunge.nn_components import check, example_runtime, seg, unet


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def f32(value):
    """Rounds a double to float32, as storing into a std::vector<float> does."""
    return struct.unpack("f", struct.pack("f", value))[0]


def wave(shape, scale):
    """std::sin(float(scale) * float(i)): the float32 product and sinf, reproduced with the tensor ops."""
    n = math.prod(shape)
    values = nn.sin(nn.mul(nn.from_vector([float(i) for i in range(n)], [n]), scale)).values()
    return nn.from_vector(values, shape)


def gradient_norms(p, objective, nonzero, what):
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


def gate_checks():
    p = nn.ParameterStore(17)
    p.set("gate/x/W", nn.from_vector([.5, -.2], [1, 1, 2, 1]))
    p.set("gate/g/W", nn.from_vector([.3, .4], [1, 1, 2, 1]))
    p.set("gate/x/b", nn.from_vector([.1], [1]))
    p.set("gate/g/b", nn.from_vector([-.1], [1]))
    p.set("gate/psi/W", nn.from_vector([.7], [1, 1, 1, 1]))
    p.set("gate/psi/b", nn.from_vector([-.2], [1]))
    a, b = [0.0] * 12, [0.0] * 12
    for i in range(6):
        a[2 * i] = f32(.2 + .1 * i)
        a[2 * i + 1] = f32(-.3 + .15 * i)
        b[2 * i] = f32(-.2 + .05 * i)
        b[2 * i + 1] = f32(.4 - .02 * i) if i else -2.0

    def run():
        """Gate forward on x = a, g = b; returns (dx, dg, coefficients, output) values."""
        x, gate = nn.from_vector(a, [1, 2, 3, 2]), nn.from_vector(b, [1, 2, 3, 2])
        x.requires_grad_(True)
        gate.requires_grad_(True)
        net = unet.AttentionGate(p, "gate", x, gate, 1)
        gradients = nn.grad(nn.sum(net.output), [x, gate])
        return gradients[0].values(), gradients[1].values(), net.coefficients.values(), net.output.values()

    grad_x, grad_g, coefficients, output = run()
    error = 0.0
    wx, wg = (.5, -.2), (.3, .4)
    for i in range(6):
        z = .5 * a[2 * i] - .2 * a[2 * i + 1] + .3 * b[2 * i] + .4 * b[2 * i + 1]
        coefficient = 1 / (1 + math.exp(-(.7 * max(z, 0.) - .2)))
        total = a[2 * i] + a[2 * i + 1]
        chain = total * coefficient * (1 - coefficient) * .7 if z > 0 else 0
        error = max(error, abs(coefficients[i] - coefficient))
        for c in range(2):
            k = 2 * i + c
            error = max(error, abs(output[k] - a[k] * coefficient))
            error = max(error, abs(grad_x[k] - (coefficient + chain * wx[c])))
            error = max(error, abs(grad_g[k] - chain * wg[c]))
    check.Near(error, 0, 2e-7, "independent additive-gate forward and chain rule")
    p.set("gate/psi/b", nn.from_vector([100.], [1]))
    saturated_x, saturated_g, _coefficients, _output = run()
    for i in range(12):
        check.Near(saturated_x[i], 1, 0, "saturated open gate identity derivative")
        check.Near(saturated_g[i], 0, 0, "saturated gate query derivative")
    print("  additive attention values/input derivatives, negative ReLU and saturated sigmoid passed; error=" + g(error))


def segmentation_model(logits, auxiliary=()):
    """seg::SegmentationModel{logits, auxiliary}."""
    model = seg.SegmentationModel()
    model.logits = logits
    model.auxiliary = list(auxiliary)
    return model


def build(p, images, nested, compact):
    if nested:
        c = unet.NestedUnetConfig()
        c.classes = 4 if compact else 2
        c.base_width = 8 if compact else 32
        net = unet.NestedUnet(p, images, c)
        return segmentation_model(net.logits, list(net.heads)[:-1])
    c = unet.AttentionUnetConfig()
    c.classes = 4 if compact else 2
    c.base_width = 8 if compact else 64
    return segmentation_model(unet.AttentionUnet(p, images, c).logits)


def objective(net):
    total = nn.mean(nn.square(net.logits))
    for output in net.auxiliary:
        total = nn.add(total, nn.mean(nn.square(output)))
    return total


def model_checks():
    for nested in (True, False):
        p = nn.ParameterStore(17)
        net = build(p, wave([1, 32, 32, 3], .13), nested, False)
        gradient_norms(p, objective(net), False, "full U-Net variant")
        check.Require(p.count() == (9156264 if nested else 31381414), "full no-BN nested/attention U-Net parameter counts")
        check.Require(p.size() == (68 if nested else 70), "full U-Net variant parameter connections (TF: 68/70)")
        if nested:
            check.Require(p.at("x0_4/conv1/W").shape()[2] == 192 and len(net.auxiliary) == 3,
                          "deepest dense node concatenates all four prior nodes and two-width lower feature")
        print(f"  full {'UNet++' if nested else 'attention U-Net'}: {p.count()} parameters, all {p.size()} "
              "derivatives (executed on 32x32)")
    for nested in (True, False):
        p = nn.ParameterStore(17)
        net = build(p, wave([2, 96, 96, 3], .13), nested, True)
        check.Require(tuple(net.logits.shape()) == (2, 96, 96, 4), "full-resolution variant logits")
        norms = gradient_norms(p, objective(net), True, "variant")
        print(f"  compact {'UNet++' if nested else 'attention U-Net'}: {p.count()} parameters, every derivative norm "
              f"[{g(norms[0])},{g(norms[1])}]")


# ---- seg:: helpers that take a LogitsFn callback (not bound), written out over the bound pieces ----

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
    weights, Adam on balanced pixel CE (averaged with the auxiliary heads) plus L2, validation-mIoU
    checkpoint selection, test confusion and the preview PNG."""
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
    epochs, accuracy, models, seed, tag = 3, False, [True, False], 17, ""
    for arg in arguments:
        if arg.startswith("epochs="):
            epochs = int(arg[7:])
        elif arg in ("accuracy=0", "accuracy=1"):
            accuracy = arg == "accuracy=1"
        elif arg == "model=nested":
            models = [True]
        elif arg == "model=attention":
            models = [False]
        elif arg.startswith("seed="):
            seed = int(arg[5:])
        elif arg.startswith("tag="):
            tag = "_" + arg[4:]
        elif arg.startswith("threads="):
            nn.set_num_threads(int(arg[8:]))
        else:
            raise ValueError("expected epochs=N, model=nested|attention, accuracy=0|1, seed=N, tag=NAME or threads=N")
    seg.SegmentationTrainingChecks()
    seg.SegmentationLossChecks()
    gate_checks()
    model_checks()
    fit = seg.SegmentationShapes(600, 96, 11)
    validation = seg.SegmentationShapes(120, 96, 22)
    test = seg.SegmentationShapes(120, 96, 33)
    explicit_epochs = any(arg.startswith("epochs=") for arg in arguments)
    for nested in models:
        cfg = seg.SegmentationConfig()
        cfg.name = ("nested_unet_shapes" if nested else "attention_unet_shapes") + tag
        cfg.seed = seed
        cfg.epochs = epochs
        cfg.accuracy = accuracy

        def builder(p, state, images, training, nested=nested):
            return build(p, images, nested, True)

        run_segmentation(builder, cfg, fit, validation, test, explicit_epochs)
    print("All UNet++ and attention U-Net checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
