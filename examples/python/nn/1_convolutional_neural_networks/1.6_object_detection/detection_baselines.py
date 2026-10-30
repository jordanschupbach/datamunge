# Native Python NN example. Detection by classification (sliding windows versus classical region proposals over an 11-way patch classifier); the C++ example's workflow, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.6_object_detection/detection_baselines.py
"""Detection by classification: sliding windows versus region proposals.

Port of tflow's detection_baselines. Preparation (in ~/projects/tflow_examples):
python3 scripts/fetch_datasets.py mnist --split both.

Build synthetic detection scenes (common/detection_scenes.hpp, identical to
tflow's scenes): one to four MNIST digits, 16-32 pixels, scattered on a
64x64 canvas, with tight boxes; 6000 training scenes from MNIST training
digits and 1000 test scenes from MNIST test digits. Train an 11-way patch
classifier (10 digits and background) on 28x28 crops: jittered squares
around training boxes and random squares that overlap no box by more than
IoU 0.2. Detect two ways: (1) slide square windows of 16, 22, 28 and 34
pixels with stride 4 over every scene and classify each; (2) propose
regions classically, from the connected components of the image
thresholded at 16, 32, 64, 96, 128, 160 and 200 and the merged boxes of
components lying within three pixels of each other, and classify a square
crop around each. Both keep every class scoring above 0.05, then apply
per-class NMS (IoU 0.3). Report windows or proposals per scene, proposal
recall at IoU 0.5 / 0.7, and test AP (VOC AP50, AP75, COCO AP).
Square windows rarely fit a digit's tight box, which caps their AP75.
Proposals fit isolated digits almost exactly (blank background), but digits
whose strokes touch merge into one component at every threshold, which caps
proposal recall well below one; faster_rcnn learns proposals instead.
The patch classifier trains eagerly (Adam .001, batch 128, L2 .0001 on
kernels, nn's generator with seed 17), so its numbers match TF statistically.
Options: data=PATH epochs=6 threads=N.

The components (detector::StandardScenes / CropScene / ComponentProposals /
Square / BoxIoU / ClassNms / EvaluateDetections and the std::mt19937_64
engine) are the C++ ones through SWIG (pydatamunge.nn_components); the
workflow below mirrors examples/cpp/source/nn/.../detection_baselines.cpp line by line.
"""
import struct
import sys
from array import array

from pydatamunge import nn
from pydatamunge.nn_components import check, detector, example_runtime, paths

PATCH, CLASSES, BACKGROUND = 28, 11, 10


def f32(value):
    """Round a double to the nearest float, as C++ float arithmetic does after every operation."""
    return struct.unpack("f", struct.pack("f", value))[0]


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


F05, F02, F03, F07 = f32(.05), f32(.2), f32(.3), f32(.7)


class Patches:
    """Flat float32 pixels (PATCH * PATCH per row) and the label of every patch."""

    def __init__(self):
        self.pixels = array("f")
        self.labels = []


def make_patches(scenes, first, last, seed):
    """Three jittered squares per ground-truth box plus up to three background squares per box
    (IoU <= 0.2 with every box), cropped to 28x28; std::mt19937_64(seed) draws in the C++ order."""
    p = Patches()
    random = detector.Mt19937_64(seed)
    all_boxes, all_labels = scenes.boxes, scenes.labels

    def unit():
        return random.uniform_float(0, 1)

    def add(row, box, label):
        p.pixels.extend(detector.CropScene(scenes, row, box, PATCH).values())
        p.labels.append(label)

    for row in range(first, last):
        boxes, labels = all_boxes[row], all_labels[row]
        for k in range(len(boxes)):
            for _copy in range(3):
                b = boxes[k]
                side = f32(max(f32(b[2] - b[0]), f32(b[3] - b[1])) * f32(1 + f32(F03 * unit())))
                cy = f32(f32(f32(b[0] + b[2]) / 2) + f32(f32(f32(unit() - .5) * F02) * side))
                cx = f32(f32(f32(b[1] + b[3]) / 2) + f32(f32(f32(unit() - .5) * F02) * side))
                half = f32(side / 2)
                add(row, (f32(cy - half), f32(cx - half), f32(cy + half), f32(cx + half)), labels[k])
        copy = tries = 0
        while copy < 3 * len(boxes) and tries < 200:
            tries += 1
            side = f32(12 + f32(28 * unit()))
            y = f32(unit() * f32(scenes.height - side))
            x = f32(unit() * f32(scenes.width - side))
            window = (y, x, f32(y + side), f32(x + side))
            clear = True
            for b in boxes:
                if detector.BoxIoU(window, b) > F02:
                    clear = False
            if clear:
                add(row, window, BACKGROUND)
                copy += 1
    return p


def patch_logits(p, x):
    """conv(32)/pool, conv(64)/pool, fc1 128 and an 11-way output."""
    cfg = nn.ConvLayerOptions()
    cfg.init = nn.init.he_uniform()
    cfg.activation = nn.Activation.Relu
    h = nn.max_pool2d(nn.conv2d(p, "conv1", nn.reshape(x, [-1, PATCH, PATCH, 1]), 32, 3, cfg), 2, 2, nn.Padding.Same)
    h = nn.max_pool2d(nn.conv2d(p, "conv2", h, 64, 3, cfg), 2, 2, nn.Padding.Same)
    f = nn.dense(p, "fc1", nn.reshape(h, [-1, 7 * 7 * 64]), 128, nn.Activation.Relu)
    return nn.dense(p, "output", f, CLASSES)


def batch(p, rows):
    """Patch rows as an [n, 784] tensor and their labels."""
    x, labels = array("f"), []
    for r in rows:
        x.extend(p.pixels[r * PATCH * PATCH:(r + 1) * PATCH * PATCH])
        labels.append(p.labels[r])
    return nn.from_vector(x, [len(rows), PATCH * PATCH]), labels


def classify(model, scenes, crops):
    """Class probabilities [boxes x 11] of square crops around boxes, in chunks of 4096."""
    result = []
    with nn.no_grad():
        for start in range(0, len(crops), 4096):
            chunk = crops[start:start + 4096]
            x = array("f")
            for row, box in chunk:
                x.extend(detector.CropScene(scenes, row, box, PATCH).values())
            probabilities = nn.softmax(patch_logits(model, nn.from_vector(x, [len(chunk), PATCH * PATCH])), -1).values()
            for i in range(len(chunk)):
                result.append(probabilities[i * CLASSES:(i + 1) * CLASSES])
    return result


def detect(crops, reported, probabilities):
    """Every digit class scoring above 0.05, then per-class NMS at IoU 0.3."""
    dets = detector.DetectionVector()
    for i, (row, _box) in enumerate(crops):
        for c in range(BACKGROUND):
            if probabilities[i][c] > F05:
                d = detector.Detection()
                d.image, d.label, d.score, d.box = row, c, probabilities[i][c], reported[i]
                dets.push_back(d)
    return detector.ClassNms(dets, .3)


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    directory = paths.Dataset("mnist")
    epochs = 6
    for argument in arguments:
        key, separator, value = argument.partition("=")
        if not separator:
            raise ValueError("expected key=value option")
        if key == "data":
            directory = value
        elif key == "epochs":
            epochs = int(value)
        elif key == "threads":
            nn.set_num_threads(int(value))
        else:
            raise ValueError("unknown option " + key)
    epochs = example_runtime.Limit(epochs, 1)
    example_runtime.Describe()
    train, test = detector.StandardScenes(str(directory))
    if not example_runtime.Full():
        test = detector.CopyDetectionRows(test, list(range(min(test.size(), 8))))
    objects = sum(len(b) for b in test.boxes)
    print(f"  scenes: {train.size()} training ({example_runtime.Limit(5500, 16)} for patches, "
          f"{example_runtime.Limit(500, 16)} validation), {test.size()} test with {objects} digits", flush=True)
    # Patch classifier.
    fit = make_patches(train, 0, example_runtime.Limit(5500, 16), 3)
    validation = make_patches(train, 5500, 6000 if example_runtime.Full() else 5516, 4)
    model = nn.ParameterStore(17)
    with nn.no_grad():
        patch_logits(model, nn.zeros([1, PATCH * PATCH]))
    kernels = [p for p in model.parameters() if p.dim() >= 2]
    adam = nn.OptimizerConfig()
    adam.kind = nn.OptimizerKind.Adam
    optimizer = nn.Optimizer(model, .001, adam)
    order = nn.BatchOrder(len(fit.labels), example_runtime.Limit(128, 16), 29)
    for epoch in range(1, epochs + 1):
        order.begin_epoch()
        while True:
            rows = list(order.next())
            if not rows:
                break
            x, labels = batch(fit, rows)
            loss = nn.sparse_softmax_cross_entropy(patch_logits(model, x), nn.from_vector_int64(labels))
            penalty = nn.sum(nn.square(kernels[0]))
            for k in kernels[1:]:
                penalty = nn.add(penalty, nn.sum(nn.square(k)))
            optimizer.minimize(nn.add(loss, nn.mul(penalty, .5 * .0001)))
        with nn.no_grad():
            x, labels = batch(validation, range(len(validation.labels)))
            predicted = nn.argmax(patch_logits(model, x), -1).values()
        correct = sum(int(predicted[i]) == labels[i] for i in range(len(labels)))
        print(f"  patch classifier epoch {epoch}: {len(fit.labels)} training patches, validation accuracy "
              f"{g(correct / len(labels))} on {len(validation.labels)}", flush=True)
    # Sliding windows.
    windows = []
    for row in range(test.size()):
        for side in (16, 22, 28, 34):
            for y in range(0, test.height - side + 1, 4):
                for x in range(0, test.width - side + 1, 4):
                    windows.append((row, (float(y), float(x), float(y + side), float(x + side))))
    window_boxes = [w[1] for w in windows]
    sliding = detect(windows, window_boxes, classify(model, test, windows))
    rows = list(range(test.size()))
    sliding_ap = detector.EvaluateDetections(sliding, test, rows, 10)
    # Region proposals.
    crops, proposals = [], []
    covered50 = covered70 = 0
    test_boxes = test.boxes
    for row in range(test.size()):
        boxes = detector.ComponentProposals(test, row, [16, 32, 64, 96, 128, 160, 200], 3)
        for b in boxes:
            crops.append((row, detector.Square(b, 1.15)))
            proposals.append(b)
        for truth in test_boxes[row]:
            best = 0.0
            for b in boxes:
                best = max(best, detector.BoxIoU(b, truth))
            covered50 += best >= .5
            covered70 += best >= F07
    proposed = detect(crops, proposals, classify(model, test, crops))
    proposal_ap = detector.EvaluateDetections(proposed, test, rows, 10)
    print(f"  sliding windows: {len(windows) // test.size()} per scene; test {detector.ApSummary(sliding_ap)}\n"
          f"  region proposals: {len(crops) / test.size():.1f} per scene, recall {covered50 / objects:.3f} at IoU 0.5 and "
          f"{covered70 / objects:.3f} at 0.7; test {detector.ApSummary(proposal_ap)}", flush=True)
    check.TrainingQuality(proposal_ap.ap50 > .6 and proposal_ap.ap50 > sliding_ap.ap50 and proposal_ap.ap75 > sliding_ap.ap75
                          and sliding_ap.ap50 > .2, "detection baselines")
    print("All detection baseline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
