# Native Python NN example. Faster R-CNN on MNIST digit scenes: anchor-matching check, joint RPN/RoI-head training, RPN recall and test AP; the C++ example's loop, written out in Python.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/1_convolutional_neural_networks/1.6_object_detection/faster_rcnn.py
"""Faster R-CNN: a region proposal network shares the backbone with the Fast R-CNN head.

Port of tflow's faster_rcnn. Preparation (in ~/projects/tflow_examples):
python3 scripts/fetch_datasets.py mnist --split both.

On the synthetic digit scenes, replace the classical proposals of
fast_rcnn with a learned region proposal network on the shared stride-4
feature map: a 3x3 convolution (128 channels, ReLU) and two 1x1 layers
scoring and refining 9 anchors per position (sizes 12, 20 and 30 pixels,
height/width 0.5, 1 and 2; 2304 anchors). Anchors with IoU >= 0.7 (and each
target's best anchor) are positive and below 0.3 negative; 128 anchors per
image are sampled, at most half positive, for a sigmoid objectness loss
and smooth-L1 box loss. Proposals are the 300 best decoded anchors, reduced
by NMS at IoU 0.7 to 100 per image. Training is approximate joint training:
each step computes proposals with the current RPN, samples 64 RoIs from
them (a quarter foreground) and minimizes the RPN and head losses together.
Check anchor matching on a fixture, then train with 5500 / 500 / 1000
scenes, two images per step, Adam(.001), kernel L2(.0001) and the best
validation AP50 checkpoint; report RPN proposal recall and test AP.
https://arxiv.org/abs/1506.01497

Eager port: each step runs the backbone and RPN once and takes the
(detached) proposals from that same forward pass; tflow runs a separate
proposal pass with the same, not yet updated, weights. Proposal recall is
measured on the test scenes only (tflow's counter also pools the 500
restored-validation scenes into its printed recall). Weights come from nn's
generator, so trained numbers match TF statistically.
Checkpoint (.npz) and history: output/nn/models/faster_rcnn_digits/.
Options: data=PATH epochs=10 threads=N.

The components (detector::StrideFourBackbone, BuildRpn, RoiHead, SampleRois,
MakeRpnTargets, ComputeProposals, the losses, the digit scenes and the AP
evaluation) are the C++ ones through SWIG (pydatamunge.nn_components); the
workflow below mirrors examples/cpp/source/nn/.../faster_rcnn.cpp line by line.
"""
import math
import os
import struct
import sys

from pydatamunge import nn
from pydatamunge.nn_components import check, detector, example_runtime, paths

K_CLASSES, K_BACKGROUND, K_STRIDE, K_FEATURE, K_ANCHORS_PER_CELL, K_ROIS = 11, 10, 4, 16, 9, 64
K_ANCHORS = detector.AnchorGrid(K_FEATURE, K_FEATURE, K_STRIDE, [12, 20, 30], [.5, 1, 2])
K_ANCHOR_COUNT = len(K_ANCHORS)


def g(value):
    """C++ default ostream formatting (six significant digits)."""
    return f"{value:.6g}"


def f32(value):
    """The float literal the C++ compares against (single precision)."""
    return struct.unpack("f", struct.pack("f", value))[0]


def matching_checks():
    # A target equal to an anchor is matched to it; distant anchors are negatives.
    index = (5 * K_FEATURE + 7) * K_ANCHORS_PER_CELL + 4
    target = K_ANCHORS[index]
    match = detector.MatchAnchors(K_ANCHORS, [target], .7, .3)
    positives = sum(1 for m in match if m == 0)
    negatives = sum(1 for m in match if m == -1)
    ignored = sum(1 for m in match if m == -2)
    check.Require(match[index] == 0 and positives >= 1 and positives + negatives + ignored == K_ANCHOR_COUNT,
                  "anchor matching")
    delta = detector.EncodeBox(target, target)
    check.Near(abs(delta[0]) + abs(delta[1]) + abs(delta[2]) + abs(delta[3]), 0, 1e-6,
               "zero deltas for a matching anchor")
    print(f"  anchors: {K_ANCHOR_COUNT} (16x16 cells x 3 sizes x 3 ratios); a target equal to one anchor has "
          f"{positives} positive, {ignored} ignored and {negatives} negative anchors", flush=True)


def roi_images(scenes, rows):
    return detector.SceneImages(scenes, rows)


def forward(p, images):
    """Shared stride-4 backbone and the RPN over its single feature level."""
    features = detector.StrideFourBackbone(p, images)
    level = detector.RpnLevel()
    level.features, level.channels, level.cells, level.anchors_per_cell = features, 64, K_FEATURE, K_ANCHORS_PER_CELL
    levels = detector.RpnLevelVector()
    levels.push_back(level)
    return features, detector.BuildRpn(p, levels)


def head(p):
    return lambda features, rois, indices: detector.RoiHead(p, features, rois, indices, K_CLASSES)


def detect_rois(head_fn, features, rows, boxes, classes, stride, feature_size):
    """detector::DetectRois (two_stage.hpp): classify and refine boxes [(batch position, pixel box)]
    of the given rows; keeps class scores above 0.05."""
    dets = []
    if not boxes:
        return dets
    rois = [detector.FeatureBox(box, stride, feature_size) for _index, box in boxes]
    indices = [index for index, _box in boxes]
    out = head_fn(features, detector.RoiTensor(rois), detector.IndexTensor(indices))
    probabilities = out.probabilities.values()
    deltas = out.deltas.values()
    threshold = f32(.05)
    for k, (index, box) in enumerate(boxes):
        for c in range(classes - 1):
            score = probabilities[k * classes + c]
            if score <= threshold:
                continue
            at = (k * (classes - 1) + c) * 4
            d = detector.Detection()
            d.image, d.label, d.score = rows[index], c, score
            d.box = detector.DecodeBox(box, deltas[at:at + 4])
            dets.append(d)
    return dets


def main(arguments=None):
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    directory = paths.Dataset("mnist")
    cfg = detector.DetectorTrainingConfig()
    cfg.name = "faster_rcnn_digits"
    cfg.epochs = 10
    cfg.batch = 2
    cfg.validation = 500
    cfg.minimum_test_ap50 = .7
    for argument in arguments:
        key, separator, value = argument.partition("=")
        if not separator:
            raise ValueError("expected key=value option")
        if key == "data":
            directory = value
        elif key == "epochs":
            cfg.epochs = int(value)
        elif key == "threads":
            nn.set_num_threads(int(value))
        else:
            raise ValueError("unknown option " + key)
    matching_checks()
    train, test = detector.StandardScenes(str(directory))
    random = detector.Mt19937_64(7)

    def hooks_build(p, _state):
        features, _rpn = forward(p, nn.zeros([1, 64, 64, 1]))
        detector.RoiHead(p, features, detector.RoiTensor([[0, 0, 1, 1]]), detector.IndexTensor([0]), K_CLASSES)

    def hooks_loss(p, _state, scenes, rows):
        features, rpn = forward(p, roi_images(scenes, rows))
        with nn.no_grad():
            proposals = detector.ComputeProposals(rpn, K_ANCHORS, scenes.height, scenes.width)
        batch = detector.RoiBatch()
        for i, row in enumerate(rows):
            detector.SampleRois(scenes, row, i, proposals[i], K_ROIS, K_BACKGROUND, K_STRIDE, K_FEATURE, random, batch)
        targets = detector.MakeRpnTargets(K_ANCHORS, scenes, rows, random)
        out = detector.RoiHead(p, features, detector.RoiTensor(batch.rois), detector.IndexTensor(batch.indices), K_CLASSES)
        return nn.add(detector.RpnLoss(rpn, targets.labels, targets.weights, targets.targets, targets.positives),
                      detector.RoiLoss(out, batch, K_CLASSES))

    # The trainer evaluates validation once per epoch, again after restoring
    # the best checkpoint, then the test rows: measure RPN recall on that last call.
    test_scenes = test
    recall = {"objects": 0, "recall50": 0.0, "recall70": 0.0}
    half, seventy = f32(.5), f32(.7)

    def hooks_detect(p, _state, scenes, rows):
        with nn.no_grad():
            final_call = scenes is test_scenes
            dets = detector.DetectionVector()
            for start in range(0, len(rows), 32):
                chunk = rows[start:min(len(rows), start + 32)]
                features, rpn = forward(p, roi_images(scenes, chunk))
                proposals = detector.ComputeProposals(rpn, K_ANCHORS, scenes.height, scenes.width)
                boxes = []
                for i in range(len(chunk)):
                    for b in proposals[i]:
                        boxes.append((i, b))
                    if final_call:
                        for truth in scenes.boxes[chunk[i]]:
                            best = 0.0
                            for b in proposals[i]:
                                best = max(best, detector.BoxIoU(b, truth))
                            recall["objects"] += 1
                            recall["recall50"] += best >= half
                            recall["recall70"] += best >= seventy
                found = detect_rois(head(p), features, chunk, boxes, K_CLASSES, K_STRIDE, K_FEATURE)
                for d in found:
                    dets.push_back(d)
            return detector.ClassNms(dets, .3)

    def train_detector():
        training_cfg = cfg
        explicit_epochs = any(option.startswith("epochs=") for option in arguments)
        if not explicit_epochs:
            training_cfg.epochs = example_runtime.Limit(training_cfg.epochs, 1)
        training_cfg.batch = example_runtime.Limit(training_cfg.batch, 2)
        training_cfg.validation = example_runtime.Limit(training_cfg.validation, 8)
        if not example_runtime.Full():
            training_cfg.fit = min(training_cfg.fit, 8) if training_cfg.fit else 8
        example_runtime.Describe()
        check.Require(0 < training_cfg.validation < train.size() and training_cfg.epochs > 0 and training_cfg.batch > 0,
                      "valid detector training configuration")
        check.Require(not example_runtime.Full() or training_cfg.accuracy or training_cfg.epochs >= 2,
                      "short learning demonstration needs at least two epochs")
        fit_count = (min(training_cfg.fit, train.size() - training_cfg.validation) if training_cfg.fit > 0
                     else train.size() - training_cfg.validation)
        fit = list(range(fit_count))
        validation = list(range(train.size() - training_cfg.validation, train.size()))
        test_rows = list(range(example_runtime.Limit(test.size(), 8)))
        params, state = nn.ParameterStore(17), nn.ParameterStore(17)
        with nn.no_grad():
            hooks_build(params, state)
        kernels = [p for p in params.parameters() if p.dim() >= 2]
        directory = detector.TrainingDirectory(training_cfg.name)
        os.makedirs(directory, exist_ok=True)
        prefix = os.path.join(directory, "best")
        if training_cfg.warm_start:
            check.Require(os.path.normpath(os.path.abspath(training_cfg.warm_start)) !=
                          os.path.normpath(os.path.abspath(prefix)),
                          "warm start requires a separate output directory/run tag")
            detector.RestoreCheckpoint(params, state, training_cfg.warm_start)
            print(f"  warm start: {training_cfg.warm_start}; fresh Adam moments", flush=True)
        optimizer_config = nn.OptimizerConfig()
        optimizer_config.kind = nn.OptimizerKind.Adam
        optimizer = nn.Optimizer(params, training_cfg.learning_rate, optimizer_config)
        history = open(os.path.join(directory, "history.tsv"), "w")
        history.write("epoch\ttraining_loss\tvalidation_ap50\tvalidation_coco_ap\n")
        print(f"== {training_cfg.name}: {fit_count} fit / {training_cfg.validation} validation / {len(test_rows)} "
              f"test scenes ({train.height}x{train.width})\n  parameters={params.count()}; batch={training_cfg.batch}; "
              f"lr={g(training_cfg.learning_rate)}", flush=True)
        batches = nn.BatchOrder(fit_count, training_cfg.batch, 29)
        best, best_epoch = -1.0, -1
        epoch_losses = []
        if training_cfg.warm_start:
            baseline = detector.EvaluateDetections(hooks_detect(params, state, train, validation), train, validation,
                                                   training_cfg.classes)
            best, best_epoch = baseline.ap50, 0
            detector.SaveCheckpoint(params, state, prefix)
            history.write(f"0\tnan\t{g(baseline.ap50)}\t{g(baseline.coco)}\n")
            history.flush()
            print(f"  restored baseline validation {detector.ApSummary(baseline)}", flush=True)
        for epoch in range(1, training_cfg.epochs + 1):
            batches.begin_epoch()
            total, seen = 0.0, 0
            while True:
                picked = list(batches.next())
                if not picked:
                    break
                rows = [fit[k] for k in picked]
                objective = hooks_loss(params, state, train, rows)
                batch_loss = objective.item()
                check.Require(math.isfinite(batch_loss), "finite detector batch loss")
                if training_cfg.l2 > 0 and kernels:
                    penalty = nn.sum(nn.square(kernels[0]))
                    for kernel in kernels[1:]:
                        penalty = nn.add(penalty, nn.sum(nn.square(kernel)))
                    objective = nn.add(objective, nn.mul(penalty, 0.5 * training_cfg.l2))
                optimizer.minimize(objective)
                total += batch_loss * len(rows)
                seen += len(rows)
            epoch_losses.append(total / seen)
            check.Require(math.isfinite(epoch_losses[-1]), "finite detector epoch mean")
            result = detector.EvaluateDetections(hooks_detect(params, state, train, validation), train, validation,
                                                 training_cfg.classes)
            check.Require(math.isfinite(result.ap50), "finite validation AP50")
            if result.ap50 > best:
                best, best_epoch = result.ap50, epoch
                detector.SaveCheckpoint(params, state, prefix)
            history.write(f"{epoch}\t{g(total / seen)}\t{g(result.ap50)}\t{g(result.coco)}\n")
            history.flush()
            print(f"  epoch {epoch}: training loss={g(total / seen)}, validation {detector.ApSummary(result)}", flush=True)
        history.close()
        detector.RestoreCheckpoint(params, state, prefix)
        restored = detector.EvaluateDetections(hooks_detect(params, state, train, validation), train, validation,
                                               training_cfg.classes)
        check.Near(restored.ap50, best, 1e-6, "restored checkpoint reproduces selected validation AP50")
        print(f"  restored validation AP50={g(restored.ap50)}; checkpoint agreement verified", flush=True)
        result = detector.EvaluateDetections(hooks_detect(params, state, test, test_rows), test, test_rows,
                                             training_cfg.classes)
        print(f"  selected epoch={best_epoch}; test {detector.ApSummary(result)}\n  per-class test AP50:"
              + "".join(f" {ap:.3f}" for ap in result.per_class50)
              + f"\n  best checkpoint and history: {directory}", flush=True)
        if example_runtime.Full():
            detector.CheckDetectorDemonstration(training_cfg, epoch_losses, result)
        else:
            check.Require(math.isfinite(result.ap50), "finite quick detector evaluation")
        if example_runtime.Full() and not training_cfg.accuracy:
            print(f"  short learning demonstration passed: mean loss {g(epoch_losses[0])} -> {g(epoch_losses[-1])}; "
                  f"accuracy benchmark not requested (accuracy=1 enables AP50 gate {g(training_cfg.minimum_test_ap50)})",
                  flush=True)
        return result

    result = train_detector()
    objects = recall["objects"]
    print(f"  RPN proposals (at most 100 per test scene): recall {recall['recall50'] / objects:.3f} at IoU 0.5 and "
          f"{recall['recall70'] / objects:.3f} at 0.7; test AP50 {result.ap50:.4f}", flush=True)
    check.Require(objects > 0 and math.isfinite(recall["recall50"] / objects),
                  "finite RPN proposal recall on evaluated test objects")
    check.TrainingQuality(recall["recall50"] / objects > .9, "RPN proposal recall")
    print("All Faster R-CNN checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
