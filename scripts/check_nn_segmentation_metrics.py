"""Recompute scores from the saved predictions in the NN segmentation reports."""

import argparse
from pathlib import Path

import numpy as np


def masks(path):
    with np.load(path) as file:
        p, g, pl, gl, pi, gi, scores, metrics = (
            file[k]
            for k in (
                "predicted_masks",
                "truth_masks",
                "predicted_labels",
                "truth_labels",
                "predicted_images",
                "truth_images",
                "scores",
                "metrics",
            )
        )
    assert p.dtype == g.dtype == bool and np.isfinite(scores).all()
    p = p.reshape(len(p), -1).astype(float)
    g = g.reshape(len(g), -1).astype(float)
    intersection = p @ g.T
    iou = intersection / np.maximum(1, p.sum(1)[:, None] + g.sum(1) - intersection)

    def ap(threshold):
        values = []
        for label in np.unique(gl):
            truth_count = np.count_nonzero(gl == label)
            used = set()
            hit = []
            for index in np.argsort(-scores, kind="stable"):
                if pl[index] != label:
                    continue
                valid = np.where((gl == label) & (gi == pi[index]), iou[index], -1)
                best = valid.argmax()
                detected = valid[best] >= threshold and best not in used
                if detected:
                    used.add(best)
                hit.append(detected)
            if hit:
                tp = np.cumsum(hit)
                precision = tp / np.arange(1, len(tp) + 1)
                envelope = np.maximum.accumulate(precision[::-1])[::-1]
                values.append(np.dot(hit, envelope) / truth_count)
            else:
                values.append(0.0)
        return np.mean(values)

    actual = (
        [ap(0.5)]
        if len(metrics) == 1
        else [ap(0.5), ap(0.75), np.mean([ap(0.5 + 0.05 * k) for k in range(10)])]
    )
    np.testing.assert_allclose(metrics, actual, atol=2e-7, rtol=0)
    print(path.stem, actual)


def pixels(path):
    with np.load(path) as file:
        truth, pred, counts, metrics = (
            file[k] for k in ("truth", "prediction", "counts", "metrics")
        )
    k = len(counts)
    assert truth.shape == pred.shape
    valid = (truth >= 0) & (truth < k)
    assert ((pred >= 0) & (pred < k)).all()
    matrix = np.bincount(k * truth[valid] + pred[valid], minlength=k * k).reshape(k, k)
    np.testing.assert_array_equal(matrix, counts)
    accuracy = matrix.trace() / matrix.sum()
    diag = matrix.diagonal()
    union = matrix.sum(0) + matrix.sum(1) - diag
    if "voxels" in path.stem:
        t = truth.astype(bool).reshape(len(truth), -1)
        p = pred.astype(bool).reshape(len(pred), -1)
        tp = np.sum(p & t, axis=1)
        paired = p.sum(1) + t.sum(1)
        dice = np.where(paired > 0, 2 * tp / np.maximum(paired, 1), 1)
        iou = np.where(paired > 0, tp / np.maximum(paired - tp, 1), 1)
        actual = [dice.mean(), iou.mean(), accuracy]
    else:
        actual = [accuracy, np.mean(diag[union > 0] / union[union > 0])]
    np.testing.assert_allclose(metrics, actual, atol=2e-7, rtol=0, err_msg=path.stem)
    print(path.stem, truth.size, actual)


REPORTS = {
    "deeplab_v1_v2": (2, False),
    "deeplab_v3_plus": (2, False),
    "fcn": (3, False),
    "hrnet": (1, False),
    "mask_rcnn": (0, True),
    "panoptic_fpn": (1, True),
    "pspnet": (1, False),
    "segnet": (1, False),
    "unet": (1, False),
    "unet_variants": (2, False),
    "volumetric_segmentation": (2, False),
    "yolact": (0, True),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "reports",
        nargs="*",
        metavar="REPORT",
        help="Report basenames. Omit to check the whole chapter.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "docs/mepml/nn/1_convolutional_neural_networks/1.7_segmentation/build",
    )
    args = parser.parse_args()
    names = args.reports or REPORTS
    for name in names:
        if name not in REPORTS:
            parser.error(f"Unknown segmentation report: {name}")
        expected_count, has_masks = REPORTS[name]
        paths = sorted(args.root.glob(f"{name}_pixels_*.npz"))
        paths += sorted(args.root.glob(f"{name}_voxels_*.npz"))
        if len(paths) != expected_count:
            raise ValueError(
                f"{name}: expected {expected_count} semantic or volume archives, found {len(paths)}"
            )
        for path in paths:
            pixels(path)
        if has_masks:
            masks(args.root / f"{name}_masks.npz")


if __name__ == "__main__":
    main()
