# Native Python NN example. Flip, crop, color jitter, cutout, mixup and CutMix on NHWC batches.
# CMD: nix develop .#python --command env -u PYTHONPATH build/venv/pydatamunge-run/bin/python examples/python/nn/0_foundations/0.4_data_loading_and_preprocessing/data_augmentation.py
"""Flip, crop, color jitter, cutout, mixup and CutMix on NHWC batches.

Port of tflow's data_augmentation. Check exact pixel positions, unchanged
evaluation inputs, clipping, soft labels and area-correct CutMix labels on
small known NHWC batches. Crop coordinates come from a host RNG that is
reproducible for a fixed seed and toolchain. Batch-shared transforms keep
this example compact.

Mixup blends images and probability-vector labels by the same Beta-sampled
weight (https://arxiv.org/abs/1710.09412). CutMix replaces a rectangle and
weights labels by its realized clipped area (https://arxiv.org/abs/1905.04899).
Evaluation bypasses augmentation. This example checks transformations and
label consistency, not a claim about classification accuracy improvements.

Differences from TF: the transforms are nn.augment functions called
eagerly. Invalid parameters raise at the call instead of failing a graph
Assert. Beta weights are drawn from nn.Generator (seed 7), so the 20 sampled
weights differ from TF's mt19937_64 draws; the crop coordinates use
mt19937_64(17) through nn.Mt19937_64 and reproduce the C++ (libc++)
std::uniform_int_distribution(0, 2): the low two bits of each engine output,
rejecting 3.
"""
import struct
import sys

from pydatamunge import nn
from pydatamunge.nn_workflows import helper

checks = helper("checks")
aug = nn.augment


def f32(value):
    return struct.unpack("f", struct.pack("f", value))[0]


class View:
    """[N, H, W, C] element accessor for a contiguous tensor."""

    def __init__(self, tensor):
        self.v = tensor.values()
        self.h, self.w, self.c = tensor.size(1), tensor.size(2), tensor.size(3)

    def __call__(self, n, y, x, ch):
        return self.v[((n * self.h + y) * self.w + x) * self.c + ch]


def uniform_int_0_2(engine):
    """libc++ uniform_int_distribution<int>(0, 2) over a 64-bit engine."""
    while True:
        value = engine.modulo(4)  # engine() & 3
        if value < 3:
            return value


def transformations():
    pv = [f32(i / 120) for i in range(96)]
    image = nn.from_vector(pv, [2, 4, 4, 3])
    pixels = View(image)

    def flip(x, enabled):
        return aug.flip_left_right(x) if enabled else x

    flipped = View(flip(image, True))
    twice = View(flip(flip(image, True), True))
    bypass = View(flip(image, False))
    crop = View(aug.crop(image, 1, 1, 2, 2))
    identity = View(aug.color_jitter(image, 0, 1, 1, 0))
    bright = aug.color_jitter(nn.full([1, 2, 2, 3], 0.8), 0.3, 1, 1, 0).values()
    green = aug.color_jitter(nn.from_vector([1, 0, 0], [1, 1, 1, 3]), 0, 1, 1, 1.0 / 3).values()
    rect = nn.Rect()
    rect.top, rect.left, rect.height, rect.width = 1, 1, 2, 2
    cutout = View(aug.cutout(image, rect))
    for n in range(2):
        for y in range(4):
            for x in range(4):
                for c in range(3):
                    original = pixels(n, y, x, c)
                    checks.near(flipped(n, y, x, c), pixels(n, y, 3 - x, c), 0, "horizontal flip axis")
                    checks.near(twice(n, y, x, c), original, 0, "double flip identity")
                    checks.near(bypass(n, y, x, c), original, 0, "evaluation bypass")
                    checks.near(identity(n, y, x, c), original, 2e-7, "identity color jitter")
                    erased = 1 <= y < 3 and 1 <= x < 3
                    checks.near(cutout(n, y, x, c), 0 if erased else original, 0, "cutout region")
    for n in range(2):
        for y in range(2):
            for x in range(2):
                for c in range(3):
                    checks.near(crop(n, y, x, c), pixels(n, y + 1, x + 1, c), 0, "crop offset")
    for v in bright:
        checks.near(v, 1, 0, "brightness clamp")
    for c in range(3):
        checks.near(green[c], 1 if c == 1 else 0, 2e-7, "hue red to green")
    rng, repeat = nn.Mt19937_64(17), nn.Mt19937_64(17)
    for _ in range(12):
        y = uniform_int_0_2(rng)
        x = uniform_int_0_2(rng)
        checks.require(y == uniform_int_0_2(repeat) and x == uniform_int_0_2(repeat), "seeded crop coordinates")
        out = View(aug.crop(image, y, x, 2, 2))
        checks.near(out(0, 0, 0, 0), pixels(0, y, x, 0), 0, "runtime crop coordinates")
    print("  flip, evaluation bypass, crop coordinates, color jitter and cutout passed")
    print("  12 seeded random crops reused one crop function")


def mixing():
    pv = [0.2 if i < 48 else 0.8 for i in range(96)]
    image = nn.from_vector(pv, [2, 4, 4, 3])
    labels = nn.from_vector([1, 0, 0, 1], [2, 2])
    partner = [1, 0]
    mixup = aug.mixup(image, labels, partner, 0.25)
    box = nn.Rect()
    box.top, box.left, box.height, box.width = -1, 2, 3, 4
    cutmix = aug.cutmix(image, labels, partner, box)
    mixed_images, mixed_labels = mixup.images.values(), mixup.labels.values()
    cut_labels = cutmix.labels.values()
    checks.near(mixed_images[0], 0.65, 1e-7, "mixup image interpolation")
    checks.near(mixed_labels[0], 0.25, 0, "mixup same label coefficient")
    checks.near(mixed_labels[1], 0.75, 0, "mixup partner label")
    checks.near(cutmix.weight, 0.75, 0, "CutMix clipped area is 4/16, not nominal 12/16")
    checks.near(cut_labels[0], 0.75, 0, "area-correct CutMix label")
    pasted_images = View(cutmix.images)
    for n in range(2):
        for y in range(4):
            for x in range(4):
                for c in range(3):
                    pasted = y < 2 and x >= 2
                    expected = f32(0.2) if (1 - n if pasted else n) == 0 else f32(0.8)
                    checks.near(pasted_images(n, y, x, c), expected, 0, "CutMix paste location")
    rng, repeat = nn.Generator(7), nn.Generator(7)
    for _ in range(20):
        weight = aug.beta_mix_weight(rng, 0.4)
        checks.near(weight, aug.beta_mix_weight(repeat, 0.4), 0, "reproducible beta mixing weights")
        sampled = aug.mixup(image, labels, partner, weight)
        si, sl = sampled.images.values(), sampled.labels.values()
        checks.near(si[0], 0.2 * weight + 0.8 * (1 - weight), 1e-7, "runtime mixup weight")
        for row in range(2):
            checks.near(sl[2 * row] + sl[2 * row + 1], 1, 1e-7, "soft label probability mass")
    checks.throws(lambda: aug.mixup(image, labels, partner, 1.5), "mixup out-of-range coefficient must fail")
    negative = nn.Rect()
    negative.top, negative.left, negative.height, negative.width = -1, 2, -1, 4
    checks.throws(lambda: aug.cutmix(image, labels, partner, negative), "negative rectangle size must fail")
    checks.throws(lambda: aug.mixup(image, labels, [1, 1], 0.5), "duplicate mixing permutation rejected")
    print("  mixup weight .25: image .65, labels=(.25,.75)")
    print("  CutMix clipped area 4/16: labels=(.75,.25), every pasted pixel verified")
    print("  20 seeded Beta(.4,.4) draws, label mass and invalid-parameter checks passed")


def main(arguments=None):
    try:
        nn.set_num_threads(2)
        print("== Spatial and color augmentation")
        transformations()
        print("\n== Image/label mixing")
        mixing()
        print("All augmentation checks passed.")
    except Exception as error:  # noqa: BLE001 - mirrors the C++ catch-all
        print(error, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
