# Native R NN example. Flip, crop, color jitter, cutout, mixup and CutMix on NHWC batches.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
# CMD: just r-example data_augmentation
# Port of tflow's data_augmentation. Check exact pixel positions, unchanged
# evaluation inputs, clipping, soft labels and area-correct CutMix labels on
# small known NHWC batches. Crop coordinates come from a host RNG that is
# reproducible for a fixed seed and toolchain.
#
# Mixup blends images and probability-vector labels by the same Beta-sampled
# weight (https://arxiv.org/abs/1710.09412). CutMix replaces a rectangle and
# weights labels by its realized clipped area (https://arxiv.org/abs/1905.04899).
# Evaluation bypasses augmentation.
#
# The transforms are nn$augment functions called eagerly. Beta weights come
# from nn$Generator(7); crop coordinates use nn$Mt19937_64(17) and reproduce
# the C++ (libc++) std::uniform_int_distribution(0, 2): the low two bits of
# each engine output, rejecting 3.
nn_helper("checks")

augmentation_f32 <- function(values) readBin(writeBin(as.numeric(values), raw(), size = 4L), "double",
    n = length(values), size = 4L)

# [N, H, W, C] accessor (0-based coordinates) for a contiguous tensor.
augmentation_view <- function(tensor) {
    v <- tensor$values()
    h <- tensor$size(1L)
    w <- tensor$size(2L)
    channels <- tensor$size(3L)
    function(n, y, x, ch) v[((n * h + y) * w + x) * channels + ch + 1]
}

augmentation_rect <- function(top, left, height, width) {
    rect <- nn$Rect()
    rect$top <- top
    rect$left <- left
    rect$height <- height
    rect$width <- width
    rect
}

# libc++ uniform_int_distribution<int>(0, 2) over a 64-bit engine.
augmentation_uniform_int_0_2 <- function(engine) {
    repeat {
        value <- engine$modulo(4)  # engine() & 3
        if (value < 3) return(value)
    }
}

augmentation_transformations <- function() {
    pv <- augmentation_f32(0:95 / 120)
    image <- nn$from_vector(pv, c(2, 4, 4, 3))
    pixels <- augmentation_view(image)
    flip <- function(x, enabled) if (enabled) nn$augment$flip_left_right(x) else x
    flipped <- augmentation_view(flip(image, TRUE))
    twice <- augmentation_view(flip(flip(image, TRUE), TRUE))
    bypass <- augmentation_view(flip(image, FALSE))
    crop <- augmentation_view(nn$augment$crop(image, 1, 1, 2, 2))
    identity <- augmentation_view(nn$augment$color_jitter(image, 0, 1, 1, 0))
    bright <- nn$augment$color_jitter(nn$full(c(1, 2, 2, 3), 0.8), 0.3, 1, 1, 0)$values()
    green <- nn$augment$color_jitter(nn$from_vector(c(1, 0, 0), c(1, 1, 1, 3)), 0, 1, 1, 1 / 3)$values()
    cutout <- augmentation_view(nn$augment$cutout(image, augmentation_rect(1, 1, 2, 2)))
    for (n in 0:1) for (y in 0:3) for (x in 0:3) for (c in 0:2) {
        original <- pixels(n, y, x, c)
        nn_near(flipped(n, y, x, c), pixels(n, y, 3 - x, c), 0, "horizontal flip axis")
        nn_near(twice(n, y, x, c), original, 0, "double flip identity")
        nn_near(bypass(n, y, x, c), original, 0, "evaluation bypass")
        nn_near(identity(n, y, x, c), original, 2e-7, "identity color jitter")
        erased <- y >= 1 && y < 3 && x >= 1 && x < 3
        nn_near(cutout(n, y, x, c), if (erased) 0 else original, 0, "cutout region")
    }
    for (n in 0:1) for (y in 0:1) for (x in 0:1) for (c in 0:2)
        nn_near(crop(n, y, x, c), pixels(n, y + 1, x + 1, c), 0, "crop offset")
    for (v in bright) nn_near(v, 1, 0, "brightness clamp")
    for (c in 0:2) nn_near(green[c + 1], if (c == 1) 1 else 0, 2e-7, "hue red to green")
    rng <- nn$Mt19937_64(17)
    again <- nn$Mt19937_64(17)
    for (trial in 1:12) {
        y <- augmentation_uniform_int_0_2(rng)
        x <- augmentation_uniform_int_0_2(rng)
        nn_require(y == augmentation_uniform_int_0_2(again) && x == augmentation_uniform_int_0_2(again),
            "seeded crop coordinates")
        out <- augmentation_view(nn$augment$crop(image, y, x, 2, 2))
        nn_near(out(0, 0, 0, 0), pixels(0, y, x, 0), 0, "runtime crop coordinates")
    }
    cat("  flip, evaluation bypass, crop coordinates, color jitter and cutout passed\n")
    cat("  12 seeded random crops reused one crop function\n")
}

augmentation_mixing <- function() {
    pv <- ifelse(0:95 < 48, 0.2, 0.8)
    image <- nn$from_vector(pv, c(2, 4, 4, 3))
    labels <- nn$from_vector(c(1, 0, 0, 1), c(2, 2))
    partner <- c(1, 0)
    mixup <- nn$augment$mixup(image, labels, partner, 0.25)
    cutmix <- nn$augment$cutmix(image, labels, partner, augmentation_rect(-1, 2, 3, 4))
    mixed_images <- mixup$images$values()
    mixed_labels <- mixup$labels$values()
    cut_labels <- cutmix$labels$values()
    nn_near(mixed_images[1], 0.65, 1e-7, "mixup image interpolation")
    nn_near(mixed_labels[1], 0.25, 0, "mixup same label coefficient")
    nn_near(mixed_labels[2], 0.75, 0, "mixup partner label")
    nn_near(cutmix$weight, 0.75, 0, "CutMix clipped area is 4/16, not nominal 12/16")
    nn_near(cut_labels[1], 0.75, 0, "area-correct CutMix label")
    pasted_images <- augmentation_view(cutmix$images)
    low_high <- augmentation_f32(c(0.2, 0.8))
    for (n in 0:1) for (y in 0:3) for (x in 0:3) for (c in 0:2) {
        pasted <- y < 2 && x >= 2
        expected <- if ((if (pasted) 1 - n else n) == 0) low_high[1] else low_high[2]
        nn_near(pasted_images(n, y, x, c), expected, 0, "CutMix paste location")
    }
    rng <- nn$Generator(7)
    again <- nn$Generator(7)
    for (trial in 1:20) {
        weight <- nn$augment$beta_mix_weight(rng, 0.4)
        nn_near(weight, nn$augment$beta_mix_weight(again, 0.4), 0, "reproducible beta mixing weights")
        sampled <- nn$augment$mixup(image, labels, partner, weight)
        si <- sampled$images$values()
        sl <- sampled$labels$values()
        nn_near(si[1], 0.2 * weight + 0.8 * (1 - weight), 1e-7, "runtime mixup weight")
        for (row in 0:1) nn_near(sl[2 * row + 1] + sl[2 * row + 2], 1, 1e-7, "soft label probability mass")
    }
    nn_throws(nn$augment$mixup(image, labels, partner, 1.5), "mixup out-of-range coefficient must fail")
    nn_throws(nn$augment$cutmix(image, labels, partner, augmentation_rect(-1, 2, -1, 4)),
        "negative rectangle size must fail")
    nn_throws(nn$augment$mixup(image, labels, c(1, 1), 0.5), "duplicate mixing permutation rejected")
    cat("  mixup weight .25: image .65, labels=(.25,.75)\n")
    cat("  CutMix clipped area 4/16: labels=(.75,.25), every pasted pixel verified\n")
    cat("  20 seeded Beta(.4,.4) draws, label mass and invalid-parameter checks passed\n")
}

main <- function(arguments = commandArgs(TRUE)) {
    tryCatch({
        nn$set_num_threads(2L)
        cat("== Spatial and color augmentation\n")
        augmentation_transformations()
        cat("\n== Image/label mixing\n")
        augmentation_mixing()
        cat("All augmentation checks passed.\n")
        0L
    }, error = function(e) {
        message(conditionMessage(e))
        1L
    })
}

if (sys.nframe() == 0L) quit(status = main())
