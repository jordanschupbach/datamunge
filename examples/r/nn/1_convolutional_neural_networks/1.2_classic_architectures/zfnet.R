# Native R NN example. ZFNet architecture checks, compact CIFAR-10 training and deconvnet visualization; the C++ example's loop, written out in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.2_classic_architectures/zfnet.R
# ZFNet: finer early sampling, dense connectivity and deconvnet visualization.
#
# Port of tflow's zfnet. Build the Figure-3 network from Zeiler/Fergus, ECCV
# 2014. Check full 224-pixel stage shapes, 62,357,608 parameters and finite
# nonzero derivatives through every learned tensor. The shared header
# documents exact padding.
# Reference: https://cs.nyu.edu/~fergus/papers/zeilerECCV2014.pdf
#
# Train a separate compact version on 12k CIFAR-10 rows: quarter channel
# widths, 256-unit dense layers, 67-pixel resized input, ten epochs, crop/flip
# augmentation, He initialization and Adam. Keep dense connectivity, 7x7/5x5
# stride-2 early convolutions, LRN after pooling and .5 dropout. Select on a
# fixed 5k validation split and evaluate all 10k official test rows.
# These compact results are not full-model ImageNet results.
#
# Restore the selected model and project its strongest conv5 activation for
# the first test image back to pixels: unpool at forward switches, rectify the
# backward signal and apply convolution adjoints, omitting LRN. Compare with
# the ordinary input gradient. A switch fixture and a ReLU sign fixture
# distinguish this reconstruction rule from standard or guided backprop.
# CIFAR-10 is read from paths::Dataset("cifar10"). Export:
# output/nn/visualizations/zfnet/features.npz (tflow: output/visualizations/zfnet,
# rendered by tflow's scripts/plot_zfnet.py). This is a single-feature
# diagnostic, not an inverse network or a complete explanation of the class.
# Options: data=PATH epochs=10 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.
#
# The model components (zfnet::ZFNet and ZFNetDeconvolution, the deconvnet
# primitives, architecture fixtures, image batching, cross-entropy and L2) are
# the C++ ones through SWIG (datamungerNN::nn_components); the checks and the
# workflow below mirror examples/cpp/source/nn/.../zfnet.cpp line by line.
suppressPackageStartupMessages(library(datamungerNN))
architecture <- nn_components$architecture
check <- nn_components$check
dataset_assets <- nn_components$dataset_assets
deconvnet <- nn_components$deconvnet
example_runtime <- nn_components$example_runtime
image_training <- nn_components$image_training
paths <- nn_components$paths
zfnet <- nn_components$zfnet

# C++ default ostream formatting (six significant digits).
g6 <- function(x) sprintf("%g", x)

# Quarter-width, 256-unit, 67-pixel, ten-class ZFNet (CompactConfig in the C++).
compact_config <- function() {
    cfg <- zfnet$ZFNetConfig()
    cfg$image_size <- 67L
    cfg$classes <- 10L
    cfg$width_divisor <- 4L
    cfg$dense_width <- 256L
    cfg
}

# Switch fixture and ReLU sign fixture: deconvnet rectification versus ordinary backprop.
reconstruction_checks <- function() {
    values <- nn$from_vector(c(1, 9, 2, 3, 4, 5, 6, 10, 7, 8, 12, 0, 11, 2, 1, 4), c(1L, 4L, 4L, 1L))
    signal <- nn$from_vector(c(1, 2, 3, 4), c(1L, 2L, 2L, 1L))
    unpooled <- deconvnet$UnpoolAtSwitches(values, signal, 2L, 2L, "VALID")$values()
    x <- nn$from_vector(c(-1, 2), c(1L, 1L, 1L, 2L))
    x$requires_grad_(TRUE)
    backward <- nn$from_vector(c(3, -4), c(1L, 1L, 1L, 2L))
    identity <- nn$from_vector(c(1, 0, 0, 1), c(1L, 1L, 2L, 2L))
    ordinary <- nn$grad(nn$sum(nn$mul(nn$relu(x), backward)), list(x))[[1L]]$values()
    projection <- deconvnet$DeconvLayer(x, identity, backward, 1L, "VALID")$values()
    expected <- numeric(16L)  # C++ 0-based positions 1, 7, 12, 10
    expected[2L] <- 1
    expected[8L] <- 2
    expected[13L] <- 3
    expected[11L] <- 4
    for (i in seq_len(16L))
        check$Near(unpooled[i], expected[i], 0, "unpool uses original maximum locations")
    check$Near(ordinary[1L], 0, 0, "ordinary ReLU blocks negative forward input")
    check$Near(ordinary[2L], -4, 0, "ordinary ReLU can pass a negative derivative")
    check$Near(projection[1L], 3, 0, "deconvnet passes positive backward signal")
    check$Near(projection[2L], 0, 0, "deconvnet blocks negative backward signal")
    cat("  unpool switches and deconvnet-vs-backprop rectification checks passed\n")
}

# Figure-3 stage shapes, 62,357,608 parameters and finite nonzero parameter derivatives at 224 pixels.
full_architecture_checks <- function() {
    p <- nn$ParameterStore(17L)
    net <- zfnet$ZFNet(p, architecture$SinePixels(c(1L, 224L, 224L, 3L), .5, .13), zfnet$ZFNetConfig())
    check$Require(p$count() == 62357608, "full ZFNet parameter count")
    stages <- c(lapply(0:4, function(i) net$convolution(i)), lapply(0:2, function(i) net$pooling(i)),
                list(net$fc6, net$fc7, net$logits))
    expected <- list(c(1L, 110L, 110L, 96L), c(1L, 26L, 26L, 256L), c(1L, 13L, 13L, 384L), c(1L, 13L, 13L, 384L),
                     c(1L, 13L, 13L, 256L), c(1L, 55L, 55L, 96L), c(1L, 13L, 13L, 256L), c(1L, 6L, 6L, 256L),
                     c(1L, 4096L), c(1L, 4096L), c(1L, 1000L))
    for (i in seq_along(expected))
        architecture$RequireShape(stages[[i]], expected[[i]], "ZFNet Figure-3 stage shape")
    norms <- architecture$GradientNormRange(p, nn$mean(nn$square(net$logits)), "ZFNet parameter tensor")
    cat(sprintf("  full ZFNet: %d parameters; conv sizes 110/26/13/13/13; pool sizes 55/13/6\n", p$count()))
    cat(sprintf("  dense widths 4096/4096/1000; parameter gradient norms [%s,%s]\n", g6(norms$first), g6(norms$second)))
}

# Reload the selected compact checkpoint and project the first test image's strongest conv5 unit to pixels.
visualize <- function(directory) {
    p <- nn$ParameterStore(17L)
    test <- dataset_assets$LoadCifar10(directory, "test")
    batch <- image_training$ImageBatch(test, 0)
    pixels <- batch$first
    labels <- batch$second
    image <- nn$resize_bilinear(pixels, c(67L, 67L), FALSE, TRUE)
    nn$no_grad(function() zfnet$ZFNet(p, image, compact_config()))
    p$load(file.path(paths$OutputDir("models/zfnet_compact_cifar10"), "best_params.npz"))
    image$requires_grad_(TRUE)
    net <- zfnet$ZFNet(p, image, compact_config())
    activation_map <- net$convolution(4L)$values()
    check$Require(identical(as.numeric(net$convolution(4L)$shape()), c(1, 3, 3, 64)), "compact conv5 map")
    index <- which.max(activation_map)  # the first strictly greatest activation, as the C++ scan
    activation <- activation_map[index]
    check$Require(activation > 0, "positive selected conv5 activation")
    seed <- numeric(length(activation_map))
    seed[index] <- activation
    signal <- nn$from_vector(seed, c(1L, 3L, 3L, 64L))
    gradient <- nn$grad(nn$sum(nn$mul(net$convolution(4L), signal)), list(image))[[1L]]
    deconvolution <- nn$no_grad(function() zfnet$ZFNetDeconvolution(p, net, signal))
    check$Require(identical(as.numeric(deconvolution$shape()), c(1, 67, 67, 3)) &&
                      identical(as.numeric(gradient$shape()), as.numeric(deconvolution$shape())),
                  "projection image dimensions")
    a <- deconvolution$values()
    b <- gradient$values()
    deconv_energy <- 0
    gradient_energy <- 0
    difference <- 0
    for (i in seq_along(a)) {  # sequential double accumulation, in the C++ order
        check$Require(is.finite(a[i]) && is.finite(b[i]), "finite feature projections")
        deconv_energy <- deconv_energy + a[i] * a[i]
        gradient_energy <- gradient_energy + b[i] * b[i]
        difference <- difference + (a[i] - b[i]) * (a[i] - b[i])
    }
    check$Require(deconv_energy > 0 && gradient_energy > 0 && difference > 1e-8,
                  "nonzero distinct deconvnet and ordinary gradients")
    unit <- index - 1L  # C++ 0-based flat index into [1, 3, 3, 64]
    unit_row <- unit %/% (3L * 64L)
    unit_column <- (unit %/% 64L) %% 3L
    unit_channel <- unit %% 64L
    prediction <- nn$argmax(net$logits$detach(), 1L)
    path <- file.path(paths$OutputDir("visualizations/zfnet"), "features.npz")
    nn$save_npz(path, list(image = image$detach(),
                           label = labels,
                           prediction = prediction,
                           unit = nn$from_vector_int64(c(unit_row, unit_column, unit_channel), 3L),
                           deconvolution = deconvolution,
                           gradient = gradient))
    cat(sprintf("  first test image conv5 unit=%d,%d,%d; activation=%s; deconv energy=%s; ordinary gradient energy=%s\n",
                unit_row, unit_column, unit_channel, g6(activation), g6(deconv_energy), g6(gradient_energy)))
    cat(sprintf("  wrote %s\nAll ZFNet architecture and visualization checks passed.\n", path))
}

# image_training::ParseCifarOption over key=value arguments; returns list(directory, explicit_epochs).
parse_cifar_options <- function(arguments, cfg) {
    directory <- paths$Dataset("cifar10")
    explicit_epochs <- FALSE
    for (argument in arguments) {
        parts <- regmatches(argument, regexpr("=", argument), invert = TRUE)[[1L]]
        if (length(parts) != 2L) stop("expected key=value option")
        key <- parts[1L]
        value <- parts[2L]
        if (key == "data") directory <- value
        else if (key == "epochs") { cfg$epochs <- as.integer(value); explicit_epochs <- TRUE }
        else if (key == "batch") cfg$batch <- as.integer(value)
        else if (key == "lr") cfg$learning_rate <- as.numeric(value)
        else if (key == "fit") cfg$maximum_fit_rows <- as.integer(value)
        else if (key == "augment" && value %in% c("0", "1")) cfg$augment <- value == "1"
        else if (key == "threads") nn$set_num_threads(as.integer(value))
        else if (key == "seed") cfg$seed <- as.numeric(value)
        else stop("unknown option: ", argument)
    }
    list(directory = directory, explicit_epochs = explicit_epochs)
}

main <- function(arguments = commandArgs(TRUE)) {
    reconstruction_checks()
    full_architecture_checks()
    cfg <- image_training$ImageTrainingConfig()
    cfg$name <- "zfnet_compact_cifar10"
    cfg$maximum_fit_rows <- 12000L
    cfg$minimum_test_accuracy <- .35
    options <- parse_cifar_options(arguments, cfg)
    explicit_epochs <- options$explicit_epochs
    data <- image_training$LoadCifar10(options$directory)
    development <- data$first
    test <- data$second
    # Compact ZFNet on CIFAR-10: fixed development split, Adam on cross-entropy
    # plus L2, validation-NLL checkpoint selection, one test pass; visualize then
    # reloads the selected checkpoint.
    model <- function(params, state, x, training) {
        zfnet$ZFNet(params, nn$resize_bilinear(x, c(67L, 67L), FALSE, TRUE), compact_config(), training)$logits
    }

    # Quick mode bounds the data and batch; an explicit epoch budget is honored.
    if (!explicit_epochs) cfg$epochs <- example_runtime$Limit(cfg$epochs, 1L)
    cfg$batch <- example_runtime$Limit(cfg$batch, 8L)
    if (!example_runtime$Full())
        cfg$maximum_fit_rows <- if (cfg$maximum_fit_rows > 0) min(cfg$maximum_fit_rows, 32L) else 32L
    example_runtime$Describe()
    development_shape <- as.numeric(development$images$shape())
    test_shape <- as.numeric(test$images$shape())
    development_rows <- development_shape[1L]
    test_size <- example_runtime$LimitInt64(test_shape[1L], 32)
    check$Require(length(development_shape) == 4L && length(test_shape) == 4L && cfg$validation_rows > 0 &&
                      cfg$validation_rows < development_rows && cfg$epochs > 0 && cfg$batch > 0 && cfg$classes > 1,
                  "valid image training dimensions/config")
    check$Require(identical(development_shape[2:4], test_shape[2:4]), "training/test image dimensions match")
    available <- development_rows - cfg$validation_rows
    check$Require(cfg$maximum_fit_rows >= 0 && cfg$maximum_fit_rows <= available, "valid optional fitting-row limit")
    fit_count <- if (cfg$maximum_fit_rows > 0) cfg$maximum_fit_rows else available
    check$Require(fit_count >= 2, "at least two fitting rows for model initialization")

    # Split the development rows once (std::mt19937_64(2026) shuffle); keep the
    # validation split fixed when limiting the fitting subset.
    order <- as.numeric(nn$Mt19937_64(2026)$shuffle(seq_len(development_rows) - 1))
    fit <- order[seq_len(fit_count)]
    validation <- order[(available + 1):development_rows]
    validation <- head(validation, example_runtime$Limit(length(validation), 32L))
    test_rows <- seq_len(test_size) - 1

    params <- nn$ParameterStore(cfg$seed)
    state <- nn$ParameterStore(cfg$seed)
    nn$no_grad(function() {  # create every parameter (and batch-norm state) before building the optimizer
        first <- image_training$ImageBatch(development, fit[1:2])
        model(params, state, first$first, FALSE)
    })
    config <- nn$OptimizerConfig()
    config$kind <- nn$OptimizerKind$Adam
    optimizer <- nn$Optimizer(params, cfg$learning_rate, config)

    # Inference-mode cross-entropy, accuracy and confusion over `rows`, in batches of 256.
    evaluate <- function(dataset, rows) {
        loss <- nn$WeightedMean()
        accuracy <- nn$WeightedMean()
        confusion <- nn$ConfusionMatrix(cfg$classes)
        nn$no_grad(function() {
            for (start in seq.int(1L, length(rows), by = 256L)) {
                ids <- rows[start:min(start + 255L, length(rows))]
                batch <- image_training$ImageBatch(dataset, ids)
                logits <- model(params, state, batch$first, FALSE)
                prediction <- nn$argmax(logits, 1L)
                loss$add(nn$sparse_softmax_cross_entropy(logits, batch$second)$item(), length(ids))
                accuracy$add(nn$accuracy(prediction, batch$second), length(ids))
                confusion$add(batch$second, prediction)
            }
        })
        list(loss = loss$value(), accuracy = accuracy$value(), confusion = confusion)
    }

    directory <- paths$OutputDir(file.path("models", cfg$name))
    params_path <- file.path(directory, "best_params.npz")
    state_path <- file.path(directory, "best_state.npz")
    history <- file(file.path(directory, "history.tsv"), "w")
    writeLines("epoch\tvalidation_loss\tvalidation_accuracy", history)
    cat(sprintf("== %s: %d fit / %d validation / %d test\n", cfg$name, as.integer(fit_count), length(validation),
                as.integer(test_size)))
    cat(sprintf("  parameters=%.0f; augmentation=%d; batch=%d; lr=%s; threads=%d\n", params$count(),
                as.integer(cfg$augment), as.integer(cfg$batch), g6(cfg$learning_rate), nn$get_num_threads()))
    batches <- nn$BatchOrder(fit_count, cfg$batch, 29L)
    augmentation <- nn$Generator(151L)
    best <- Inf
    best_epoch <- -1L
    epoch_seconds <- numeric()
    for (epoch in seq_len(cfg$epochs)) {
        epoch_start <- proc.time()[["elapsed"]]
        batches$begin_epoch()
        used <- 0
        repeat {
            rows <- batches$`_next`()
            if (!length(rows)) break
            used <- used + length(rows)
            rows <- fit[rows + 1]
            # Augmented batch; training-mode forward; cross-entropy plus 0.5 * l2 * sum(W^2); one Adam step.
            batch <- if (cfg$augment) image_training$ImageBatch(development, rows, augmentation)
                     else image_training$ImageBatch(development, rows)
            logits <- model(params, state, batch$first, TRUE)
            objective <- image_training$WithL2(
                image_training$TrainingCrossEntropy(logits, batch$second, cfg$classes, cfg$label_smoothing), params, cfg$l2)
            optimizer$minimize(objective)
        }
        check$Require(used == fit_count, "each fit image used once per epoch")
        train_seconds <- proc.time()[["elapsed"]] - epoch_start
        result <- evaluate(development, validation)
        if (result$loss < best) {
            best <- result$loss
            best_epoch <- epoch
            params$save(params_path)
            if (state$size() > 0) state$save(state_path)
        }
        writeLines(sprintf("%d\t%s\t%s", epoch, g6(result$loss), g6(result$accuracy)), history)
        flush(history)
        seconds <- proc.time()[["elapsed"]] - epoch_start
        epoch_seconds <- c(epoch_seconds, seconds)
        cat(sprintf("  epoch %d: validation NLL=%s, accuracy=%s (%.3g s train, %.3g s total)\n", epoch,
                    g6(result$loss), g6(result$accuracy), train_seconds, seconds))
    }
    close(history)
    params$load(params_path)
    if (state$size() > 0) state$load(state_path)
    check$Near(evaluate(development, validation)$loss, best, 1e-6, "best image checkpoint validation loss")
    result <- evaluate(test, test_rows)
    cat(sprintf("  selected epoch=%d; test NLL=%s; accuracy=%s\n", best_epoch, g6(result$loss), g6(result$accuracy)))
    # Confusion matrix: every evaluation row, truth on rows and predictions on columns.
    cat("  confusion (truth rows):\n")
    cells <- matrix(round(result$confusion$to_tensor()$values()), nrow = cfg$classes, byrow = TRUE)
    for (i in seq_len(nrow(cells)))
        cat("   ", paste0(" ", formatC(cells[i, ], width = 4L), collapse = ""), "\n", sep = "")
    check$Require(sum(cells) == test_size, "all test images evaluated")
    check$Near(result$accuracy, sum(diag(cells)) / sum(cells), 1e-7, "confusion agrees with accuracy")
    check$TrainingQuality(result$accuracy >= cfg$minimum_test_accuracy,
                          "image classifier demonstration accuracy threshold")
    train_total <- sum(epoch_seconds)
    cat(sprintf("  training time %.1f s (%.2f s/epoch)\n", train_total, train_total / cfg$epochs))
    cat(sprintf("  best checkpoint and history: %s\nAll %s checks passed.\n", directory, cfg$name))
    visualize(options$directory)
    0L
}

if (sys.nframe() == 0L) quit(status = main())
