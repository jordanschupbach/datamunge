# Native R NN example. CIFAR-10 CNN with crop/flip augmentation, trained in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.1_fundamentals/cnn_cifar10.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# A small CIFAR-10 CNN trained with independent random crops and flips.
#
# Port of tflow's cnn_cifar10. Preparation (in ~/projects/tflow_examples):
# python3 scripts/fetch_datasets.py cifar10 --split both.
# Architecture: Conv(32,3)-ReLU-pool -> Conv(64,3)-ReLU-pool -> Dense(128)-
# ReLU -> 10 logits. Train on 45k original images with four-pixel zero padding,
# random 32x32 crops and horizontal flips; normalize RGB pixels to [-1,1].
# Select by NLL on 5k untouched validation images, restore that checkpoint,
# then evaluate all 10k original test images once. Adam/L2; no pretrained data.
# Options: data=PATH epochs=10 batch=128 lr=.001 augment=0|1 threads=N seed=17 fit=N.
# Quick mode uses 32 fit / 32 validation / 32 test images, batch <= 8 and one
# epoch unless epochs= is given.
# The default run requires 55% test accuracy as a training sanity check.
# Checkpoint/history: output/nn/models/cnn_cifar10/ (or cnn_cifar10_no_augmentation).

nn_helper("checks")
nn_helper("image_training")
nn_helper("mnist")

g6 <- function(value) sprintf("%.6g", value)

augmentation_checks <- function() {
    fixture <- nn$ImageDataset()
    fixture$images <- nn$ones(c(2L, 32L, 32L, 3L))
    fixture$labels <- nn$from_vector_int64(c(2, 7), 2L)
    first <- nn$Generator(151L)
    repeat_generator <- nn$Generator(151L)
    plain <- image_training_image_batch(fixture, c(1, 0))
    augmented <- image_training_image_batch(fixture, c(1, 0), first)
    reproduced <- image_training_image_batch(fixture, c(1, 0), repeat_generator)
    p <- plain$x$values()
    a <- augmented$x$values()
    r <- reproduced$x$values()
    nn_require(all(p == 1), "evaluation only normalizes pixels")
    nn_require(identical(a, r), "seeded per-image augmentation")
    nn_require(all(a == -1 | a == 1), "crop preserves white image or zero padding")
    nn_require(any(a == -1) && any(a == 1), "random crop includes padded and original pixels")
    labels <- augmented$y$values()
    nn_require(labels[1L] == 7 && labels[2L] == 2, "augmentation preserves labels and row order")
    cat("  augmentation checks: reproducible crops, normalized padding and paired labels passed\n")
}

# Conv(32,3)-ReLU-pool, Conv(64,3)-ReLU-pool, Dense(128)-ReLU, 10 logits.
cifar_model <- function(p, state, image, training = FALSE) {
    conv <- nn$ConvLayerOptions()
    conv$activation <- nn$Activation$Relu
    first <- nn$max_pool2d(nn$conv2d(p, "conv1", image, 32L, 3L, conv), 2L, 2L)
    second <- nn$max_pool2d(nn$conv2d(p, "conv2", first, 64L, 3L, conv), 2L, 2L)
    hidden <- nn$dense(p, "dense", nn$reshape(second, c(-1L, 8L * 8L * 64L)), 128L, nn$Activation$Relu)
    nn$dense(p, "output", hidden, 10L)
}

main <- function(arguments = commandArgs(TRUE)) {
    defaults <- image_training_cifar_option_defaults(nn_dataset("cifar10"))
    defaults$epochs <- 0  # zero: the configuration default (one epoch in quick mode)
    defaults$full <- Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"
    args <- nn_options(arguments, defaults)
    Sys.setenv(DATAMUNGE_EXAMPLES_FULL = if (isTRUE(args$full)) "1" else "0")
    explicit_epochs <- args$epochs != 0
    cfg <- image_training_config(name = "cnn_cifar10")
    default_epochs <- cfg$epochs
    cfg <- image_training_apply_cifar_options(args, cfg)
    if (!explicit_epochs) cfg$epochs <- default_epochs
    if (!cfg$augment) cfg$name <- paste0(cfg$name, "_no_augmentation")
    augmentation_checks()
    data <- image_training_load_cifar10(args$data)
    development <- data$development
    test <- data$test

    # Quick mode bounds the data and batch; an explicit epoch budget is honored.
    if (!explicit_epochs) cfg$epochs <- nn_limit(cfg$epochs, 1L)
    cfg$batch <- nn_limit(cfg$batch, 8L)
    if (!nn_full()) cfg$maximum_fit_rows <- if (cfg$maximum_fit_rows > 0) min(cfg$maximum_fit_rows, 32L) else 32L
    nn_describe()
    development_shape <- as.numeric(development$images$shape())
    test_shape <- as.numeric(test$images$shape())
    development_rows <- development_shape[1L]
    test_size <- nn_limit(test_shape[1L], 32)
    nn_require(length(development_shape) == 4L && length(test_shape) == 4L && cfg$validation_rows > 0 &&
                   cfg$validation_rows < development_rows && cfg$epochs > 0 && cfg$batch > 0 && cfg$classes > 1,
               "valid image training dimensions/config")
    nn_require(identical(development_shape[2:4], test_shape[2:4]), "training/test image dimensions match")
    available <- development_rows - cfg$validation_rows
    nn_require(cfg$maximum_fit_rows >= 0 && cfg$maximum_fit_rows <= available, "valid optional fitting-row limit")
    fit_count <- if (cfg$maximum_fit_rows > 0) cfg$maximum_fit_rows else available

    # Split the development rows once (std::mt19937_64(2026) shuffle); keep the
    # validation split fixed when limiting the fitting subset.
    order <- as.numeric(nn$Mt19937_64(2026)$shuffle(seq_len(development_rows) - 1))
    fit <- order[seq_len(fit_count)]
    validation <- order[(available + 1):development_rows]
    validation <- head(validation, nn_limit(length(validation), 32L))
    test_rows <- seq_len(test_size) - 1

    params <- nn$ParameterStore(cfg$seed)
    state <- nn$ParameterStore(cfg$seed)
    nn$no_grad(function() cifar_model(params, state, image_training_image_batch(development, fit[1:2])$x, FALSE))
    config <- nn$OptimizerConfig()
    config$kind <- nn$OptimizerKind$Adam
    optimizer <- nn$Optimizer(params, cfg$learning_rate, config)

    directory <- nn_output_dir(file.path("models", cfg$name))
    params_path <- file.path(directory, "best_params.npz")
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
            # Augmented batch, cross-entropy plus 0.5 * l2 * sum(W^2), one Adam step.
            batch <- image_training_image_batch(development, rows, if (cfg$augment) augmentation else NULL)
            optimizer$minimize(image_training_objective(params, state, cifar_model, batch$x, batch$y, cfg))
        }
        nn_require(used == fit_count, "each fit image used once per epoch")
        train_seconds <- proc.time()[["elapsed"]] - epoch_start
        result <- image_training_evaluate_images(params, state, cifar_model, development, validation, cfg$classes)
        loss <- result$loss$value()
        accuracy <- result$accuracy$value()
        if (loss < best) {
            best <- loss
            best_epoch <- epoch
            params$save(params_path)
        }
        writeLines(sprintf("%d\t%s\t%s", epoch, g6(loss), g6(accuracy)), history)
        flush(history)
        seconds <- proc.time()[["elapsed"]] - epoch_start
        epoch_seconds <- c(epoch_seconds, seconds)
        cat(sprintf("  epoch %d: validation NLL=%s, accuracy=%s (%.3g s train, %.3g s total)\n", epoch, g6(loss),
                    g6(accuracy), train_seconds, seconds))
    }
    close(history)
    params$load(params_path)
    nn_near(image_training_evaluate_images(params, state, cifar_model, development, validation,
                                           cfg$classes)$loss$value(), best, 1e-6, "best image checkpoint validation loss")
    result <- image_training_evaluate_images(params, state, cifar_model, test, test_rows, cfg$classes)
    cat(sprintf("  selected epoch=%d; test NLL=%s; accuracy=%s\n", best_epoch, g6(result$loss$value()),
                g6(result$accuracy$value())))
    cells <- mnist_print_confusion(result$confusion, "confusion (truth rows):")
    nn_require(sum(cells) == test_size, "all test images evaluated")
    nn_near(result$accuracy$value(), sum(diag(cells)) / sum(cells), 1e-7, "confusion agrees with accuracy")
    nn_training_quality(result$accuracy$value() >= cfg$minimum_test_accuracy,
                        "image classifier demonstration accuracy threshold")
    train_total <- sum(epoch_seconds)
    cat(sprintf("  training time %.1f s (%.2f s/epoch)\n", train_total, train_total / cfg$epochs))
    cat(sprintf("  best checkpoint and history: %s\nAll %s checks passed.\n", normalizePath(directory), cfg$name))
    0L
}

if (sys.nframe() == 0L) quit(status = main())
