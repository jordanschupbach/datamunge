# Native R NN example. SqueezeNet fire-module/geometry checks and three CIFAR-10 trainings (v1.0, v1.0 bypass, v1.1); the C++ example's loop, written out in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/squeezenet.R
# SqueezeNet: AlexNet-level design with 50x fewer parameters via fire modules.
#
# Check that a fire module's concatenated 1x1 and 3x3 expand layers equal a
# single 3x3 convolution whose first filters are the 1x1 kernels placed at
# the centre tap, including input derivatives, and compare its parameters
# with a plain 3x3 convolution of the same width (the paper's strategies 1
# and 2). Check the released spatial geometry, which depends on Caffe's
# ceil-rounded pooling (54/27/13 for v1.0 and 55/27/13 for v1.1 at 224),
# the 1,248,424 / 1,235,496 learned parameters of v1.0 / v1.1 (torchvision's
# counts; simple bypasses add none) and finite nonzero derivatives.
# https://arxiv.org/abs/1602.07360
# https://github.com/forresti/SqueezeNet
#
# Train v1.0, v1.0 with simple bypasses and v1.1 at full width on CIFAR-10
# with a 3x3 stride-one first convolution, keeping all fire modules, pools,
# dropout and the convolutional classifier. Use 12k fit / 5k validation,
# twelve epochs, crop/flip, Adam(.001), kernel L2(.0001) and the
# best-validation checkpoint before evaluating all 10k test images. These are
# CIFAR adaptations, not the paper's ImageNet benchmark.
# Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1.
#
# Port: datamunge eager autograd; seeded native initialization gives statistical
# training parity. Dataset cache and training options use image_training.hpp.
# Port default lr=.0001: with native initialization the original .001 kills
# the final ReLU classifier (chance predictions and zero data gradients).
# The original learning rate remains available with lr=.001.
#
# The model components (squeezenet::Fire/SqueezeNet, the mobile_checks sine
# fixtures, architecture::GradientNormRange, image batching, cross-entropy and
# L2) are the C++ ones through SWIG (datamungerNN::nn_components); the checks and
# the training workflow below mirror examples/cpp/source/nn/.../squeezenet.cpp line by line.
suppressPackageStartupMessages(library(datamungerNN))
architecture <- nn_components$architecture
check <- nn_components$check
example_runtime <- nn_components$example_runtime
image_training <- nn_components$image_training
mobile_checks <- nn_components$mobile_checks
paths <- nn_components$paths
squeezenet <- nn_components$squeezenet
Pattern <- mobile_checks$Pattern

# C++ default ostream formatting (six significant digits).
g6 <- function(x) sprintf("%g", x)

# nn::pad's {{before, after}, ...} list as the bound PaddingPairs vector.
padding_pairs <- function(...) {
    pairs <- nn$PaddingPairs()
    for (v in list(...)) pairs$push_back(nn$PaddingPair(v[1L], v[2L]))
    pairs
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

fire_checks <- function() {
    p <- nn$ParameterStore(17L)
    x <- Pattern(c(2L, 6L, 6L, 5L), 1, .1)$requires_grad_()
    squeezenet$Fire(p, "fire", x, 3L, 4L, 6L)
    for (name in p$names())
        if (mobile_checks$EndsWith(name, "/b")) p$set(name, Pattern(p$at(name)$shape(), .2, nchar(name)))
    fire <- squeezenet$Fire(p, "fire", x, 3L, 4L, 6L)
    centred <- nn$pad(p$at("fire/expand1x1/W"), padding_pairs(c(1L, 1L), c(1L, 1L), c(0L, 0L), c(0L, 0L)))
    kernel <- nn$concat(nn$tensor_vector(list(centred, p$at("fire/expand3x3/W"))), 3L)
    bias <- nn$concat(nn$tensor_vector(list(p$at("fire/expand1x1/b"), p$at("fire/expand3x3/b"))), 0L)
    o <- nn$Conv2DOptions()
    o$padding <- nn$Padding$Same
    merged <- nn$relu(nn$add(nn$conv2d(fire$squeeze, kernel, o), bias))
    weight <- Pattern(c(2L, 6L, 6L, 10L), 1, .3)
    ga <- nn$grad(nn$sum(nn$mul(fire$output, weight)), list(x))
    # The first backward releases its graph, so recompute the merged branch.
    fire <- squeezenet$Fire(p, "fire", x, 3L, 4L, 6L)
    merged <- nn$relu(nn$add(nn$conv2d(fire$squeeze, kernel, o), bias))
    gb <- nn$grad(nn$sum(nn$mul(merged, weight)), list(x))
    check$Near(nn$max(nn$abs(nn$sub(fire$output, merged)))$item(), 0, 1e-5,
               "fire module equals centre-embedded convolution")
    check$Near(nn$max(nn$abs(nn$sub(ga[[1L]], gb[[1L]])))$item(), 0, 1e-5, "fire input derivatives")
    fire2 <- 96 * 16 + 16 + 16 * 64 + 64 + 9 * 16 * 64 + 64
    plain <- 9 * 96 * 128 + 128
    cat(sprintf("fire2 parameters %d versus plain %d\n", as.integer(fire2), as.integer(plain)))
}

full_checks <- function() {
    cases <- list(
        list(10L, FALSE, 1248424, c(54L, 54L, 54L, 27L, 27L, 27L, 27L, 13L)),
        list(10L, TRUE, 1248424, c(54L, 54L, 54L, 27L, 27L, 27L, 27L, 13L)),
        list(11L, FALSE, 1235496, c(55L, 55L, 27L, 27L, 13L, 13L, 13L, 13L)))
    for (cs in cases) {
        version <- cs[[1L]]
        bypass <- cs[[2L]]
        p <- nn$ParameterStore(17L)
        gen <- nn$Generator(101L)
        cfg <- squeezenet$SqueezeNetConfig()
        cfg$version <- version
        cfg$simple_bypass <- bypass
        net <- squeezenet$SqueezeNet(p, Pattern(c(2L, 224L, 224L, 3L), 1, 0), cfg, TRUE, gen)
        check$Require(p$count() == cs[[3L]] && identical(as.integer(net$sizes), cs[[4L]]),
                      "SqueezeNet count and ceil geometry")
        stem <- if (version == 10L) 109 else 111
        fires <- nn$tensor_list(net$fires)
        last <- fires[[length(fires)]]
        check$Require(net$stem$size(1L) == stem && last$size(1L) == 13 && last$size(3L) == 512 &&
                          identical(as.numeric(net$logits$shape()), c(2, 1000)),
                      "SqueezeNet stem and classifier dimensions")
        norms <- architecture$GradientNormRange(p, nn$mean(nn$square(net$logits)), "SqueezeNet")
        cat(sprintf("SqueezeNet %d bypass=%d parameters=%.0f smallest gradient=%s\n", version, as.integer(bypass),
                    p$count(), g6(norms$first)))
    }
}

main <- function(arguments = commandArgs(TRUE)) {
    nn$set_num_threads(2L)
    fire_checks()
    full_checks()
    runs <- list(
        list("squeezenet10_cifar10", 10L, FALSE, 729034),
        list("squeezenet10_bypass_cifar10", 10L, TRUE, 729034),
        list("squeezenet11_cifar10", 11L, FALSE, 727626))
    for (run in runs) {
        run_name <- run[[1L]]
        run_version <- run[[2L]]
        run_bypass <- run[[3L]]
        run_params <- run[[4L]]
        cfg <- image_training$ImageTrainingConfig()
        cfg$name <- run_name
        cfg$maximum_fit_rows <- 12000L
        cfg$learning_rate <- .0001
        cfg$epochs <- 12L
        cfg$minimum_test_accuracy <- .4
        generator <- nn$Generator(101L)
        # image_training::LogitsBuilder: the CIFAR SqueezeNet variant of this run (one dropout generator per run).
        model <- function(p, state, image, training) {
            model_cfg <- squeezenet$SqueezeNetConfig()
            model_cfg$version <- run_version
            model_cfg$simple_bypass <- run_bypass
            model_cfg$classes <- 10L
            model_cfg$image_size <- 32L
            model_cfg$cifar_stem <- TRUE
            net <- squeezenet$SqueezeNet(p, image, model_cfg, training, generator)
            check$Require(p$count() == run_params && state$count() == 0, "CIFAR SqueezeNet parameter count")
            net$logits
        }
        options <- parse_cifar_options(arguments, cfg)
        explicit_epochs <- options$explicit_epochs
        check$Require(cfg$classes == 10, "CIFAR-10 requires ten output classes")
        data <- image_training$LoadCifar10(options$directory)
        development <- data$first
        test <- data$second
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
                # Augmented batch; training-mode forward; cross-entropy plus 0.5 * l2 * sum(W^2); one
                # Adam step.
                batch <- if (cfg$augment) image_training$ImageBatch(development, rows, augmentation)
                         else image_training$ImageBatch(development, rows)
                logits <- model(params, state, batch$first, TRUE)
                objective <- image_training$WithL2(
                    image_training$TrainingCrossEntropy(logits, batch$second, cfg$classes, cfg$label_smoothing),
                    params, cfg$l2)
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
        cat(sprintf("  selected epoch=%d; test NLL=%s; accuracy=%s\n", best_epoch, g6(result$loss),
                    g6(result$accuracy)))
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
    }
    cat("All SqueezeNet checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
