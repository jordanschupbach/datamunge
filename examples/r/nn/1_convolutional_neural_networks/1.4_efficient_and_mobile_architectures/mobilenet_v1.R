# Native R NN example. MobileNet v1 cost/Table-2/gradient checks and alpha 1 / 0.5 CIFAR-10 training; the C++ example's loop, written out in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/mobilenet_v1.R
# MobileNet v1: depthwise separable convolutions with width and resolution multipliers.
#
# Port of tflow's mobilenet_v1. Measure every layer's output size from a
# forward pass and tabulate multiply-adds and parameters. Check the paper's
# Tables 6 and 7 exactly as counted here (alpha 1/.75/.5/.25 at 224:
# 569/325/149/41M mult-adds and 4.2/2.6/1.3/0.5M parameters; 224/192/160/128
# pixels: 569/418/290/186M), the full-convolution baseline of Table 4 (29.3M
# parameters, 4866M mult-adds without the classifier) and Table 2's breakdown
# by layer type (1x1: 94.86% of mult-adds, 74.59% of weights). Table 2 prints
# 1.19% for the first convolution, but 1.91% is what makes the shares sum to
# 100%. Check every parameter derivative of the full model in training mode
# (moving averages not updated), with shifted BN betas: at beta=0, ReLU6 is
# positively homogeneous below 6, so a pointwise BN's scale reaches the next
# training-mode BN only as a removable per-channel factor and its derivative
# nearly vanishes.
# https://arxiv.org/abs/1704.04861
# https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet_v1.py
#
# Train alpha 1 and alpha 0.5 on CIFAR-10 with a stride-one first convolution
# (all thirteen blocks and their four remaining downsamplings). Use 12k fit /
# 5k validation, twelve epochs for alpha 1 and twenty-four for alpha 0.5,
# crop/flip, Adam(.001), kernel L2(.0001) and the best-validation checkpoint
# before evaluating all 10k test images. These are CIFAR adaptations, not the
# ImageNet benchmark. Initialization uses nn's generator (He-uniform, as
# tflow), so trained numbers match TF only statistically. CIFAR-10 is read
# from the tflow dataset cache (common/paths.hpp) or data=.
# Options: alpha=1|0.5 (default both), data=PATH epochs=N batch=128 lr=.001
# fit=12000 augment=0|1 threads=N seed=17.
#
# The model components (mobilenet::MobileNetV1 and its config, image batching,
# cross-entropy and L2) are the C++ ones through SWIG (datamungerNN::nn_components);
# the checks and the experiment workflow below mirror
# examples/cpp/source/nn/.../mobilenet_v1.cpp line by line.
suppressPackageStartupMessages(library(datamungerNN))
check <- nn_components$check
example_runtime <- nn_components$example_runtime
image_training <- nn_components$image_training
mobilenet <- nn_components$mobilenet
paths <- nn_components$paths

# C++ default ostream formatting (six significant digits).
g6 <- function(x) sprintf("%g", x)

# Rounds doubles to float32 (vectorized), as the C++ float arithmetic does.
f32 <- function(values) nn$from_vector(values, length(values))$values()

# ImageTrainingConfig is passed by value to each width in the C++; copy every field.
copy_config <- function(cfg) {
    copy <- image_training$ImageTrainingConfig()
    copy$name <- cfg$name
    copy$classes <- cfg$classes
    copy$epochs <- cfg$epochs
    copy$batch <- cfg$batch
    copy$validation_rows <- cfg$validation_rows
    copy$maximum_fit_rows <- cfg$maximum_fit_rows
    copy$learning_rate <- cfg$learning_rate
    copy$l2 <- cfg$l2
    copy$label_smoothing <- cfg$label_smoothing
    copy$augment <- cfg$augment
    copy$minimum_test_accuracy <- cfg$minimum_test_accuracy
    copy$seed <- cfg$seed
    copy
}

# image_training::ParseCifarOption; returns list(known, directory) (directory NULL when unchanged).
parse_cifar_option <- function(key, value, cfg) {
    directory <- NULL
    if (key == "data") directory <- value
    else if (key == "epochs") cfg$epochs <- as.integer(value)
    else if (key == "batch") cfg$batch <- as.integer(value)
    else if (key == "lr") cfg$learning_rate <- as.numeric(value)
    else if (key == "fit") cfg$maximum_fit_rows <- as.integer(value)
    else if (key == "augment" && value %in% c("0", "1")) cfg$augment <- value == "1"
    else if (key == "threads") nn$set_num_threads(as.integer(value))
    else if (key == "seed") cfg$seed <- as.numeric(value)
    else return(list(known = FALSE, directory = directory))
    list(known = TRUE, directory = directory)
}

# Multiply-adds from measured output sizes; weight shares exclude BN and the classifier bias.
# Returns list(params, state, macs, mac_share, weight_share) (the shares are named numeric vectors).
measure <- function(cfg, resolution) {
    p <- nn$ParameterStore(17L)
    state <- nn$ParameterStore(17L)
    nn$no_grad(function() {
        net <- mobilenet$MobileNetV1(p, state, nn$zeros(c(1L, resolution, resolution, 3L)), cfg)
        params <- p$count()
        state_count <- state$count()
        macs <- numeric()
        weights <- numeric()
        add <- function(table, kind, value) {
            table[kind] <- (if (kind %in% names(table)) table[[kind]] else 0) + value
            table
        }
        layers <- net$layers
        for (i in seq_len(layers$size()) - 1L) {
            layer <- layers$`__getitem__`(i)
            name <- layer$name
            kind <- if (name == "conv0") "conv 3x3"
                    else if (grepl("depthwise", name, fixed = TRUE)) "depthwise 3x3"
                    else if (grepl("pointwise", name, fixed = TRUE)) "conv 1x1"
                    else "conv 3x3"
            macs <- add(macs, kind, layer$Macs())
            weights <- add(weights, kind, p$at(paste0(name, "/W"))$numel())
        }
        macs["classifier"] <- net$classifier_macs
        weights["classifier"] <- p$at("output/W")$numel()
        total_macs <- sum(macs)
        total_weights <- sum(weights)
        list(params = params, state = state_count, macs = total_macs,
             mac_share = 100 * macs / total_macs, weight_share = 100 * weights / total_weights)
    })
}

cost_checks <- function() {
    cases <- list(
        list(1, 224L, 4231976, 21888, 568740352),
        list(.75, 224L, 2585560, 16416, 325400448),
        list(.5, 224L, 1331592, 10944, 149497088),
        list(.25, 224L, 470072, 5472, 41030272),
        list(1, 192L, 4231976, 21888, 418121728),
        list(1, 160L, 4231976, 21888, 290675200),
        list(1, 128L, 4231976, 21888, 186400768))
    for (cs in cases) {
        alpha <- cs[[1L]]
        resolution <- cs[[2L]]
        cfg <- mobilenet$MobileNetV1Config()
        cfg$alpha <- alpha
        cost <- measure(cfg, resolution)
        check$Require(cost$params == cs[[3L]] && cost$state == cs[[4L]] && cost$macs == cs[[5L]],
                      "MobileNet v1 parameters and multiply-adds")
        cat(sprintf("  alpha %s, %d px: %.1fM mult-adds, %.2fM parameters (%.0f, state %.0f)\n", g6(alpha), resolution,
                    cost$macs / 1e6, cost$params / 1e6, cost$params, cost$state))
        if (alpha == 1 && resolution == 224L) {
            # std::map iteration order: kinds sorted bytewise.
            table2 <- list("classifier" = c(.18, 24.33), "conv 1x1" = c(94.86, 74.59), "conv 3x3" = c(1.91, .02),
                           "depthwise 3x3" = c(3.06, 1.06))
            line <- "   "
            for (kind in names(table2)) {
                expected <- table2[[kind]]
                check$Near(cost$mac_share[[kind]], expected[1L], .005, paste0("Table 2 mult-add share: ", kind))
                check$Near(cost$weight_share[[kind]], expected[2L], .005, paste0("Table 2 parameter share: ", kind))
                line <- paste0(line, sprintf(" %s %.2f%% / %.2f%%;", kind, cost$mac_share[[kind]], cost$weight_share[[kind]]))
            }
            cat(line, " (mult-adds / weights)\n", sep = "")
        }
    }
    dense <- mobilenet$MobileNetV1Config()
    dense$full_convolutions <- TRUE
    cost <- measure(dense, 224L)
    classifier <- 1024 * 1000
    check$Require(cost$params == 29294088 && cost$macs - classifier == 4866269184, "full-convolution MobileNet")
    cat(sprintf("  full 3x3 convolutions: %.1fM mult-adds without the classifier, %.1fM parameters (%.2fx and %.2fx MobileNet)\n",
                (cost$macs - classifier) / 1e6, cost$params / 1e6, cost$macs / 568740352, cost$params / 4231976))
}

gradient_checks <- function() {
    p <- nn$ParameterStore(17L)
    state <- nn$ParameterStore(17L)
    cfg <- mobilenet$MobileNetV1Config()
    cfg$update_stats <- FALSE
    # pixels[i] = std::sin(.13f * float(i)) in float arithmetic.
    pixels <- f32(sin(f32(f32(.13) * (seq_len(2 * 224 * 224 * 3) - 1))))
    x <- nn$from_vector(pixels, c(2L, 224L, 224L, 3L))
    nn$no_grad(function() {  # create the parameters
        mobilenet$MobileNetV1(p, state, x, cfg, TRUE)
        # At initialization (beta=0) ReLU6 is positively homogeneous below 6, so a
        # pointwise BN's scale passes through ReLU6 and the per-channel depthwise
        # convolution into the next training-mode BN, which removes it: those gamma
        # derivatives nearly vanish. Shifted betas, as after training, avoid that.
        for (name in p$names())
            if (nchar(name) > 5 && endsWith(name, "/beta")) {
                numel <- p$at(name)$numel()
                beta <- f32(.5 * f32(sin(f32(f32(f32(.7) * (seq_len(numel) - 1)) + nchar(name)))))
                p$set(name, nn$from_vector(beta, as.integer(p$at(name)$shape())))
            }
    })
    net <- mobilenet$MobileNetV1(p, state, x, cfg, TRUE)
    check$Require(identical(as.numeric(net$features$shape()), c(2, 7, 7, 1024)) &&
                      identical(as.numeric(net$logits$shape()), c(2, 1000)),
                  "MobileNet output dimensions")
    gradients <- nn$grad(nn$mean(nn$square(net$logits)), p$parameters())
    names <- p$names()
    minimum <- Inf
    maximum <- 0
    smallest <- ""
    for (i in seq_along(gradients)) {
        norm <- nn$sqrt(nn$sum(nn$square(gradients[[i]])))$item()
        check$Require(is.finite(norm), "finite MobileNet derivative")
        if (norm < minimum) {
            minimum <- norm
            smallest <- names[i]
        }
        maximum <- max(maximum, norm)
    }
    check$Require(minimum > 1e-4 * maximum, "clearly nonzero MobileNet derivatives with shifted BN")
    cat(sprintf("  alpha 1 at 224 with shifted BN betas: 7x7x1024 features; training-mode gradient norms [%s,%s], smallest %s\n",
                g6(minimum), g6(maximum), smallest))
}

# Fits one CIFAR MobileNet v1 width: fixed development split, Adam on
# cross-entropy plus L2, validation-NLL checkpoint selection, one test pass.
# Without an explicit epochs= budget, alpha 1 trains twelve epochs and alpha
# 0.5 twenty-four.
train_compact_mobilenet_v1 <- function(alpha, development, test, cfg, explicit_epochs) {
    cfg <- copy_config(cfg)
    cfg$name <- if (alpha == 1) "mobilenet_v1_cifar10" else "mobilenet_v1_050_cifar10"
    if (!explicit_epochs) cfg$epochs <- if (alpha == 1) 12L else 24L
    expected <- if (alpha == 1) 3217226 else 823722
    model <- function(params, state, image, training) {
        options <- mobilenet$MobileNetV1Config()
        options$alpha <- alpha
        options$classes <- 10L
        options$small_stem <- TRUE
        logits <- mobilenet$MobileNetV1(params, state, image, options, training)$logits
        check$Require(params$count() == expected, "CIFAR MobileNet parameter count")
        logits
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
    invisible(NULL)
}

main <- function(arguments = commandArgs(TRUE)) {
    widths <- c(1, .5)
    directory <- paths$Dataset("cifar10")
    cfg <- image_training$ImageTrainingConfig()
    cfg$maximum_fit_rows <- 12000
    cfg$minimum_test_accuracy <- .4
    explicit_epochs <- FALSE
    for (argument in arguments) {
        if (argument == "alpha=1") widths <- 1
        else if (argument == "alpha=0.5") widths <- .5
        else if (startsWith(argument, "alpha=")) stop("alpha must be 1 or 0.5")
        else {
            parts <- regmatches(argument, regexpr("=", argument), invert = TRUE)[[1L]]
            if (length(parts) != 2L) stop("expected key=value option")
            key <- parts[1L]
            value <- parts[2L]
            explicit_epochs <- explicit_epochs || key == "epochs"
            # Parsed before the checks so threads= applies to them too.
            parsed <- parse_cifar_option(key, value, cfg)
            if (!parsed$known) stop("unknown option: ", argument)
            if (!is.null(parsed$directory)) directory <- parsed$directory
        }
    }
    cost_checks()
    gradient_checks()
    data <- image_training$LoadCifar10(directory)
    development <- data$first
    test <- data$second
    for (alpha in widths) train_compact_mobilenet_v1(alpha, development, test, cfg, explicit_epochs)
    cat("All MobileNet v1 checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
