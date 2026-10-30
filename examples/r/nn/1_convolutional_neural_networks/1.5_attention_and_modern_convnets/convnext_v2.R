# Native R NN example. ConvNeXt V2: GRN formula/invariance/derivative checks, the eight released sizes' costs, V2-Tiny derivatives and compact V2-Atto CIFAR-10 training; the C++ example's loop, written out in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.5_attention_and_modern_convnets/convnext_v2.R
# ConvNeXt V2: global response normalization in place of layer scale.
#
# Port of tflow's convnext_v2 (model: common/convnext.hpp). Check global
# response normalization against its formula (per-channel spatial L2 norms
# divided by their channel mean, gamma * x * N + beta + x), that it is the
# identity at initialization (gamma = beta = 0), that its normalized
# responses are invariant to a global rescaling of the input, and that an
# all-zero input or a dead channel keeps finite, identity input derivatives.
# Measure all eight released sizes at 224 pixels: Atto 3,708,400 learned
# parameters (0.55G multiply-adds), Femto 5,233,240 (0.78G), Pico 9,066,280
# (1.37G), Nano 15,623,800 (2.45G), Tiny 28,635,496 (4.46G), Base 88,717,800
# (15.35G), Large 197,956,840 (34.36G) and Huge 660,289,640 (114.92G), the
# paper's 3.7/5.2/9.1/15.6/28.6/89/198/660M and 0.55/0.78/1.37/2.45/4.47/
# 15.4/34.4/115G. As in tflow the multiply-adds follow from the fixed stage
# sizes 56/28/14/7; the parameters are counted by building each model (here
# eagerly on a 32-pixel input, since counts do not depend on the image size;
# Huge allocates its 2.6 GB of weights briefly). Check every derivative of
# V2-Tiny in training mode with active GRN (nonzero gamma and beta). The
# masked-autoencoder (FCMAE) pretraining that motivated GRN is not
# implemented here.
# https://arxiv.org/abs/2301.00808
# https://github.com/facebookresearch/ConvNeXt-V2
#
# Train V2-Atto on CIFAR-10 exactly as convnext trains V1-Atto (2x2/2 stem,
# stochastic depth 0.1), so the two runs compare layer scale with GRN. Use
# 12k fit / 5k validation, twelve epochs, crop/flip, Adam(.001), kernel
# L2(.0001) and the best-validation checkpoint before evaluating all 10k
# test images. Weights and drop-path masks come from nn's generator, so
# trained numbers match TF only statistically. CIFAR-10 is read from the
# tflow dataset cache (common/paths.hpp) or data=.
# Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.
#
# The components (convnext::GlobalResponseNorm, ConvNeXtLayout and ConvNeXt,
# mobile_checks::Pattern and GradientNorms, image batching, cross-entropy and
# L2) are the C++ ones through SWIG (datamungerNN::nn_components); the checks
# and the experiment workflow below mirror examples/cpp/source/nn/.../convnext_v2.cpp
# line by line.
suppressPackageStartupMessages(library(datamungerNN))
check <- nn_components$check
example_runtime <- nn_components$example_runtime
image_training <- nn_components$image_training
paths <- nn_components$paths
convnext <- nn_components$convnext
mobile_checks <- nn_components$mobile_checks
Pattern <- mobile_checks$Pattern

# C++ default ostream formatting (six significant digits).
g6 <- function(x) sprintf("%g", x)

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

# Global response normalization: identity at initialization, the formula with
# active gamma/beta, scale invariance of the responses, and finite identity
# input derivatives for an all-zero input and a dead channel.
grn_checks <- function() {
    size <- 5L
    channels <- 12L
    p <- nn$ParameterStore(17)
    value <- Pattern(c(2L, size, size, channels), 1, .2)
    identity <- NULL
    nn$no_grad(function() {
        identity <<- nn$max(nn$abs(nn$sub(convnext$GlobalResponseNorm(p, "grn", value)$output, value)))$item()
        p$set("grn/gamma", Pattern(c(channels), .8, 1))
        p$set("grn/beta", Pattern(c(channels), .3, 2))
    })
    grn <- convnext$GlobalResponseNorm(p, "grn", value)
    out <- grn$output$values()
    response <- grn$response$values()
    scaled <- convnext$GlobalResponseNorm(p, "grn", nn$mul(value, 7))$response$values()
    gamma <- p$at("grn/gamma")$values()
    beta <- p$at("grn/beta")$values()
    v <- value$values()
    error <- 0
    invariance <- 0
    for (n in 0:1) {
        norm <- numeric(channels)
        mean <- 0
        for (i in 0:(size * size - 1L))
            for (c in 0:(channels - 1L))
                norm[c + 1L] <- norm[c + 1L] + v[(n * size * size + i) * channels + c + 1L]^2
        for (c in seq_len(channels)) {
            norm[c] <- sqrt(norm[c])
            mean <- mean + norm[c] / channels
        }
        for (i in 0:(size * size - 1L))
            for (c in 0:(channels - 1L)) {
                k <- (n * size * size + i) * channels + c + 1L
                expected <- gamma[c + 1L] * v[k] * norm[c + 1L] / (mean + 1e-6) + beta[c + 1L] + v[k]
                error <- max(error, abs(out[k] - expected))
            }
        for (c in 0:(channels - 1L))
            invariance <- max(invariance, abs(response[n * channels + c + 1L] - scaled[n * channels + c + 1L]))
    }
    check$Near(identity, 0, 0, "GRN is the identity at initialization")
    check$Near(error, 0, 1e-5, "global response normalization")
    check$Near(invariance, 0, 1e-5, "GRN responses are invariant to input scale")
    input_gradient <- function(tensor) {
        x <- tensor$detach()$requires_grad_(TRUE)
        nn$grad(nn$sum(convnext$GlobalResponseNorm(p, "grn", x)$output), nn$tensor_vector(list(x)))[[1L]]$values()
    }
    zero <- input_gradient(nn$zeros(c(2L, size, size, channels)))
    for (gradient in zero) check$Near(gradient, 1, 0, "all-zero GRN has finite identity input derivative")
    mixed <- v
    for (i in 0:(2L * size * size - 1L)) mixed[i * channels + 1L] <- 0
    dead <- input_gradient(nn$from_vector(mixed, c(2L, size, size, channels)))
    for (gradient in dead) check$Require(is.finite(gradient), "finite derivatives with a dead channel")
    for (i in 0:(2L * size * size - 1L))
        check$Near(dead[i * channels + 1L], 1, 0, "dead channel retains identity derivative")
    cat("  all-zero and single-zero-channel GRN gradients remain finite with identity derivative\n")
    cat(sprintf(paste0("  GRN (12 channels): identity at initialization; output matches gamma*x*N+beta+x (error %s); ",
                       "responses unchanged when the input is scaled by 7 (difference %s)\n"), g6(error), g6(invariance)))
}

# Parameters (built eagerly on a 32-pixel input) and multiply-adds at 224
# pixels (fixed stage sizes 56/28/14/7) of the eight released V2 sizes.
cost_checks <- function() {
    cases <- list(list("atto", 3708400, 547332480), list("femto", 5233240, 779629824),
                  list("pico", 9066280, 1367055360), list("nano", 15623800, 2445528320),
                  list("tiny", 28635496, 4455531264), list("base", 88717800, 15354729472),
                  list("large", 197956840, 34361433600), list("huge", 660289640, 114922100480))
    for (case in cases) {
        size <- case[[1L]]
        layout <- convnext$ConvNeXtLayout(size)
        params <- NULL
        p <- nn$ParameterStore(17)
        nn$no_grad(function() {
            cfg <- convnext$ConvNeXtConfig()
            cfg$size <- size
            cfg$v2 <- TRUE
            convnext$ConvNeXt(p, nn$zeros(c(1L, 32L, 32L, 3L)), cfg)
            params <<- p$count()
        })
        rm(p)
        invisible(gc())  # Huge's 2.6 GB of weights live only for the count
        # Stage sizes are fixed by the stem and stride-two joins: 56/28/14/7.
        depths <- as.numeric(layout$depths)
        dims <- as.numeric(layout$dims)
        macs <- dims[length(dims)] * 1000
        n <- 56
        for (stage in seq_along(depths)) {
            d <- dims[stage]
            macs <- macs + depths[stage] * n * n * (49 * d + 8 * d * d) +
                (if (stage > 1L) n * n * 4 * dims[stage - 1L] * d else n * n * 48 * d)
            n <- n / 2
        }
        check$Require(params == case[[2L]] && macs == case[[3L]], "ConvNeXt V2 parameters and multiply-adds")
        cat(sprintf("  ConvNeXt V2-%s%s: %.0f parameters, %.2fG mult-adds\n", toupper(substr(size, 1L, 1L)),
                    substring(size, 2L), params, macs / 1e9))
    }
}

# Every derivative of V2-Tiny in training mode with active GRN (nonzero gamma and beta).
gradient_checks <- function() {
    p <- nn$ParameterStore(17)
    cfg <- convnext$ConvNeXtConfig()
    cfg$v2 <- TRUE
    nn$no_grad(function() {
        convnext$ConvNeXt(p, nn$zeros(c(1L, 32L, 32L, 3L)), cfg)  # create the parameters
        for (name in as.character(p$names()))
            if (grepl("/grn/", name, fixed = TRUE))
                p$set(name, Pattern(as.integer(p$at(name)$shape()), .5, nchar(name)))
    })
    net <- convnext$ConvNeXt(p, Pattern(c(2L, 224L, 224L, 3L), 1, 0), cfg, TRUE)
    features <- as.numeric(net$features$shape())
    check$Require(features[2L] == 7 && features[4L] == 768, "ConvNeXt V2-Tiny 7x7x768 features")
    norms <- as.numeric(mobile_checks$GradientNorms(p, nn$mean(nn$square(net$logits))))
    names <- as.character(p$names())
    minimum <- Inf
    maximum <- 0
    smallest <- ""
    for (i in seq_along(norms)) {
        check$Require(is.finite(norms[i]), "finite ConvNeXt V2 derivative")
        if (norms[i] < minimum) {
            minimum <- norms[i]
            smallest <- names[i]
        }
        maximum <- max(maximum, norms[i])
    }
    check$Require(minimum > 1e-6 * maximum, "clearly nonzero ConvNeXt V2 derivatives")
    cat(sprintf("  ConvNeXt V2-Tiny with active GRN: %d derivatives in [%s,%s] (smallest %s)\n", length(norms),
                g6(minimum), g6(maximum), smallest))
}

main <- function(arguments = commandArgs(TRUE)) {
    for (argument in arguments)  # threads= also applies to the checks
        if (startsWith(argument, "threads=")) nn$set_num_threads(as.integer(substring(argument, 9L)))
    grn_checks()
    cost_checks()
    gradient_checks()
    cfg <- image_training$ImageTrainingConfig()
    cfg$name <- "convnext_v2_atto_cifar10"
    cfg$maximum_fit_rows <- 12000L
    cfg$epochs <- 12L
    cfg$minimum_test_accuracy <- .4
    options <- parse_cifar_options(arguments, cfg)
    explicit_epochs <- options$explicit_epochs
    data <- image_training$LoadCifar10(options$directory)
    development <- data$first
    test <- data$second
    model <- function(p, state, image, training) {
        architecture <- convnext$ConvNeXtConfig()
        architecture$size <- "atto"
        architecture$v2 <- TRUE
        architecture$classes <- 10L
        architecture$cifar_stem <- TRUE
        architecture$drop_path <- .1
        logits <- convnext$ConvNeXt(p, image, architecture, training)$logits
        check$Require(p$count() == 3389170 && state$count() == 0, "CIFAR ConvNeXt V2-Atto parameter count")
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
    cat("All ConvNeXt V2 checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
