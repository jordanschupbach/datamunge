# Native R NN example. Stochastic depth: DropPath mask/schedule/shortcut checks, then constant-depth versus stochastic-depth 20-layer CIFAR-10 training; the C++ example's loop, written out in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.3_residual_and_dense_families/stochastic_depth.R
# Stochastic depth: survival schedules, masking conventions and learning.
#
# Port of tflow's stochastic_depth. Verify original train=bF/test=pF and
# modern inverted train=bF/p/test=F conventions, whole-batch and per-example
# masks, exact input/weight derivatives, fresh masks, empirical survival rates
# and p=0/1 boundaries. The schedule p_l=1-l/L*(1-p_last) has expected active
# depth (3L-1)/4 when p_last=.5. Check the 110-layer CIFAR architecture with
# all gates open, including its average-pool/zero-pad shortcuts and all
# parameter gradients.
# https://arxiv.org/abs/1603.09382
# https://github.com/pytorch/vision/blob/main/torchvision/ops/stochastic_depth.py
#
# Compare two 20-layer, 269,722-parameter CIFAR models: constant depth and
# inverted whole-batch DropPath with final survival .5. Use the same seed,
# 12k fit / 5k validation split, twelve epochs, crop/flip, Adam(.001), kernel
# L2(.0001), and independent best-validation checkpoints before test scoring.
# Every residual is computed and every BN statistic updated; this does not
# implement the original conditional execution or claim speedup.
# Bias-free convolutions and the short optimizer schedule are adaptations.
# One paired run illustrates the procedure, not a general accuracy claim.
# Deviations: masks are drawn from one nn::Generator per model (tflow seeds
# each block's RandomUniform with 71+index); weights come from nn's generator
# (seed=), so trained numbers match TF only statistically. The 110-layer
# gradient check disables the moving-average updates.
# Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.
#
# The components (stochastic_depth::DropPath, SurvivalProbabilities,
# CifarIdentityShortcut and StochasticCifarResNet, image batching, cross-entropy
# and L2) are the C++ ones through SWIG (datamungerNN::nn_components); the checks
# and the experiment workflow below mirror examples/cpp/source/nn/.../stochastic_depth.cpp
# line by line.
suppressPackageStartupMessages(library(datamungerNN))
check <- nn_components$check
example_runtime <- nn_components$example_runtime
image_training <- nn_components$image_training
paths <- nn_components$paths
sd <- nn_components$stochastic_depth

# C++ default ostream formatting (six significant digits).
g6 <- function(x) sprintf("%g", x)

# The C++ `float` literal: round doubles to single precision.
f32 <- function(x) readBin(writeBin(as.numeric(x), raw(), size = 4L), "numeric", n = length(x), size = 4L)

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

# Whole-batch or per-row DropPath masks: Bernoulli gates, exact input/weight
# derivatives, deterministic inference scaling and the empirical survival rate.
mask_checks <- function(per_example, scaling, survival) {
    p <- nn$ParameterStore(17)
    gen <- nn$Generator(73L)
    rows <- if (per_example) 8192L else 4L
    repeats <- if (per_example || survival == 0 || survival == 1) 1L else 512L
    i <- 0:(rows * 8L - 1L)
    # .2f + .1f * sin(.17f * i), evaluated in single precision as the C++ does.
    values <- f32(f32(.2) + f32(f32(.1) * f32(sin(f32(f32(.17) * i)))))
    x <- nn$from_vector(values, c(rows, 2L, 2L, 2L))
    invisible(x$requires_grad_(TRUE))
    p$set("weight", nn$scalar(1.5))
    weight <- p$at("weight")
    inverted <- scaling == sd$DropPathScaling_kInverted
    alive <- 0
    samples <- 0
    error <- 0
    for (repeat_index in seq_len(repeats)) {
        branch <- nn$mul(x, weight)
        train <- sd$DropPath(branch, survival, TRUE, per_example, scaling, gen)
        infer <- sd$DropPath(branch, survival, FALSE, per_example, scaling, gen)
        output <- nn$add(x, train$value)
        gradients <- nn$grad(nn$sum(output), nn$tensor_vector(list(x, weight)))
        out <- output$values()
        gate <- train$gate$values()
        dx <- gradients[[1L]]$values()
        inference <- infer$value$values()
        scalar_gate <- length(gate) == 1L
        weight_derivative <- 0
        for (row in 0:(rows - 1L)) {
            gate_value <- gate[if (scalar_gate) 1L else row + 1L]
            check$Require(gate_value == 0 || gate_value == 1, "Bernoulli gate")
            if (per_example || row == 0L) {
                alive <- alive + gate_value
                samples <- samples + 1
            }
            multiplier <- if (inverted && survival > 0) gate_value / survival else gate_value
            for (item in 0:7) {
                index <- row * 8L + item + 1L
                value <- values[index]
                check$Near(out[index], value * (1 + 1.5 * multiplier), 3e-6,
                           "identity plus uniformly masked row/whole batch")
                check$Near(dx[index], 1 + 1.5 * multiplier, 3e-6, "identity plus masked branch input derivative")
                check$Near(inference[index], 1.5 * value * (if (inverted) 1 else survival), 3e-6,
                           "deterministic inference scaling")
                weight_derivative <- weight_derivative + multiplier * value
            }
        }
        error <- max(error, abs(gradients[[2L]]$item() - weight_derivative) / max(1, abs(weight_derivative)))
    }
    check$Near(error, 0, 3e-5, "masked parameter derivative including zero branch gradient")
    empirical <- alive / samples
    check$Near(empirical, survival, 4 * sqrt(survival * (1 - survival) / samples) + .002,
               "empirical survival and expected branch multiplier")
    if (survival > 0 && survival < 1)
        check$Require(alive > 0 && alive < samples, "both active and inactive masks observed")
    cat(sprintf("  %s/%s, p=%s: empirical=%s, derivative error=%s\n", if (per_example) "row" else "batch",
                if (inverted) "inverted" else "original", g6(survival), g6(empirical), g6(error)))
}

# The linear survival schedule's expected active depth and the option-A
# (average-pool, zero-padded) CIFAR shortcut with its derivative.
schedule_and_shortcut_checks <- function() {
    survival <- sd$SurvivalProbabilities(54L, .5)
    check$Near(survival[1L], 1 - .5 / 54, 1e-7, "first scheduled survival")
    check$Near(survival[54L], .5, 0, "last scheduled survival")
    expected <- 0
    for (probability in survival) expected <- expected + probability
    check$Near(expected, 40.25, 2e-6, "expected active blocks in 110-layer model")
    values <- as.numeric(1:32)
    x <- nn$from_vector(values, c(1L, 4L, 4L, 2L))
    invisible(x$requires_grad_(TRUE))
    shortcut <- sd$CifarIdentityShortcut(x, 4L, 2L)
    gradient <- nn$grad(nn$sum(shortcut), nn$tensor_vector(list(x)))[[1L]]$values()
    check$Require(identical(as.numeric(shortcut$shape()), c(1, 2, 2, 4)), "CIFAR shortcut spatial/channel dimensions")
    out <- shortcut$values()
    for (r in 0:1)
        for (c in 0:1)
            for (channel in 0:3) {
                value <- 0
                if (channel < 2)
                    for (dr in 0:1)
                        for (dc in 0:1)
                            value <- value + .25 * values[((2 * r + dr) * 4 + 2 * c + dc) * 2 + channel + 1L]
                check$Near(out[(r * 2 + c) * 4 + channel + 1L], value, 0, "host average-pool and channel padding")
            }
    for (i in 1:32) check$Near(gradient[i], .25, 0, "average-pool shortcut derivative")
    cat(sprintf("  schedule: 54 possible blocks, %s expected active; option-A shortcut checks passed\n", g6(expected)))
}

# The 110-layer CIFAR model with every gate open: counts, stage shapes and
# finite nonzero derivatives through every parameter (moving averages frozen).
full_checks <- function() {
    p <- nn$ParameterStore(17)
    state <- nn$ParameterStore(17)
    gen <- nn$Generator(71L)
    cfg <- sd$StochasticCifarConfig()
    cfg$last_survival <- 1
    cfg$update_stats <- FALSE
    i <- 0:(2L * 32L * 32L * 3L - 1L)
    # .1f * sin(i * .13f), evaluated in single precision as the C++ does.
    pixels <- f32(f32(.1) * f32(sin(f32(i * f32(.13)))))
    net <- sd$StochasticCifarResNet(p, state, nn$from_vector(pixels, c(2L, 32L, 32L, 3L)), cfg, TRUE, gen)
    stages <- nn$tensor_list(net$stages)
    check$Require(p$count() == 1727962 && state$count() == 8096 && length(nn$tensor_list(net$gates)) == 54L,
                  "110-layer CIFAR parameter/state/block counts")
    for (stage in 0:2)
        check$Require(identical(as.numeric(stages[[stage + 1L]]$shape()),
                                c(2, bitwShiftR(32L, stage), bitwShiftR(32L, stage), bitwShiftL(16L, stage))),
                      "CIFAR residual stage dimensions")
    check$Require(identical(as.numeric(net$logits$shape()), c(2, 10)), "CIFAR residual classifier dimensions")
    gradients <- nn$grad(nn$mean(nn$square(net$logits)), p$parameters())
    names <- as.character(p$names())
    minimum <- Inf
    maximum <- 0
    for (i in seq_along(gradients)) {
        norm <- nn$sqrt(nn$sum(nn$square(gradients[[i]])))$item()
        check$Require(is.finite(norm) && norm > 0, paste0("finite nonzero residual derivative: ", names[i]))
        minimum <- min(minimum, norm)
        maximum <- max(maximum, norm)
    }
    cat(sprintf("  110-layer CIFAR model with open gates: parameters=%.0f, gradient norms [%s,%s]\n", p$count(),
                g6(minimum), g6(maximum)))
}

# One 20-layer comparison model (constant depth when survival is 1, inverted
# whole-batch DropPath otherwise): CIFAR-10 options, fixed development split,
# Adam on cross-entropy plus L2, validation-NLL checkpoint selection, one test pass.
train_depth20 <- function(survival, arguments) {
    cfg <- image_training$ImageTrainingConfig()
    cfg$name <- if (survival == 1) "constant_depth20_cifar10" else "stochastic_depth20_cifar10"
    cfg$maximum_fit_rows <- 12000L
    cfg$epochs <- 12L
    cfg$minimum_test_accuracy <- .45
    gen <- nn$Generator(71L)  # one mask stream per model, persistent across steps
    model <- function(p, state, image, training) {
        model_cfg <- sd$StochasticCifarConfig()
        model_cfg$depth <- 20L
        model_cfg$last_survival <- survival
        logits <- sd$StochasticCifarResNet(p, state, image, model_cfg, training, gen)$logits
        check$Require(p$count() == 269722 && state$count() == 1376, "20-layer comparison model parameter/state counts")
        logits
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
    for (argument in arguments)  # threads= also applies to the checks
        if (startsWith(argument, "threads=")) nn$set_num_threads(as.integer(substring(argument, 9L)))
    for (scaling in list(sd$DropPathScaling_kOriginal, sd$DropPathScaling_kInverted)) {
        mask_checks(FALSE, scaling, f32(.6))
        mask_checks(TRUE, scaling, f32(.6))
        mask_checks(FALSE, scaling, 0)
        mask_checks(TRUE, scaling, 1)
    }
    schedule_and_shortcut_checks()
    full_checks()
    for (survival in c(1, .5)) train_depth20(survival, arguments)
    cat("All stochastic-depth checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
