# Native R NN example. Wide ResNet dropout/full-model checks and three WRN-16 CIFAR-10 trainings; the C++ example's loop, written out in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.3_residual_and_dense_families/wide_resnet.R
# Wide ResNet: width scaling and dropout inside the residual branch.
#
# Port of tflow's wide_resnet. Implement the authors' CIFAR 6n+4 layout with
# preactivated basic blocks, fixed 16-channel stem and 16k/32k/64k stages.
# Dropout follows the second BN/ReLU and precedes the second convolution;
# shortcuts remain undropped. Check WRN-16-1, WRN-16-2 and the full
# 36,479,194-parameter WRN-28-10, including stage shapes and derivatives
# through every learned parameter. Count convolution/classifier MACs for one
# image; BN, ReLU, additions and pooling are excluded. These operation counts
# are not measured latency.
# https://arxiv.org/abs/1605.07146
# https://github.com/szagoruyko/wide-residual-networks/blob/master/models/wide-resnet.lua
#
# Controlled checks verify inverted dropout values and derivatives, repeatable
# inference and signed identity paths even with training dropout enabled.
# Train three WRN-16 configurations on CIFAR-10: width 1 without dropout,
# width 2 without dropout, and width 2 with dropout .3. Each uses the same
# 12k fit / 5k validation split, seed, twelve epochs, crop/flip, Adam(.001)
# and kernel L2(.0001). Restore each lowest validation-NLL checkpoint and
# evaluate 10k test images. These single-seed short runs illustrate the
# comparison procedure; they do not establish an architecture ranking or
# reproduce the paper's long SGD schedule. Initialization uses He-uniform.
# Deviations: dropout masks come from one nn::Generator per model (tflow
# seeds each block); weights come from nn's generator (seed=), so trained
# numbers match TF only statistically. Fixture and full-model checks run
# training mode with the moving-average updates disabled (tflow does not run
# its update ops there).
# Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.
#
# The model components (wide_resnet::WideBasicBlock/WideResNet, preact
# normalization, the sine fixtures, image batching, cross-entropy and L2) are
# the C++ ones through SWIG (datamungerNN::nn_components); the checks and the
# workflow below mirror examples/cpp/source/nn/.../wide_resnet.cpp line by line.
suppressPackageStartupMessages(library(datamungerNN))
architecture <- nn_components$architecture
check <- nn_components$check
example_runtime <- nn_components$example_runtime
image_training <- nn_components$image_training
paths <- nn_components$paths
preact <- nn_components$preact
wide_resnet <- nn_components$wide_resnet

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

# preact::NormConfig{training, update_stats, momentum, epsilon}.
norm_config <- function(training, update_stats, momentum, epsilon) {
    norm <- preact$NormConfig()
    norm$training <- training
    norm$update_stats <- update_stats
    norm$momentum <- momentum
    norm$epsilon <- epsilon
    norm
}

dropout_checks <- function() {
    p <- nn$ParameterStore(17L)
    state <- nn$ParameterStore(17L)
    gen <- nn$Generator(17L)
    # sin(.17f * i) in float, i = 0..2047 (architecture::SinePixels computes exactly that).
    x <- architecture$SinePixels(c(8L, 8L, 8L, 4L), 1, .17)
    x$requires_grad_(TRUE)
    values <- x$values()
    train_norm <- norm_config(TRUE, FALSE, .9, 1e-5)
    infer_norm <- norm_config(FALSE, FALSE, .9, 1e-5)
    block <- function(norm, dropout) wide_resnet$WideBasicBlock(p, state, "block", x, 4L, 1L, norm, dropout, gen)
    train <- block(train_norm, .3)
    infer <- block(infer_norm, .3)$output$values()
    no_dropout <- block(infer_norm, 0)$output$values()
    mask_gradient <- nn$grad(nn$sum(train$after_dropout), list(train$before_dropout))[[1L]]$values()
    before <- train$before_dropout$values()
    after <- train$after_dropout$values()
    shortcut <- train$shortcut$values()
    positive <- 0L
    dropped <- 0L
    for (i in seq_along(values)) {
        check$Near(infer[i], no_dropout[i], 0, "dropout rate does not affect inference")
        check$Near(shortcut[i], values[i], 0, "dropout leaves raw shortcut untouched")
        if (before[i] <= 1e-6) next
        positive <- positive + 1L
        removed <- after[i] == 0
        dropped <- dropped + removed
        check$Near(after[i], if (removed) 0 else before[i] / .7, 2e-6, "inverted dropout values")
        check$Near(mask_gradient[i], if (removed) 0 else 1 / .7, 2e-6, "inverted dropout derivatives")
    }
    check$Require(positive > 500 && abs(dropped / positive - .3) < .06, "wide dropout empirical mask rate")
    again_train <- block(train_norm, .3)$after_dropout$values()
    again_infer <- block(infer_norm, .3)$output$values()
    difference <- 0
    for (i in seq_along(values)) {
        check$Near(again_infer[i], infer[i], 0, "repeatable wide inference")
        difference <- difference + abs(again_train[i] - after[i])
    }
    check$Require(difference > 1, "training dropout draws fresh masks")
    p$set("block/conv2/W", nn$zeros(c(3L, 3L, 4L, 4L)))  # zero residual
    train <- block(train_norm, .3)
    output <- train$output$values()
    dx <- nn$grad(nn$sum(train$output), list(x))[[1L]]$values()
    for (i in seq_along(values)) {
        check$Near(output[i], values[i], 0, "signed identity survives residual dropout")
        check$Near(dx[i], 1, 2e-6, "unit identity derivative survives residual dropout")
    }
    cat(sprintf("  dropout values/derivatives, changing masks, deterministic inference and signed identity passed; removed fraction=%s\n",
                g6(dropped / positive)))
}

multiply_accumulates <- function(depth, width) {
    total <- 32 * 32 * 9 * 3 * 16
    in_channels <- 16
    spatial <- 32
    for (stage in 0:2) {
        out <- 16 * 2^stage * width
        if (stage > 0) spatial <- spatial / 2
        for (block in seq_len((depth - 4) %/% 6)) {
            total <- total + spatial * spatial * (9 * in_channels * out + 9 * out * out +
                                                  (if (in_channels != out) in_channels * out else 0))
            in_channels <- out
        }
    }
    total + in_channels * 10
}

full_checks <- function(depth, width, parameters, state_parameters) {
    p <- nn$ParameterStore(17L)
    state <- nn$ParameterStore(17L)
    gen <- nn$Generator(17L)
    cfg <- wide_resnet$WideResNetConfig()
    cfg$depth <- as.integer(depth)
    cfg$widen_factor <- as.integer(width)
    cfg$dropout <- 0
    cfg$update_stats <- FALSE
    # .1f * sin(i * .13f) in float (architecture::SinePixels computes exactly that).
    net <- wide_resnet$WideResNet(p, state, architecture$SinePixels(c(2L, 32L, 32L, 3L), .1, .13), cfg, TRUE, gen)
    check$Require(p$count() == parameters && state$count() == state_parameters, "canonical wide parameter/state counts")
    check$Require(length(nn$tensor_list(net$blocks)) == (depth - 4) %/% 2, "wide residual block count")
    for (stage in 0:3) {
        spatial <- if (stage < 2) 32 else 32 / 2^(stage - 1)
        channels <- if (stage == 0) 16 else 16 * 2^(stage - 1) * width
        shape <- as.numeric((if (stage == 0) net$stem else net$stages(stage - 1L))$shape())
        check$Require(identical(shape, c(2, spatial, spatial, channels)), "canonical wide stage dimensions")
    }
    check$Require(identical(as.numeric(net$logits$shape()), c(2, 10)), "wide classifier dimensions")
    gradients <- nn$grad(nn$mean(nn$square(net$logits)), nn$tensor_list(p$parameters()))
    names <- as.character(p$names())
    minimum <- Inf
    maximum <- 0
    for (i in seq_along(gradients)) {
        norm <- nn$sqrt(nn$sum(nn$square(gradients[[i]])))$item()
        check$Require(is.finite(norm) && norm > 0, paste0("finite nonzero wide derivative: ", names[i]))
        minimum <- min(minimum, norm)
        maximum <- max(maximum, norm)
    }
    cat(sprintf("  WRN-%d-%d: parameters=%.0f, state=%.0f, MACs/image=%.0f; gradient norms [%s,%s]\n", depth, width,
                parameters, state_parameters, multiply_accumulates(depth, width), g6(minimum), g6(maximum)))
}

# One WRN-16 setting (the body of the C++ settings loop): options, CIFAR-10 split,
# Adam on cross-entropy plus L2, validation-NLL checkpoint selection, one test pass.
train_wide_resnet <- function(width, dropout, arguments) {
    cfg <- image_training$ImageTrainingConfig()
    cfg$name <- sprintf("wide_resnet16_%d%s_cifar10", width, if (dropout > 0) "_dropout" else "")
    cfg$maximum_fit_rows <- 12000L
    cfg$epochs <- 12L
    cfg$minimum_test_accuracy <- 0.45
    gen <- nn$Generator(17L)  # one dropout stream per model, persistent across steps
    model <- function(params, state, image, training) {
        model_options <- wide_resnet$WideResNetConfig()
        model_options$depth <- 16L
        model_options$widen_factor <- as.integer(width)
        model_options$dropout <- dropout
        wide_resnet$WideResNet(params, state, image, model_options, training, gen)$logits
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
    dropout_checks()
    full_checks(16L, 1L, 175066, 928)
    full_checks(16L, 2L, 691674, 1824)
    full_checks(28L, 10L, 36479194, 17952)
    for (setting in list(list(1L, 0), list(2L, 0), list(2L, .3)))
        train_wide_resnet(setting[[1L]], setting[[2L]], arguments)
    cat("All Wide ResNet checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
