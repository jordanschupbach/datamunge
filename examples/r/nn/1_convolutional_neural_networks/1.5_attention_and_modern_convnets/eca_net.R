# Native R NN example. eca_net: the C++ checks and training workflow over SWIG components.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.5_attention_and_modern_convnets/eca_net.R
suppressPackageStartupMessages(library(datamungerNN))
check <- nn_components$check
example_runtime <- nn_components$example_runtime
image_training <- nn_components$image_training
paths <- nn_components$paths
attention_cnn <- nn_components$attention_cnn
mobile_checks <- nn_components$mobile_checks

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

module_checks <- function() {
    channels <- 24
    p <- nn$ParameterStore(17)
    x <- mobile_checks$Pattern(c(2L, 4, 4, channels), 1, 0.2)
    net <- attention_cnn$Eca(p, "eca", x, 5L)
    weights <- p$at("eca/W")$values()
    output <- net$output$values()
    values <- x$values()
    error <- 0
    low <- 1
    high <- 0
    for (n in 0:1) {
        means <- vapply(0:(channels - 1L), function(c)
            sum(values[(n * 16 + 0:(16 - 1L)) * channels + c + 1L] / 16), numeric(1))
        gates <- vapply(0:(channels - 1L), function(c) {
            z <- 0
            for (j in 0:4) {
                source <- c + j - 2L
                if (source >= 0 && source < channels) z <- z + weights[j + 1L] * means[source + 1L]
            }
            1 / (1 + exp(-z))
        }, numeric(1))
        for (c in 0:(channels - 1L)) {
            gate <- gates[c + 1L]
            low <- min(low, gate)
            high <- max(high, gate)
            for (i in 0:(16 - 1L)) {
                k <- (n * 16 + i) * channels + c + 1L
                error <- max(error, abs(output[k] - gate * values[k]))
            }
        }
    }
    check$Near(error, 0, 1e-6, "ECA channel attention")
    sizes <- vapply(c(64L, 128L, 256L, 512L, 256L, 512L, 1024L, 2048L), attention_cnn$EcaKernel, integer(1))
    check$Require(all(sizes == c(3, 5, 5, 5, 5, 5, 5, 7)), "adaptive ECA kernel sizes")
    cat(sprintf(paste0("  ECA module (24 channels, k=5): output equals the host-computed 1-D convolution gate (error %s); ",
                      "adaptive k for 64/128/256/512 channels = 3/5/5/5, for 256/512/1024/2048 = 5/5/5/7\n"), g6(error)))
}

full_checks <- function() {
    for (setting in list(FALSE, TRUE)) {
        adaptive <- setting
        p <- nn$ParameterStore(17)
        state <- nn$ParameterStore(17)
        cfg <- attention_cnn$AttentionResNetConfig()
        cfg$attention <- attention_cnn$ChannelAttention_kEca
        cfg$adaptive_eca <- adaptive
        cfg$update_stats <- FALSE
        nn$no_grad(function() attention_cnn$AttentionResNet(p, state, nn$zeros(c(1L, 64L, 64L, 3L)), cfg))
        check$Require(p$count() == if (adaptive) 25557118 else 25557080, "ECA-ResNet-50 parameter count")
        if (!adaptive) {
            cat(sprintf("  ECA-ResNet-50, k=3 (release): %d parameters (+%d)", p$count(), p$count() - 25557032), "\n", sep = "")
            next
        }
        mobile_checks$ShiftBetas(p, FALSE)
        net <- attention_cnn$AttentionResNet(p, state, mobile_checks$Pattern(c(2L, 224L, 224L, 3L), 1, 0), cfg, TRUE)
        norms <- as.numeric(mobile_checks$GradientNorms(p, nn$mean(nn$square(net$logits))))
        minimum <- Inf
        maximum <- 0
        smallest <- ""
        names <- as.character(p$names())
        for (i in seq_along(norms)) {
            norm <- norms[i]
            check$Require(is.finite(norm), "finite ECA-ResNet derivative")
            if (norm < minimum) { minimum <- norm; smallest <- names[i] }
            maximum <- max(maximum, norm)
        }
        check$Require(minimum > 1e-5 * maximum, "clearly nonzero ECA-ResNet derivatives")
        cat(sprintf("  ECA-ResNet-50, adaptive k: %d parameters (+%d); training-mode gradient norms with shifted BN betas [%s,%s] (smallest %s)", p$count(), p$count() - 25557032, g6(minimum), g6(maximum), smallest), "\n", sep = "")
    }
}

main <- function(arguments = commandArgs(TRUE)) {
    for (argument in arguments)
        if (startsWith(argument, "threads=")) nn$set_num_threads(as.integer(substring(argument, 9L)))
    module_checks()
    full_checks()
    cfg <- image_training$ImageTrainingConfig()
    cfg$name <- "eca_resnet50_compact_cifar10"
    cfg$maximum_fit_rows <- 12000L
    cfg$epochs <- 12L
    cfg$minimum_test_accuracy <- .4
    options <- parse_cifar_options(arguments, cfg)
    explicit_epochs <- options$explicit_epochs
    data <- image_training$LoadCifar10(options$directory)
    development <- data$first
    test <- data$second
    model_cfg <- attention_cnn$AttentionResNetConfig()
    model_cfg$classes <- 10L
    model_cfg$width_divisor <- 4L
    model_cfg$small_stem <- TRUE
    model_cfg$stride_on_3x3 <- FALSE
    model_cfg$attention <- attention_cnn$ChannelAttention_kEca
    model_cfg$adaptive_eca <- TRUE
    model <- function(params, state, image, training) {
        result <- attention_cnn$AttentionResNet(params, state, image, model_cfg, training)
        check$Require(params$count() == 1484260, "compact ECA-ResNet-50 parameter count")
        result$logits
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
    cat("All ECA-Net checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
