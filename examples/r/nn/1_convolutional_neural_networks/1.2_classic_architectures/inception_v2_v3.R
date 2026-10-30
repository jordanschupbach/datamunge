# Native R NN example. inception_v2_v3: factorization/BN/label-smoothing checks, full v2/v3 architecture checks and compact CIFAR-10 training of both versions over SWIG components.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.2_classic_architectures/inception_v2_v3.R
# Inception v2 and v3: batch normalization, factorization and label smoothing.
#
# Port of tflow's inception_v2_v3. Implement full TF-Slim reference layouts:
# v2 at 224 pixels, v3 at 299 pixels, with 1000 classes. Check every stage
# shape, independently counted learned parameters and running statistics, and
# finite nonzero parameter derivatives. v2 includes the separable 7x7 stem and
# ten ordinary/reduction modules. v3 includes asymmetric 1x7/7x1 convolutions,
# split 1x3/3x1 branches and an auxiliary head. Reference definitions and
# initialization differences are documented in common/inception23.hpp.
#
# Demonstrate linear spatial factorization with an outer-product 7x7 kernel:
# two rectangular convolutions match the composed convolution and its input
# and factor gradients. This equivalence needs a factorizable kernel and no
# intervening nonlinearities; real Inception modules include BN/ReLU. Verify
# beta-only BN against host moments, running-statistic updates, and smoothed
# training cross entropy separately from unsmoothed evaluation NLL.
#
# Train compact CIFAR-10 versions, retaining every block: quarter widths,
# 64-pixel v2 / 107-pixel v3 inputs, 12k fit / 5k validation rows, twelve
# epochs, Adam(.001), crop/flip augmentation and .8 dropout retention. Use
# momentum .9 for BN on these short runs; canonical configuration uses .9997.
# v3 adds .1 label smoothing and .4-weight auxiliary cross entropy. Validation
# selection and the final 10k-row test use only unsmoothed primary NLL.
# These demonstrations are not full-model ImageNet benchmark results.
# CIFAR-10 is read from paths::Dataset("cifar10"). Batch-norm moving averages
# update eagerly during training passes (tflow runs collected update ops), so
# "inference requests no updates" becomes "an inference pass leaves the state
# store unchanged".
# Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.
#
# The model components (inception23::Inception23, normalized_conv::NormalizedConv,
# the architecture helpers, image batching, cross-entropy and L2) are the C++ ones
# through SWIG (datamungerNN::nn_components); the checks and the experiment workflow
# below mirror examples/cpp/source/nn/.../inception_v2_v3.cpp line by line.
suppressPackageStartupMessages(library(datamungerNN))
architecture <- nn_components$architecture
check <- nn_components$check
example_runtime <- nn_components$example_runtime
image_training <- nn_components$image_training
inception23 <- nn_components$inception23
normalized_conv <- nn_components$normalized_conv
paths <- nn_components$paths

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

# image_training::TrainingObjective written out over the bound components: (smoothed)
# cross-entropy of the training logits, plus weight * cross-entropy of every auxiliary head,
# plus 0.5 * l2 * sum(W^2) over parameters of rank >= 2.
training_objective <- function(params, state, model, x, y, cfg, training = NULL) {
    network <- if (is.null(training)) list(logits = model(params, state, x, TRUE), auxiliary = list())
               else training(params, state, x)
    objective <- image_training$TrainingCrossEntropy(network$logits, y, cfg$classes, cfg$label_smoothing)
    for (head in network$auxiliary) {
        if (!is.finite(head$weight) || head$weight <= 0) stop("auxiliary loss weight must be positive and finite")
        objective <- nn$add(objective, nn$mul(
            image_training$TrainingCrossEntropy(head$logits, y, cfg$classes, cfg$label_smoothing), head$weight))
    }
    image_training$WithL2(objective, params, cfg$l2)
}

factorization_checks <- function() {
    x <- architecture$SinePixels(c(1L, 11L, 11L, 1L), 1, .31)
    vertical <- nn$from_vector(c(.1, .2, -.1, .3, .1, -.2, .2), c(7L))
    horizontal <- nn$from_vector(c(.3, -.1, .2, .1, .2, -.3, .1), c(7L))
    for (t in list(x, vertical, horizontal)) t$requires_grad_(TRUE)
    v <- nn$reshape(vertical, c(7L, 1L, 1L, 1L))
    h <- nn$reshape(horizontal, c(1L, 7L, 1L, 1L))
    valid <- nn$conv_options(1L, nn$Padding$Valid)
    factored <- nn$conv2d(nn$conv2d(x, h, valid), v, valid)
    composed <- nn$conv2d(x, nn$mul(v, h), valid)
    keep <- nn$GradOptions()
    keep$retain_graph <- TRUE  # both graphs share the reshaped factors
    a <- nn$grad(nn$mean(nn$square(factored)), nn$tensor_vector(list(x, vertical, horizontal)), keep)
    b <- nn$grad(nn$mean(nn$square(composed)), nn$tensor_vector(list(x, vertical, horizontal)))
    error <- 0
    compare <- function(p, q) error <<- max(error, max(abs(p$values() - q$values())))
    compare(factored, composed)
    for (i in seq_along(a)) compare(a[[i]], b[[i]])
    check$Near(error, 0, 2e-6, "factorized output and input/factor gradients")
    cat(sprintf("  1x7 then 7x1 vs outer-product 7x7: output/gradient max error=%s; 14 vs 49 coefficients per channel pair at equal intermediate width\n",
                g6(error)))
}

normalization_checks <- function() {
    p <- nn$ParameterStore(5L)
    state <- nn$ParameterStore(5L)
    cfg <- normalized_conv$ConvNormConfig()
    cfg$training <- TRUE
    cfg$momentum <- .9
    input <- architecture$SinePixels(c(2L, 4L, 4L, 1L), 1, .27)
    output <- normalized_conv$NormalizedConv(p, state, "rect", input, 2L, 1L, 3L, cfg, 1L, "VALID", FALSE)$values()
    convolution <- nn$conv2d(input, p$at("rect/W"), nn$conv_options(1L, nn$Padding$Valid))$values()
    check$Require(p$count() == 8 && state$count() == 4, "beta-only BN parameters and running state")
    mean_state <- state$at("rect/bn/moving_mean")$values()
    var_state <- state$at("rect/bn/moving_var")$values()
    rows <- length(output) %/% 2
    for (c in 0:1) {
        mean <- 0; variance <- 0
        for (i in seq_len(rows) - 1) mean <- mean + convolution[2 * i + c + 1] / rows
        for (i in seq_len(rows) - 1) variance <- variance + (convolution[2 * i + c + 1] - mean)^2 / rows
        for (i in seq_len(rows) - 1)
            check$Near(output[2 * i + c + 1], (convolution[2 * i + c + 1] - mean) / sqrt(variance + .001), 2e-6,
                       "fixed-gamma BN output")
        check$Near(mean_state[c + 1], .1 * mean, 2e-6, "one BN mean update")
        check$Near(var_state[c + 1], .9 + .1 * variance * rows / (rows - 1), 2e-6, "one unbiased-variance update")
    }
    cat("  rectangular convolution and beta-only BN: host output/moments and one EMA update passed\n")
}

smoothing_checks <- function() {
    p <- nn$ParameterStore(17L)
    state <- nn$ParameterStore(17L)
    p$set("W", nn$from_vector(c(10, 0, 0), c(1L, 3L)))
    model <- function(q, state, x, training) nn$matmul(x, q$at("W"))
    cfg <- image_training$ImageTrainingConfig()
    cfg$classes <- 3L
    cfg$l2 <- 0
    cfg$label_smoothing <- .1
    x <- nn$ones(c(1L, 1L))
    y <- nn$from_vector_int64(c(0), c(1L))
    evaluation <- 0; training <- 0
    nn$no_grad(function() {
        evaluation <<- nn$sparse_softmax_cross_entropy(model(p, state, x, FALSE), y)$item()
        training <<- training_objective(p, state, model, x, y, cfg)$item()
    })
    nll <- log1p(2 * exp(-10))
    check$Near(evaluation, nll, 2e-7, "evaluation retains unsmoothed NLL")
    check$Near(training, nll + 2 / 3, 2e-6, "uniform label smoothing cross entropy")
    config <- nn$OptimizerConfig()
    config$kind <- nn$OptimizerKind$Adam
    optimizer <- nn$Optimizer(p, .01, config)
    optimizer$minimize(training_objective(p, state, model, x, y, cfg))
    check$Require(p$at("W")$values()[1] < 10, "smoothing reduces an overconfident correct-class score")
    cat(sprintf("  .1 label smoothing: evaluation NLL=%s; training CE=%s; correct-class score decreases\n",
                g6(nll), g6(nll + 2 / 3)))
}

full_architecture_checks <- function(version) {
    p <- nn$ParameterStore(17L)
    state <- nn$ParameterStore(17L)
    cfg <- inception23$Inception23Config()
    cfg$version <- version
    cfg$image_size <- if (version == 2L) 224L else 299L
    pixels <- architecture$SinePixels(c(1L, cfg$image_size, cfg$image_size, 3L), .1, .13)
    net <- inception23$Inception23(p, state, pixels, cfg, FALSE, TRUE)
    aux_count <- architecture$CountWithPrefix(p, "aux/")
    main_count <- p$count() - aux_count
    check$Require(main_count == (if (version == 2L) 11178336 else 23817352) &&
                      aux_count == (if (version == 2L) 0 else 3325800),
                  "reference learned parameter counts")
    check$Require(state$count() == (if (version == 2L) 19776 else 36224), "reference running-state count")
    initial_state <- state$flatten()
    expected <- if (version == 2L)
        list(c(112, 64), c(56, 64), c(56, 64), c(56, 192), c(28, 192), c(28, 256), c(28, 320), c(14, 576), c(14, 576),
             c(14, 576), c(14, 576), c(14, 576), c(7, 1024), c(7, 1024), c(7, 1024))
    else
        list(
            c(149, 32),
            c(147, 32),
            c(147, 64),
            c(73, 64),
            c(73, 80),
            c(71, 192),
            c(35, 192),
            c(35, 256),
            c(35, 288),
            c(35, 288),
            c(17, 768),
            c(17, 768),
            c(17, 768),
            c(17, 768),
            c(17, 768),
            c(8, 1280),
            c(8, 2048),
            c(8, 2048))
    stages <- nn$named_list(net$stages)  # name -> tensor, in construction order
    check$Require(length(stages) == length(expected), "reference stage count")
    for (i in seq_along(expected)) {
        spatial <- expected[[i]][1]; channels <- expected[[i]][2]
        architecture$RequireShape(stages[[i]], as.integer(c(1, spatial, spatial, channels)),
                                  paste0("reference Inception stage shape: ", names(stages)[i]))
    }
    objective <- nn$mean(nn$square(net$logits))
    for (aux in nn$tensor_list(net$auxiliary)) objective <- nn$add(objective, nn$mul(nn$mean(nn$square(aux)), .4))
    range <- architecture$GradientNormRange(p, objective, "Inception")
    check$Require(identical(state$flatten(), initial_state), "inference does not update BN running statistics")
    cat(sprintf("  full Inception v%d: main=%.0f, aux=%.0f, running state=%.0f; %d stage shapes; gradient norms [%s,%s]\n",
                version, main_count, aux_count, state$count(), length(expected), g6(range$first), g6(range$second)))
}

compact_config <- function(version) {
    cfg <- inception23$Inception23Config()
    cfg$version <- version
    cfg$image_size <- if (version == 2L) 64L else 107L
    cfg$classes <- 10L
    cfg$width_divisor <- 4L
    cfg$momentum <- .9
    cfg
}

main <- function(arguments = commandArgs(TRUE)) {
    factorization_checks()
    normalization_checks()
    smoothing_checks()
    full_architecture_checks(2L)
    full_architecture_checks(3L)
    for (version in c(2L, 3L)) {
        cfg_model <- compact_config(version)
        model <- function(p, state, x, training) {
            image <- nn$resize_bilinear(x, c(cfg_model$image_size, cfg_model$image_size), FALSE, TRUE)
            inception23$Inception23(p, state, image, cfg_model, training)$logits
        }
        training <- NULL
        if (version == 3L)
            # Training path: the same network with its auxiliary head (tflow's
            # ClassifierNetwork::auxiliary); returns list(logits, auxiliary = list(list(logits, weight))).
            training <- function(p, state, x) {
                image <- nn$resize_bilinear(x, c(cfg_model$image_size, cfg_model$image_size), FALSE, TRUE)
                net <- inception23$Inception23(p, state, image, cfg_model, TRUE, TRUE)
                list(logits = net$logits, auxiliary = list(list(logits = nn$tensor_list(net$auxiliary)[[1]], weight = .4)))
            }
        cfg <- image_training$ImageTrainingConfig()
        cfg$name <- sprintf("inception_v%d_compact_cifar10", version)
        cfg$maximum_fit_rows <- 12000L
        cfg$epochs <- 12L
        cfg$minimum_test_accuracy <- .35
        cfg$label_smoothing <- if (version == 3L) .1 else 0
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
            x <- image_training$ImageBatch(development, fit[1:2])$first
            model(params, state, x, FALSE)
            if (!is.null(training)) {
                scratch <- nn$ParameterStore(cfg$seed)
                training(params, scratch, x)
            }
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
                # Augmented batch; training-mode forward; (smoothed) cross-entropy plus the weighted
                # auxiliary cross-entropies plus 0.5 * l2 * sum(W^2); one Adam step.
                batch <- if (cfg$augment) image_training$ImageBatch(development, rows, augmentation)
                         else image_training$ImageBatch(development, rows)
                network <- if (!is.null(training)) training(params, state, batch$first)
                           else list(logits = model(params, state, batch$first, TRUE), auxiliary = list())
                objective <- image_training$TrainingCrossEntropy(network$logits, batch$second, cfg$classes, cfg$label_smoothing)
                for (head in network$auxiliary) {
                    check$Require(is.finite(head$weight) && head$weight > 0, "positive finite auxiliary loss weight")
                    objective <- nn$add(objective, nn$mul(
                        image_training$TrainingCrossEntropy(head$logits, batch$second, cfg$classes, cfg$label_smoothing),
                        head$weight))
                }
                objective <- image_training$WithL2(objective, params, cfg$l2)
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
    }
    cat("All Inception v2/v3 checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
