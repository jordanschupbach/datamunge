# Native R NN example. inception_v4_resnet: residual-scale and full v4/Inception-ResNet-v2 checks, then compact CIFAR-10 training of both variants with an auxiliary head over SWIG components.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.2_classic_architectures/inception_v4_resnet.R
# Inception v4 and Inception-ResNet-v2: full depth and scaled residual branches.
#
# Port of tflow's inception_v4_resnet. Implement the complete TF-Slim layouts:
# v4's 4/7/3 A/B/C blocks and the residual model's 10/20/10 blocks.
# Inception-ResNet uses linear projections without BN; scales are
# .17/.10/.20, except the last C block is unscaled and has no final ReLU. Both
# include the reference auxiliary head. Validate every canonical stage,
# independent main/auxiliary/state parameter counts, and gradients through all
# learned tensors at 299 pixels and 1000 classes. Architecture and adaptation
# details: common/inception4.hpp.
#
# For each residual block type, compare shared-weight evaluations at scales
# zero, s and 2s. Check the identity path, output scaling, parameter-gradient
# scaling and the identity contribution to input gradients. This isolates the
# residual mechanism without making an unsupported empirical stability claim.
#
# Train compact CIFAR-10 demonstrations with every block retained: 107-pixel
# input, one-eighth widths, 12k fit / 5k validation rows, twelve epochs,
# Adam(.001), crop/flip augmentation, BN momentum .9, .8 dropout retention,
# .1 label smoothing and .4-weight auxiliary CE. Select by unsmoothed primary
# validation NLL and evaluate all 10k test rows once. These short CPU runs
# are not full-model ImageNet benchmarks or a controlled residual ablation.
# CIFAR-10 is read from paths::Dataset("cifar10"). BN moving averages update
# eagerly in training passes; inference passes leave the state store unchanged.
# Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.
#
# The model components (inception4::Inception4, inception4::InceptionResidual,
# inception23::InceptionBuilder, the architecture helpers, image batching,
# cross-entropy and L2) are the C++ ones through SWIG (datamungerNN::nn_components);
# the checks and the experiment workflow below mirror
# examples/cpp/source/nn/.../inception_v4_resnet.cpp line by line.
suppressPackageStartupMessages(library(datamungerNN))
architecture <- nn_components$architecture
check <- nn_components$check
example_runtime <- nn_components$example_runtime
image_training <- nn_components$image_training
inception23 <- nn_components$inception23
inception4 <- nn_components$inception4
paths <- nn_components$paths

# C++ default ostream formatting (six significant digits).
g6 <- function(x) sprintf("%g", x)

# Rounds to IEEE single precision (the C++ fixtures are float arithmetic).
f32 <- function(v) nn$from_vector(v, c(length(v)))$values()

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

residual_scale_checks <- function(kind, channels, scale) {
    p <- nn$ParameterStore(17L)
    state <- nn$ParameterStore(17L)
    cfg <- inception23$Inception23Config()
    cfg$width_divisor <- 8L
    result <- inception23$Inception23Outputs()
    builder <- inception23$InceptionBuilder(p, state, result, cfg, FALSE)
    # .4f + .2f * std::sin(float(i) * .17f): single-precision fixture values.
    i <- seq_len(81 * channels) - 1
    values <- f32(.4 + f32(.2 * f32(sin(f32(i * f32(.17))))))
    input <- nn$from_vector(values, c(1L, 9L, 9L, as.integer(channels)))
    input$requires_grad_(TRUE)
    x <- inception23$ConvFeature()
    x$value <- input
    x$channels <- as.integer(channels)
    first <- inception4$InceptionResidual(builder, "block", x, kind, scale, FALSE)$value
    parameters <- p$count()
    twice <- inception4$InceptionResidual(builder, "block", x, kind, 2 * scale, FALSE)$value
    zero <- inception4$InceptionResidual(builder, "block", x, kind, 0, FALSE)$value
    check$Require(p$count() == parameters, "residual fixture shares parameters across scale choices")
    variables <- c(list(input), nn$tensor_list(p$parameters()))
    g1 <- nn$grad(nn$sum(first), nn$tensor_vector(variables))
    g2 <- nn$grad(nn$sum(twice), nn$tensor_vector(variables))
    a <- first$values(); b <- twice$values(); z <- zero$values()
    for (j in seq_along(values)) check$Near(z[j], values[j], 0, "zero residual scale preserves identity")
    error <- max(abs((b - values) - 2 * (a - values)))
    for (parameter in seq_along(g1)) {
        ga <- g1[[parameter]]$values(); gb <- g2[[parameter]]$values()
        skip <- if (parameter == 1L) 1 else 0
        error <- max(error, abs((gb - skip) - 2 * (ga - skip)) / pmax(1, abs(gb)))
    }
    check$Near(error, 0, 2e-5, "residual output, input and parameter derivative scaling")
    cat(sprintf("  residual %s at scale %s: identity and scaling checks; max normalized error=%s\n", kind, g6(scale), g6(error)))
}

full_architecture_checks <- function(residual) {
    p <- nn$ParameterStore(17L)
    state <- nn$ParameterStore(17L)
    cfg <- inception4$Inception4Config()
    cfg$residual <- residual
    net <- inception4$Inception4(p, state, architecture$SinePixels(c(1L, 299L, 299L, 3L), .1, .13), cfg, FALSE, TRUE)
    aux_count <- architecture$CountWithPrefix(p, "aux/")
    main_count <- p$count() - aux_count
    check$Require(main_count == (if (residual) 55813192 else 42648232) && aux_count == (if (residual) 3366760 else 3358568),
                  "reference main/auxiliary parameter counts")
    check$Require(state$count() == (if (residual) 62336 else 64960), "reference BN state count")
    auxiliary <- nn$tensor_list(net$auxiliary)
    check$Require(length(auxiliary) == 1L, "canonical auxiliary head")
    initial_state <- state$flatten()
    expected <- list(c(149, 32), c(147, 32), c(147, 64))
    if (residual) {
        expected <- c(expected, list(c(73, 64), c(73, 80), c(71, 192), c(35, 192), c(35, 320)))
        expected <- c(expected, rep(list(c(35, 320)), 10))
        expected <- c(expected, list(c(17, 1088)))
        expected <- c(expected, rep(list(c(17, 1088)), 20))
        expected <- c(expected, list(c(8, 2080)))
        expected <- c(expected, rep(list(c(8, 2080)), 10))
        expected <- c(expected, list(c(8, 1536)))
    } else {
        expected <- c(expected, list(c(73, 160)))
        expected <- c(expected, list(c(71, 192)))
        expected <- c(expected, list(c(35, 384)))
        expected <- c(expected, rep(list(c(35, 384)), 4))
        expected <- c(expected, list(c(17, 1024)))
        expected <- c(expected, rep(list(c(17, 1024)), 7))
        expected <- c(expected, list(c(8, 1536)))
        expected <- c(expected, rep(list(c(8, 1536)), 3))
    }
    stages <- nn$named_list(net$stages)  # name -> tensor, in construction order
    check$Require(length(stages) == length(expected), "complete architecture block count")
    for (i in seq_along(expected)) {
        spatial <- expected[[i]][1]; channels <- expected[[i]][2]
        architecture$RequireShape(stages[[i]], as.integer(c(1, spatial, spatial, channels)),
                                  paste0("reference stage shape: ", names(stages)[i]))
    }
    check$Require(identical(as.numeric(net$logits$shape()), c(1, 1000)) &&
                      identical(as.numeric(auxiliary[[1]]$shape()), c(1, 1000)),
                  "main and auxiliary classifier shapes")
    objective <- nn$add(nn$mean(nn$square(net$logits)), nn$mul(nn$mean(nn$square(auxiliary[[1]])), .4))
    range <- architecture$GradientNormRange(p, objective, "advanced Inception")
    check$Require(identical(state$flatten(), initial_state), "inference does not update BN running statistics")
    cat(sprintf("  %s: main=%.0f, aux=%.0f, state=%.0f; %d stage shapes; gradient norms [%s,%s]\n",
                if (residual) "Inception-ResNet-v2" else "Inception v4", main_count, aux_count, state$count(),
                length(expected), g6(range$first), g6(range$second)))
}

compact_config <- function(residual) {
    cfg <- inception4$Inception4Config()
    cfg$residual <- residual
    cfg$image_size <- 107L
    cfg$classes <- 10L
    cfg$width_divisor <- 8L
    cfg$momentum <- .9
    cfg
}

main <- function(arguments = commandArgs(TRUE)) {
    residual_scale_checks("A", 40L, .17)
    residual_scale_checks("B", 136L, .1)
    residual_scale_checks("C", 260L, .2)
    full_architecture_checks(FALSE)
    full_architecture_checks(TRUE)
    for (residual in c(FALSE, TRUE)) {
        compact <- compact_config(residual)
        model <- function(p, state, x, training)
            inception4$Inception4(p, state, nn$resize_bilinear(x, c(107L, 107L), FALSE, TRUE), compact, training)$logits
        # Training path: the same network with its auxiliary head (tflow's
        # ClassifierNetwork::auxiliary); returns list(logits, auxiliary = list(list(logits, weight))).
        training <- function(p, state, x) {
            net <- inception4$Inception4(p, state, nn$resize_bilinear(x, c(107L, 107L), FALSE, TRUE), compact, TRUE, TRUE)
            list(logits = net$logits, auxiliary = list(list(logits = nn$tensor_list(net$auxiliary)[[1]], weight = .4)))
        }
        cfg <- image_training$ImageTrainingConfig()
        cfg$name <- if (residual) "inception_resnet_v2_compact_cifar10" else "inception_v4_compact_cifar10"
        cfg$maximum_fit_rows <- 12000L
        cfg$epochs <- 12L
        cfg$minimum_test_accuracy <- .35
        cfg$label_smoothing <- .1
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
                # Augmented batch; training-mode forward; smoothed cross-entropy plus the weighted
                # auxiliary cross-entropy plus 0.5 * l2 * sum(W^2); one Adam step.
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
    cat("All Inception v4 / Inception-ResNet-v2 checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
