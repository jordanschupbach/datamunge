# Native R NN example. googlenet: Inception v1 branch/auxiliary/full-architecture checks and compact CIFAR-10 training with auxiliary heads over SWIG components.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.2_classic_architectures/googlenet.R
# GoogLeNet / Inception v1: parallel receptive fields and auxiliary supervision.
#
# Port of tflow's googlenet. Implement the nine modules and 224-pixel/1000-class
# widths in Table 1 of https://arxiv.org/abs/1409.4842. Check all stage
# dimensions, 6,998,552 main parameters and 6,379,728 auxiliary parameters,
# and finite nonzero gradients through every learned tensor. Independently
# check branch filtering and concatenation against host arithmetic, including
# padded image boundaries. Measure how the 1x1 reductions change
# spatial-kernel multiply-add counts.
#
# The two auxiliary heads follow modules 4a/4d: 5x5 stride-3 average pooling,
# 1x1 projection, a dense hidden layer, 70% dropout, and class logits. Their
# cross entropies each have weight .3 during training. Verify weighted losses
# against host log-sum-exp, auxiliary parameter updates with L2 enabled, and
# unchanged inference after perturbing auxiliary parameters. Main dropout is
# 40%. See common/googlenet.hpp for padding, LRN and initialization conventions.
#
# Train a separate CPU demonstration on CIFAR-10: resize to 96 pixels, quarter
# convolution widths, 128-unit auxiliary hidden layers, all nine modules,
# 12k fit / 5k validation rows, crop/flip augmentation, Adam(.001), twelve
# epochs. Select by primary validation NLL and evaluate the complete 10k test
# set. This compact demonstration is not an ImageNet benchmark.
# CIFAR-10 is read from paths::Dataset("cifar10"); auxiliary heads train through
# the written-out training objective (tflow's ClassifierNetwork::auxiliary).
# Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.
#
# The model components (googlenet::GoogLeNet, inception::InceptionV1, the
# architecture helpers, image batching, cross-entropy and L2) are the C++ ones
# through SWIG (datamungerNN::nn_components); the checks and the experiment
# workflow below mirror examples/cpp/source/nn/.../googlenet.cpp line by line.
suppressPackageStartupMessages(library(datamungerNN))
architecture <- nn_components$architecture
check <- nn_components$check
example_runtime <- nn_components$example_runtime
googlenet <- nn_components$googlenet
image_training <- nn_components$image_training
inception <- nn_components$inception
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

# inception::InceptionWidths{...}.
inception_widths <- function(one, reduce_three, three, reduce_five, five, pool) {
    widths <- inception$InceptionWidths()
    widths$one <- one; widths$reduce_three <- reduce_three; widths$three <- three
    widths$reduce_five <- reduce_five; widths$five <- five; widths$pool <- pool
    widths
}

# googlenet::GoogLeNetWidths()[0] (the std::array accessor is not bound; Table 1, module 3a).
googlenet_widths_3a <- c(64L, 96L, 128L, 16L, 32L, 32L)

branch_checks <- function() {
    p <- nn$ParameterStore(0L)
    pixels <- numeric(50)
    for (r in 0:4)
        for (c in 0:4) {
            pixels[(r * 5 + c) * 2 + 1] <- f32(f32(1 + f32(r * f32(.2))) + f32(c * f32(.1)))
            pixels[(r * 5 + c) * 2 + 2] <- f32(f32(2 + f32(r * f32(.1))) + f32(c * f32(.3)))
        }
    input <- nn$from_vector(pixels, c(1L, 5L, 5L, 2L))
    widths <- inception_widths(2L, 1L, 1L, 1L, 1L, 1L)
    nn$no_grad(function() {
        inception$InceptionV1(p, "block", input, widths)
        names <- p$names()
        parameters <- nn$tensor_list(p$parameters())
        for (i in seq_len(p$size())) {
            name <- names[i]
            value <- numeric(parameters[[i]]$numel())
            if (name == "block/one/W") value[c(1, 4)] <- 1
            else if (name == "block/three_reduce/W") value[1:2] <- c(1, 2)
            else if (name == "block/five_reduce/W") value[1:2] <- c(3, 1)
            else if (name %in% c("block/three/W", "block/five/W")) value[] <- .1
            else if (name == "block/pool/W") value[1:2] <- c(5, 6)
            p$set(name, nn$from_vector(value, parameters[[i]]$shape()))
        }
    })
    net <- inception$InceptionV1(p, "block", input, widths)
    architecture$RequireShape(net$concatenated, c(1L, 5L, 5L, 5L), "Inception branch channel sum")
    out <- net$concatenated$values()
    error <- 0
    at <- function(y, x, c) pixels[(y * 5 + x) * 2 + c + 1]
    for (r in 0:4)
        for (c in 0:4) {
            expected <- c(at(r, c, 0), at(r, c, 1), 0, 0, 0)
            m0 <- 0; m1 <- 0
            for (dr in -2:2)
                for (dc in -2:2) {
                    y <- r + dr; x <- c + dc
                    if (y < 0 || y >= 5 || x < 0 || x >= 5) next
                    a <- at(y, x, 0); b <- at(y, x, 1)
                    expected[4] <- expected[4] + .1 * (3 * a + b)
                    if (abs(dr) <= 1 && abs(dc) <= 1) {
                        expected[3] <- expected[3] + .1 * (a + 2 * b)
                        m0 <- max(m0, a)
                        m1 <- max(m1, b)
                    }
                }
            expected[5] <- 5 * m0 + 6 * m1
            for (channel in 0:4)
                error <- max(error, abs(out[(r * 5 + c) * 5 + channel + 1] - expected[channel + 1]))
        }
    check$Near(error, 0, 1e-5, "four branches and concatenation match independent boundary-aware arithmetic")
    w <- do.call(inception_widths, as.list(googlenet_widths_3a))
    reduced <- w$KernelWeights(192L)
    direct <- 192 * (w$one + 9 * w$three + 25 * w$five + w$pool)
    check$Require(reduced == 163328 && direct == 393216, "first-module kernel MACs per spatial position")
    cat(sprintf("  branch arithmetic max error=%s; Inception 3a MACs/position: reduced=%.0f, direct spatial convolutions=%.0f (pool comparisons excluded)\n",
                g6(error), reduced, direct))
}

# Host mean log-sum-exp cross-entropy of [rows, columns] scores against int64 labels (0-based).
cross_entropy <- function(scores, labels) {
    values <- scores$values()
    shape <- as.numeric(scores$shape())
    rows <- shape[1]; columns <- shape[2]
    total <- 0
    for (r in seq_len(rows) - 1) {
        row <- values[r * columns + seq_len(columns)]
        largest <- max(row)
        z <- sum(exp(row - largest))
        total <- total + largest + log(z) - row[labels[r + 1] + 1]
    }
    total / rows
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

auxiliary_loss_checks <- function() {
    p <- nn$ParameterStore(17L)
    state <- nn$ParameterStore(17L)
    model <- function(q, state, x, training) nn$dense(q, "main", x, 3L)
    training <- function(q, state, x) {
        # TrainingNetwork{main logits, {{aux logits, .3}}}: "main" parameters are created before "aux".
        logits <- nn$dense(q, "main", x, 3L)
        list(logits = logits, auxiliary = list(list(logits = nn$dense(q, "aux", x, 3L), weight = .3)))
    }
    x <- nn$from_vector(c(1, -.2, -.4, .7), c(2L, 2L))
    labels <- c(0, 2)
    y <- nn$from_vector_int64(labels, c(2L))
    cfg <- image_training$ImageTrainingConfig()
    cfg$classes <- 3L
    cfg$l2 <- 0  # tflow's training_loss excludes the L2 term
    nn$no_grad(function() training(p, state, x))
    losses <- function() nn$no_grad(function() {
        main <- nn$dense(p, "main", x, 3L)
        aux <- nn$dense(p, "aux", x, 3L)
        loss <- training_objective(p, state, model, x, y, cfg, training)$item()
        check$Near(loss, cross_entropy(main, labels) + .3 * cross_entropy(aux, labels), 2e-6,
                   "primary plus weighted auxiliary cross entropy")
        loss
    })
    inference <- function() nn$no_grad(function() nn$softmax(model(p, state, x, FALSE), 1L)$values())
    before <- inference()
    old_loss <- losses()
    p$set("aux/b", nn$from_vector(c(3, -2, 1), c(3L)))
    after <- inference()
    new_loss <- losses()
    for (i in 1:6) check$Near(before[i], after[i], 0, "auxiliary head cannot change inference")
    check$Require(abs(old_loss - new_loss) > .01, "auxiliary head changes the training objective")
    # Biases are excluded from L2: their movement demonstrates auxiliary CE
    # connectivity even when regularization is enabled.
    cfg$l2 <- .1
    config <- nn$OptimizerConfig()
    config$kind <- nn$OptimizerKind$Adam
    optimizer <- nn$Optimizer(p, .01, config)
    optimizer$minimize(training_objective(p, state, model, x, y, cfg, training))
    check$Require(abs(p$at("aux/b")$values()[1] - 3) > .001,
                  "auxiliary cross entropy updates its parameters with L2 enabled")
    cat(sprintf("  auxiliary loss: %s -> %s after aux-only perturbation; inference unchanged; training updates aux bias\n",
                g6(old_loss), g6(new_loss)))
}

full_architecture_checks <- function() {
    p <- nn$ParameterStore(17L)
    net <- googlenet$GoogLeNet(p, architecture$SinePixels(c(1L, 224L, 224L, 3L), .1, .13), googlenet$GoogLeNetConfig(),
                               FALSE, TRUE)
    aux_count <- architecture$CountWithPrefix(p, "aux")
    main_count <- p$count() - aux_count
    check$Require(main_count == 6998552 && aux_count == 6379728, "canonical main and auxiliary parameter counts")
    auxiliary <- nn$tensor_list(net$auxiliary)
    objective <- nn$mean(nn$square(net$logits))
    for (aux in auxiliary) objective <- nn$add(objective, nn$mul(nn$mean(nn$square(aux)), .3))
    stages <- c(lapply(0:2, function(i) net$stem(i)), lapply(0:8, function(i) net$inception(i)), list(net$logits), auxiliary)
    shapes <- list(
        c(1, 112, 112, 64),
        c(1, 56, 56, 64),
        c(1, 56, 56, 192),
        c(1, 28, 28, 256),
        c(1, 28, 28, 480),
        c(1, 14, 14, 512),
        c(1, 14, 14, 512),
        c(1, 14, 14, 512),
        c(1, 14, 14, 528),
        c(1, 14, 14, 832),
        c(1, 7, 7, 832),
        c(1, 7, 7, 1024),
        c(1, 1000),
        c(1, 1000),
        c(1, 1000))
    check$Require(length(stages) == length(shapes), "GoogLeNet stage count")
    for (i in seq_along(shapes))
        architecture$RequireShape(stages[[i]], as.integer(shapes[[i]]), "GoogLeNet stage shape")
    range <- architecture$GradientNormRange(p, objective, "GoogLeNet parameter tensor")
    cat(sprintf("  full GoogLeNet: main=%.0f, auxiliary=%.0f, total=%.0f; all 15 stage shapes passed; gradient norms [%s,%s]\n",
                main_count, aux_count, p$count(), g6(range$first), g6(range$second)))
}

compact_config <- function() {
    cfg <- googlenet$GoogLeNetConfig()
    cfg$image_size <- 96L
    cfg$classes <- 10L
    cfg$width_divisor <- 4L
    cfg$auxiliary_hidden <- 128L
    cfg
}

main <- function(arguments = commandArgs(TRUE)) {
    branch_checks()
    auxiliary_loss_checks()
    full_architecture_checks()
    cfg <- image_training$ImageTrainingConfig()
    cfg$name <- "googlenet_compact_cifar10"
    cfg$maximum_fit_rows <- 12000L
    cfg$epochs <- 12L
    cfg$minimum_test_accuracy <- .35
    options <- parse_cifar_options(arguments, cfg)
    explicit_epochs <- options$explicit_epochs
    data <- image_training$LoadCifar10(options$directory)
    development <- data$first
    test <- data$second
    # Compact GoogLeNet on CIFAR-10: fixed development split, Adam on the primary
    # cross-entropy plus the two .3-weight auxiliary heads plus L2, validation-NLL
    # checkpoint selection on the primary logits, one test pass. The whole workflow
    # is written out here, line by line with the C++ example, over the same bound components.
    model <- function(p, state, x, training)
        googlenet$GoogLeNet(p, nn$resize_bilinear(x, c(96L, 96L), FALSE, TRUE), compact_config(), training)$logits
    # Training path: the same network with its auxiliary heads (tflow's
    # ClassifierNetwork::auxiliary); returns list(logits, auxiliary = list(list(logits, weight), ...)).
    training <- function(p, state, x) {
        net <- googlenet$GoogLeNet(p, nn$resize_bilinear(x, c(96L, 96L), FALSE, TRUE), compact_config(), TRUE, TRUE)
        auxiliary <- nn$tensor_list(net$auxiliary)
        list(logits = net$logits, auxiliary = list(list(logits = auxiliary[[1]], weight = .3),
                                                   list(logits = auxiliary[[2]], weight = .3)))
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
        x <- image_training$ImageBatch(development, fit[1:2])$first
        model(params, state, x, FALSE)
        # Training-only parameters (auxiliary heads); their batch-norm statistics
        # go to a scratch store so this pass updates no real moving averages.
        scratch <- nn$ParameterStore(cfg$seed)
        training(params, scratch, x)
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
            # Augmented batch; training-path forward; cross-entropy of the primary logits plus the
            # weighted auxiliary cross-entropies, plus 0.5 * l2 * sum(W^2); one Adam step.
            batch <- if (cfg$augment) image_training$ImageBatch(development, rows, augmentation)
                     else image_training$ImageBatch(development, rows)
            net <- training(params, state, batch$first)
            objective <- image_training$TrainingCrossEntropy(net$logits, batch$second, cfg$classes, cfg$label_smoothing)
            for (head in net$auxiliary) {
                check$Require(is.finite(head$weight) && head$weight > 0, "auxiliary loss weight must be positive and finite")
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
    0L
}

if (sys.nframe() == 0L) quit(status = main())
