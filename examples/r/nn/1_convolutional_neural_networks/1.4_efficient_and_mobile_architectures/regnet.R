# Native R NN example. RegNet design/cost/gradient checks and Y-400MF CIFAR-10 training; the C++ example's loop, written out in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.4_efficient_and_mobile_architectures/regnet.R
# RegNet: X and Y models generated from a quantized linear width rule.
#
# Generate all fourteen released RegNetX/Y models (400MF to 32GF) from
# (depth, w0, wa, wm, group width) and check their stage widths and depths,
# learned parameters (5,495,976 for X-400MF to 145,046,770 for Y-32GF,
# torchvision's counts) and multiply-adds (0.414 to 31.736 GFLOPs for X and
# 0.402 to 32.280 for Y, torchvision's figures) at 224 pixels. Show the rule
# at work for Y-400MF. Check every derivative of Y-400MF in training mode
# with shifted BN betas and SE biases; with a ReLU after every addition, none
# vanish.
# https://arxiv.org/abs/2003.13678
# https://github.com/pytorch/vision/blob/main/torchvision/models/regnet.py
#
# Train Y-400MF on CIFAR-10 with a stride-one stem (all sixteen blocks and
# the four stage downsamplings). Use 12k fit / 5k validation, twelve epochs,
# crop/flip, Adam(.001), kernel L2(.0001) and the best-validation checkpoint
# before evaluating all 10k test images.
# Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1.
#
# The model components (regnet::RegNet and its design rule, the grouped
# convolutions, mobile_checks fixtures, image batching, cross-entropy and L2)
# are the C++ ones through SWIG (datamungerNN::nn_components); the checks and
# the workflow below mirror examples/cpp/source/nn/.../regnet.cpp line by line.
suppressPackageStartupMessages(library(datamungerNN))
check <- nn_components$check
example_runtime <- nn_components$example_runtime
grouped <- nn_components$grouped
image_training <- nn_components$image_training
mobile_checks <- nn_components$mobile_checks
paths <- nn_components$paths
regnet <- nn_components$regnet

# C++ default ostream formatting (six significant digits).
g6 <- function(x) sprintf("%g", x)

# regnet.cpp's Join: slash-separated integers.
join <- function(values) paste(as.integer(values), collapse = "/")

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

# Compare the packed grouped kernel with independent per-group convolutions,
# including every filter, bias and input derivative in float64.
fused_group_checks <- function() {
    for (bias in c(FALSE, TRUE))
        for (activation in c(nn$Activation$None, nn$Activation$Relu)) {
            p <- nn$ParameterStore(17L, nn$DType$Float64)
            values <- sin(.13 * (seq_len(2 * 7 * 6 * 12) - 1))
            x <- nn$from_vector_double(values, c(2L, 7L, 6L, 12L))$requires_grad_()
            o <- nn$ConvLayerOptions()
            o$bias <- bias
            o$activation <- activation
            o$stride <- 2L
            o$dilation <- 2L
            split <- grouped$GroupedConv2D(p, "fixture", x, 15L, 3L, 3L, o)
            inputs <- c(nn$tensor_list(p$parameters()), list(x))
            a <- nn$grad(nn$sum(nn$square(split)), inputs)
            fused <- grouped$FusedGroupedConv2D(p, "fixture", x, 15L, 3L, 3L, o)
            b <- nn$grad(nn$sum(nn$square(fused)), inputs)
            check$Near(nn$max(nn$abs(nn$sub(split, fused)))$item(), 0, 1e-12, "packed grouped convolution values")
            for (i in seq_along(a))
                check$Near(nn$max(nn$abs(nn$sub(a[[i]], b[[i]])))$item(), 0, 1e-11,
                           "every packed grouped filter/bias/input derivative")
        }
}

design_checks <- function() {
    d <- regnet$RegNetDesign(regnet$RegNetModel("y400mf"))
    # BINDING GAP (R only): `d$continuous` (std::vector<float>) fails because the SWIG R
    # getter wraps the converted numeric in an undefined pointer class; call the bound
    # getter directly until the wrapper is fixed, then replace this with d$continuous.
    continuous <- .Call("R_swig_regnet__RegNetStages_continuous_get", d@ref, FALSE, PACKAGE = "datamungerNN")
    cat(sprintf("  Y-400MF (d=16, w0=48, wa=27.89, wm=2.09): u_j = 48, %s, %s, ..., %s quantize to %s -> stages %s with depths %s\n",
                g6(continuous[2L]), g6(continuous[3L]), g6(continuous[length(continuous)]), join(d$quantized),
                join(d$widths), join(d$depths)))
}

cost_checks <- function() {
    cases <- list(
        list("x400mf", 5495976, 37856, 413812608, c(32, 64, 160, 400), c(1, 2, 7, 12)),
        list("x800mf", 7259656, 37312, 799699712, c(64, 128, 288, 672), c(1, 3, 7, 5)),
        list("x1.6gf", 9190136, 43504, 1602849792, c(72, 168, 408, 912), c(2, 4, 10, 2)),
        list("x3.2gf", 15296552, 62560, 3176621952, c(96, 192, 432, 1008), c(2, 6, 15, 2)),
        list("x8gf", 39572648, 90464, 7995132416, c(80, 240, 720, 1920), c(2, 5, 15, 1)),
        list("x16gf", 54278536, 111168, 15940755456, c(256, 512, 896, 2048), c(2, 6, 13, 1)),
        list("x32gf", 107811560, 162016, 31735929792, c(336, 672, 1344, 2520), c(2, 7, 13, 1)),
        list("y400mf", 4344144, 27152, 401842848, c(48, 104, 208, 440), c(1, 3, 6, 6)),
        list("y800mf", 6432512, 30432, 833856256, c(64, 144, 320, 784), c(1, 3, 8, 2)),
        list("y1.6gf", 11202430, 52672, 1612076160, c(48, 120, 336, 888), c(2, 6, 17, 2)),
        list("y3.2gf", 19436338, 66160, 3176482752, c(72, 216, 576, 1512), c(2, 5, 13, 1)),
        list("y8gf", 39381472, 86528, 8473070592, c(224, 448, 896, 2016), c(2, 4, 10, 1)),
        list("y16gf", 83590140, 122816, 15911510272, c(224, 448, 1232, 3024), c(2, 4, 11, 1)),
        list("y32gf", 145046770, 158288, 32279552832, c(232, 696, 1392, 3712), c(2, 5, 12, 1)))
    for (cs in cases) {
        name <- cs[[1L]]
        design <- regnet$RegNetDesign(regnet$RegNetModel(name))
        check$Require(identical(as.numeric(design$widths), cs[[5L]]) && identical(as.numeric(design$depths), cs[[6L]]),
                      paste0("RegNet stage design ", name))
        nn$no_grad(function() {
            p <- nn$ParameterStore(17L)
            state <- nn$ParameterStore(17L)
            cfg <- regnet$RegNetConfig()
            cfg$model <- name
            net <- regnet$RegNet(p, state, nn$zeros(c(1L, 224L, 224L, 3L)), cfg)
            # classifier_macs plus every layer's Macs(): MobileNetOutputs::Macs() is that sum.
            macs <- as.numeric(net$Macs())
            check$Require(p$count() == cs[[2L]] && state$count() == cs[[3L]] && macs == cs[[4L]],
                          paste0("RegNet cost ", name))
            cat(sprintf("%s:widths=%s depths=%s parameters=%.0f MACs=%.0f\n", name, join(design$widths),
                        join(design$depths), as.numeric(p$count()), macs))
        })
    }
}

gradient_checks <- function() {
    p <- nn$ParameterStore(17L)
    state <- nn$ParameterStore(17L)
    cfg <- regnet$RegNetConfig()
    cfg$update_stats <- FALSE
    x <- mobile_checks$Pattern(c(2L, 224L, 224L, 3L), 1, 0)
    nn$no_grad(function() {
        regnet$RegNet(p, state, x, cfg, TRUE)
        mobile_checks$ShiftBetas(p)
    })
    net <- regnet$RegNet(p, state, x, cfg, TRUE)
    norms <- as.numeric(mobile_checks$GradientNorms(p, nn$mean(nn$square(net$logits))))
    for (norm in norms) check$Require(is.finite(norm), "finite RegNet derivative")
    check$Require(min(norms) > 1e-5 * max(norms), "nonzero RegNet gradients with shifted BN/SE")
    cat(sprintf("Y-400MF gradient norms %s to %s\n", g6(min(norms)), g6(max(norms))))
}

main <- function(arguments = commandArgs(TRUE)) {
    nn$set_num_threads(2L)
    fused_group_checks()
    design_checks()
    cost_checks()
    gradient_checks()
    cfg <- image_training$ImageTrainingConfig()
    cfg$name <- "regnet_y400mf_cifar10"
    cfg$maximum_fit_rows <- 12000L
    cfg$epochs <- 12L
    cfg$minimum_test_accuracy <- 0.4
    options <- parse_cifar_options(arguments, cfg)
    explicit_epochs <- options$explicit_epochs
    data <- image_training$LoadCifar10(options$directory)
    development <- data$first
    test <- data$second
    # CIFAR RegNet Y-400MF: stride-one stem and ten classes; the parameter count is checked on every pass.
    model <- function(params, state, image, training) {
        model_options <- regnet$RegNetConfig()
        model_options$classes <- 10L
        model_options$small_stem <- TRUE
        net <- regnet$RegNet(params, state, image, model_options, training)
        check$Require(params$count() == 3907554, "CIFAR RegNet Y-400MF count")
        net$logits
    }

    # The training workflow is written out below, mirroring the C++. Quick mode
    # bounds the data and batch; an explicit epoch budget is honored.
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
    cat("All RegNet checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
