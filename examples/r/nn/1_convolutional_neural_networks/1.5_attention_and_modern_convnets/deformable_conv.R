# Native R NN example. Deformable convolutions v1/v2: layer checks against ordinary, shifted and host-bilinear convolutions, compact deformable ResNet-50 counts/derivatives, then DCNv1 and DCNv2 CIFAR-10 training; the C++ example's loop, written out in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.5_attention_and_modern_convnets/deformable_conv.R
# Deformable convolutions v1 / v2: learned sampling offsets (and v2 modulation masks).
#
# Port of tflow's deformable_conv (layer: common/deformable.hpp on
# nn::deformable_conv2d). Check that a new deformable layer (zero offsets)
# equals an ordinary 3x3 convolution with the same weights at strides one and
# two, and that a new modulated layer gives half of it (masks start at
# sigmoid(0)). Check integer offsets (every tap moved one row down reads the
# convolution of the shifted input, with zeros beyond the border) and
# fractional offsets against a host bilinear computation, and derivatives
# with respect to the offsets against central differences. Count the compact
# ResNet-50 with deformable 3x3 convolutions in stages 3 and 4 and check its
# derivatives (training mode, moving averages not updated).
# https://arxiv.org/abs/1703.06211
# https://arxiv.org/abs/1811.11168
#
# Train the compact ResNet-50 of resnet_bottleneck with v1 and with v2
# (modulated) deformable 3x3 convolutions in its last two stages (8x8 and
# 4x4 maps) on CIFAR-10, for comparison with the plain compact ResNet-50. Use
# 12k fit / 5k validation, twelve epochs, crop/flip, Adam(.001), kernel
# L2(.0001) and the best-validation checkpoint before evaluating all 10k test
# images. Weights come from nn's generator, so trained numbers match TF only
# statistically. CIFAR-10 is read from the tflow dataset cache
# (common/paths.hpp) or data=.
# Options: data=PATH epochs=12 batch=128 lr=.001 fit=12000 augment=0|1 threads=N seed=17.
#
# The components (deformable::DeformableConv2D, attention_cnn::AttentionResNet,
# mobile_checks::Pattern and GradientNorms, image batching, cross-entropy and
# L2) are the C++ ones through SWIG (datamungerNN::nn_components); the checks
# and the experiment workflow below mirror examples/cpp/source/nn/.../deformable_conv.cpp
# line by line.
suppressPackageStartupMessages(library(datamungerNN))
check <- nn_components$check
example_runtime <- nn_components$example_runtime
image_training <- nn_components$image_training
paths <- nn_components$paths
attention_cnn <- nn_components$attention_cnn
deformable <- nn_components$deformable
mobile_checks <- nn_components$mobile_checks
Pattern <- mobile_checks$Pattern

K_SIZE <- 7L
K_IN <- 3L
K_OUT <- 4L

# C++ default ostream formatting (six significant digits).
g6 <- function(x) sprintf("%g", x)

# Single-precision rounding, where the C++ holds a `float`.
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

# Host bilinear sample of x [1, H, W, C] (flattened) at (y, x), zero outside.
sample_at <- function(x, y, xpos, c) {
    y0 <- floor(y)
    x0 <- floor(xpos)
    v <- 0
    for (dy in 0:1)
        for (dx in 0:1) {
            yy <- y0 + dy
            xx <- x0 + dx
            if (yy < 0 || yy >= K_SIZE || xx < 0 || xx >= K_SIZE) next
            v <- v + (if (dy) y - y0 else 1 - (y - y0)) * (if (dx) xpos - x0 else 1 - (xpos - x0)) *
                x[(yy * K_SIZE + xx) * K_IN + c + 1L]
        }
    v
}

# Largest absolute single-precision difference between two flattened tensors.
max_difference <- function(a, b) max(0, abs(f32(a - b)))

# A new (zero-offset) deformable layer against the ordinary 3x3 convolution,
# integer and fractional offsets against host sampling, offset derivatives
# against central differences.
layer_checks <- function() {
    x <- Pattern(c(1L, K_SIZE, K_SIZE, K_IN), 1, .2)
    for (stride in c(1L, 2L))
        for (modulated in c(FALSE, TRUE)) {
            if (stride == 2L && modulated) next
            p <- nn$ParameterStore(17)
            layer <- deformable$DeformableConv2D(p, "dcn", x, K_OUT, 3L, stride, modulated)
            o <- nn$conv_options(stride, nn$Padding$Explicit)
            o$explicit_paddings <- c(1L, 1L, 1L, 1L)
            reference <- nn$mul(nn$conv2d(x, p$at("dcn/W"), o), if (modulated) .5 else 1)
            check$Require(identical(as.numeric(reference$shape()), as.numeric(layer$output$shape())),
                          "deformable output shape")
            error <- max_difference(layer$output$values(), reference$values())
            check$Near(error, 0, 1e-5, "a new deformable layer is an ordinary (or half an ordinary) convolution")
            cat(sprintf("  new %sdeformable layer, stride %d: equals %sordinary 3x3 convolution (error %s)\n",
                        if (modulated) "modulated " else "", stride, if (modulated) "half the " else "the ", g6(error)))
        }
    # Offsets set through the offset predictor's bias.
    p <- nn$ParameterStore(17)
    weights <- Pattern(c(1L, K_SIZE, K_SIZE, K_OUT), 1, .6)
    loss_of <- function(bias) {
        if (p$contains("dcn/offset/b"))
            nn$no_grad(function() p$set("dcn/offset/b", nn$from_vector(bias, c(18L))))
        layer <- deformable$DeformableConv2D(p, "dcn", x, K_OUT, 3L, 1L, FALSE)
        list(output = layer$output, loss = nn$sum(nn$mul(layer$output, weights)))
    }
    loss_of(rep(0, 18L))  # create the parameters
    value <- x$values()
    w <- p$at("dcn/W")$values()
    host <- function(b) {
        y <- numeric(K_SIZE * K_SIZE * K_OUT)  # single-precision accumulation, as the C++ host vector
        for (i in 0:(K_SIZE - 1L))
            for (j in 0:(K_SIZE - 1L))
                for (k in 0:8)
                    for (c in 0:(K_IN - 1L)) {
                        v <- sample_at(value, i - 1 + k %/% 3 + b[2L * k + 1L], j - 1 + k %% 3 + b[2L * k + 2L], c)
                        for (o in 0:(K_OUT - 1L)) {
                            index <- (i * K_SIZE + j) * K_OUT + o + 1L
                            y[index] <- f32(y[index] + v * w[(k * K_IN + c) * K_OUT + o + 1L])
                        }
                    }
        y
    }
    down <- rep(0, 18L)
    for (k in 0:8) down[2L * k + 1L] <- 1
    integer <- max_difference(loss_of(down)$output$values(), host(down))
    # Fractional parts in [0.2, 0.8]: central differences then stay on one bilinear piece.
    i <- 0:17
    v <- f32(.37 + .9 * sin(1.7 * i) + ifelse(i %% 3 == 0, .25, 0))
    fraction <- f32(floor(v) + .2 + .6 * (v - floor(v)))
    r <- loss_of(fraction)
    fractional <- max_difference(r$output$values(), host(fraction))
    symbolic <- nn$grad(r$loss, nn$tensor_vector(list(p$at("dcn/offset/b"))))[[1L]]$values()
    derivative <- 0
    nn$no_grad(function() {
        for (i in seq_len(18L)) {
            plus <- fraction
            minus <- fraction
            h <- f32(1e-2)
            plus[i] <- f32(plus[i] + h)
            minus[i] <- f32(minus[i] - h)
            numeric <- f32(f32(f32(loss_of(plus)$loss$item()) - f32(loss_of(minus)$loss$item())) / (2 * h))
            derivative <<- max(derivative, abs(numeric - symbolic[i]) / max(1, abs(numeric)))
        }
    })
    check$Near(integer, 0, 1e-5, "integer offsets read the shifted input")
    check$Near(fractional, 0, 1e-5, "fractional offsets interpolate bilinearly")
    check$Near(derivative, 0, 2e-3, "offset derivatives match central differences")
    cat(sprintf(paste0("  offsets: all taps one row down matches the shifted convolution (error %s), fractional offsets ",
                       "match host bilinear sampling (%s), offset derivatives match central differences (relative %s)\n"),
                g6(integer), g6(fractional), g6(derivative)))
}

# The compact ResNet-50 of resnet_bottleneck with deformable 3x3 convolutions in stages 3 and 4.
compact_config <- function(modulated) {
    cfg <- attention_cnn$AttentionResNetConfig()
    cfg$classes <- 10L
    cfg$width_divisor <- 4L
    cfg$small_stem <- TRUE
    cfg$stride_on_3x3 <- FALSE
    cfg$deformable_from <- 3L
    cfg$modulated <- modulated
    cfg
}

# Parameter counts and finite nonzero derivatives of the compact deformable
# ResNet-50 (v1 and modulated v2), training mode with frozen moving averages.
network_checks <- function() {
    for (modulated in c(FALSE, TRUE)) {
        p <- nn$ParameterStore(17)
        state <- nn$ParameterStore(17)
        cfg <- compact_config(modulated)
        cfg$update_stats <- FALSE
        net <- attention_cnn$AttentionResNet(p, state, Pattern(c(2L, 32L, 32L, 3L), 1, 0), cfg, TRUE)
        # The plain compact ResNet-50 (1,484,186) plus offset predictors: 3x3 convolutions from the
        # 64 / 128 channels of the six stage-3 and three stage-4 layers to 18 (v1) or 27 (v2) maps.
        maps <- if (modulated) 27 else 18
        expected <- 1484186 + 6 * (9 * 64 * maps + maps) + 3 * (9 * 128 * maps + maps)
        check$Require(p$count() == expected, "compact deformable ResNet-50 parameter count")
        norms <- as.numeric(mobile_checks$GradientNorms(p, nn$mean(nn$square(net$logits))))
        names <- as.character(p$names())
        minimum <- Inf
        maximum <- 0
        smallest <- ""
        for (i in seq_along(norms)) {
            check$Require(is.finite(norms[i]) && norms[i] > 0, "finite nonzero deformable ResNet derivative")
            if (norms[i] < minimum) {
                minimum <- norms[i]
                smallest <- names[i]
            }
            maximum <- max(maximum, norms[i])
        }
        cat(sprintf(paste0("  compact ResNet-50 with %s deformable layers in stages 3-4: %.0f parameters; ",
                           "derivatives in [%s,%s] (smallest %s)\n"), if (modulated) "v2 (modulated)" else "v1",
                    p$count(), g6(minimum), g6(maximum), smallest))
    }
}

# Fits the compact deformable ResNet-50 (plain DCNv1 or modulated DCNv2
# offsets) on CIFAR-10: fixed development split, Adam on cross-entropy plus
# L2, validation-NLL checkpoint selection, one test pass.
train_deformable_resnet <- function(modulated, development, test, cfg, explicit_epochs) {
    cfg$name <- if (modulated) "dcnv2_resnet50_compact_cifar10" else "dcnv1_resnet50_compact_cifar10"
    model <- function(p, state, image, training)
        attention_cnn$AttentionResNet(p, state, image, compact_config(modulated), training)$logits

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
    layer_checks()
    network_checks()
    cfg <- image_training$ImageTrainingConfig()
    cfg$maximum_fit_rows <- 12000L
    cfg$epochs <- 12L
    cfg$minimum_test_accuracy <- .4
    options <- parse_cifar_options(arguments, cfg)
    data <- image_training$LoadCifar10(options$directory)
    for (modulated in c(FALSE, TRUE))
        train_deformable_resnet(modulated, data$first, data$second, cfg, options$explicit_epochs)
    cat("All deformable convolution checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
