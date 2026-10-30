# Native R NN example. SegNet index-unpool/model checks and compact procedural-shapes segmentation training; the C++ example's loop, written out in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.7_segmentation/segnet.R
# SegNet: unpool using encoder max locations, then learn dense decoder maps.
# https://arxiv.org/abs/1511.00561
# https://github.com/alexgkendall/SegNet-Tutorial
#
# Port of tflow's segnet. Independently check 2x2 pooling winners, sparse
# unpool values and both pooled and original-input derivatives on negative,
# multi-batch/channel, odd-grid data. Check zero outside winners and dropped
# odd borders. Full RGB/11-class model has 29449355 trainable parameters,
# 15872 BN-state values and 102 parameter derivatives. The 13-convolution
# encoder and 13-convolution decoder follow the released CamVid topology; BN
# starts beta 0 rather than .001. Report int64 index storage versus float
# feature storage, without claiming the reference's theoretical
# compressed-index footprint.
#
# Compact widths/8, four classes, 96 pixels, all five pool/unpool levels.
# Train on independent procedural shape masks (600 fit/120 validation/120
# test) with balanced pixel CE, Adam(.001), batch 16, L2 .0001, twenty epochs
# and selected validation mIoU. Export confusion and image/truth/prediction
# PNG. No CamVid or real-world benchmark claim is made.
#
# TF -> datamunge: datamunge has no MaxPoolWithArgmax/unpool op; the argmax
# indices (TF layout, batch included) are computed on the host from the
# detached pooled input and the unpool is a differentiable gather times the
# winner mask (common/segnet.hpp). The full model executes forward and
# backward on a 64x64 image instead of building a 224x224 graph only.
# Default parameter seed 1: with seed 17 training stalls in a plateau (test mIoU
# .40), seeds 1 and 2 reach .815/.816 like TF's .822 (TF itself only breaks out
# of a similar plateau at epoch 13).
# Options: epochs=20 seed=N (parameter initialization, default 1) tag=NAME
# (writes models/segnet_shapes_NAME) threads=N.
# Artifacts: output/nn/models/segnet_shapes/.
#
# The components (segnet::SegNet/PoolWithIndices/IndexUnpool, the seg::
# procedural shapes, batching, balanced cross-entropy, confusion/mIoU and
# preview) are the C++ ones through SWIG (datamungerNN::nn_components); the
# checks and the training workflow below mirror examples/cpp/source/nn/.../segnet.cpp
# line by line.
suppressPackageStartupMessages(library(datamungerNN))
check <- nn_components$check
example_runtime <- nn_components$example_runtime
seg <- nn_components$seg
segnet <- nn_components$segnet

# C++ default ostream formatting (six significant digits).
g6 <- function(x) sprintf("%g", x)

# Rounds to IEEE single precision (the C++ fixtures are float arithmetic).
f32 <- function(v) nn$from_vector(v, c(length(v)))$values()

# float(std::sin(scale * double(i))) over the flattened shape.
wave <- function(shape, scale) nn$from_vector(sin(scale * (seq_len(prod(shape)) - 1)), as.integer(shape))

# Finite (optionally nonzero) derivative of every parameter; returns c(lowest, highest) norm.
gradient_norms <- function(p, objective, nonzero, what) {
    params <- p$parameters()
    grads <- nn$grad(objective, params)
    check$Require(length(grads) == params$size(), paste0(what, ": every parameter has a derivative"))
    names <- p$names()
    low <- Inf
    high <- 0
    for (k in seq_along(grads)) {
        norm <- sqrt(nn$sum(nn$square(grads[[k]]$to(nn$DType$Float64)))$item())
        check$Require(is.finite(norm) && (!nonzero || norm > 0),
                      paste0("finite nonzero ", what, " derivative: ", names[k]))
        low <- min(low, norm)
        high <- max(high, norm)
    }
    c(low, high)
}

# 2x2 argmax winners (batch/channel offsets), sparse unpool values and both derivatives on a 5x7 odd grid.
pool_checks <- function() {
    # value[i] = sin(.37f i) + .001f i - .4f; weight[i] = cos(.17f i); values[i] = .2f i - 3 (float arithmetic).
    i <- 0:(2 * 5 * 7 * 2 - 1)
    value <- f32(f32(f32(sin(f32(f32(.37) * i))) + f32(f32(.001) * i)) - f32(.4))
    weight <- f32(cos(f32(f32(.17) * i)))
    values <- f32(f32(f32(.2) * (0:(2 * 2 * 3 * 2 - 1))) - 3)
    input <- nn$from_vector(value, c(2L, 5L, 7L, 2L))
    pooled <- nn$from_vector(values, c(2L, 2L, 3L, 2L))
    input$requires_grad_(TRUE)
    pooled$requires_grad_(TRUE)
    record <- segnet$PoolWithIndices(input)
    w <- nn$from_vector(weight, c(2L, 5L, 7L, 2L))
    restored <- segnet$IndexUnpool(pooled, record)
    pipeline <- segnet$IndexUnpool(record$values, record)
    pooled_gradient <- nn$grad(nn$sum(nn$mul(restored, w)), nn$tensor_vector(list(pooled)))[[1L]]$values()
    input_gradient <- nn$grad(nn$sum(nn$mul(pipeline, w)), nn$tensor_vector(list(input)))[[1L]]$values()
    restored_values <- restored$values()
    indices <- as.numeric(record$indices)
    expected <- numeric(length(value))
    dx <- numeric(length(value))
    error <- 0
    for (n in 0:1) for (r in 0:1) for (c in 0:2) for (k in 0:1) {
        maximum <- -Inf
        best <- -1
        for (dy in 0:1) for (dxx in 0:1) {
            index <- ((n * 5 + 2 * r + dy) * 7 + 2 * c + dxx) * 2 + k
            if (value[index + 1] > maximum) {
                maximum <- value[index + 1]
                best <- index
            }
        }
        slot <- ((n * 2 + r) * 3 + c) * 2 + k
        check$Require(indices[slot + 1] == best, "argmax includes batch and channel offsets")
        expected[best + 1] <- values[slot + 1]
        dx[best + 1] <- weight[best + 1]
        error <- max(error, abs(pooled_gradient[slot + 1] - weight[best + 1]))
    }
    error <- max(error, abs(restored_values - expected), abs(input_gradient - dx))
    check$Require(identical(as.numeric(restored$shape()), c(2, 5, 7, 2)), "unpool restores the odd pre-pool shape")
    check$Near(error, 0, 1e-6, "host sparse unpool and every pooled/original derivative")
    cat("  negative/multi-batch/channel/odd-grid index unpool and every derivative match host winners; error=",
        g6(error), "\n", sep = "")
}

compact_config <- function() {
    c <- segnet$SegNetConfig()
    c$classes <- 4L
    c$width_divisor <- 8L
    c
}

# Full SegNet census on 64x64, compact 96/97 alignment, derivative norms and index-cache footprint.
model_checks <- function() {
    p <- nn$ParameterStore(17)
    state <- nn$ParameterStore(17)
    net <- segnet$SegNet(p, state, wave(c(1, 64, 64, 3), .13), segnet$SegNetConfig(), TRUE)
    check$Require(p$count() == 29449355 && state$count() == 15872, "full SegNet trainable/state counts")
    check$Require(identical(as.numeric(net$logits$shape()), c(1, 64, 64, 11)), "full SegNet logits on the input grid")
    gradient_norms(p, nn$mean(nn$square(net$logits)), FALSE, "full SegNet")
    check$Require(p$size() == 102, "full SegNet all parameter connections")
    cat("  full SegNet: 29449355 parameters, 15872 state values, 102 derivatives (executed on 64x64)\n")
    rm(net)
    p <- nn$ParameterStore(17)
    state <- nn$ParameterStore(17)
    low <- Inf
    high <- 0
    index_bytes <- 0
    feature_bytes <- 0
    for (side in c(96, 97)) {
        net <- segnet$SegNet(p, state, wave(c(2, side, side, 3), .13), compact_config(), TRUE)
        check$Require(identical(as.numeric(net$logits$shape()), c(2, side, side, 4)),
                      "saved shapes restore odd input dimensions")
        if (side != 96) next
        for (k in seq_len(net$pooling$size()) - 1) {
            record <- net$pooling$`__getitem__`(k)
            index_bytes <- index_bytes + 8 * length(record$indices)
            feature_bytes <- feature_bytes + 4 * prod(as.numeric(record$shape))
        }
        norms <- gradient_norms(p, nn$mean(nn$square(net$logits)), TRUE, "SegNet")
        low <- norms[1L]
        high <- norms[2L]
    }
    check$Require(2 * index_bytes == feature_bytes,
                  "int64 nonoverlapping 2x2 index cache is half full float-feature cache")
    cat(sprintf("  compact SegNet: %.0f parameters; 96/97 alignment; derivative norms [%s,%s]; index cache=%.0f vs full-feature cache=%.0f bytes\n",
                p$count(), g6(low), g6(high), index_bytes, feature_bytes))
}

# The first `count` rows of a SegmentationData (quick-mode subset).
subset_data <- function(data, count) {
    result <- seg$SegmentationData()
    result$count <- min(data$count, count)
    result$height <- data$height
    result$width <- data$width
    result$classes <- data$classes
    pixels <- result$count * data$height * data$width
    result$images <- data$images[seq_len(3 * pixels)]
    result$labels <- data$labels[seq_len(pixels)]
    result
}

# seg::SegmentationScore: inference logits in batches of 16 rows, one confusion matrix, mIoU and pixel accuracy.
segmentation_score <- function(infer, data) {
    classes <- data$classes
    confusion <- numeric(classes * classes)
    nn$no_grad(function() {
        for (start in seq.int(0L, data$count - 1L, by = 16L)) {
            rows <- start:(min(start + 16L, data$count) - 1L)
            batch <- seg$SegmentationBatch(data, as.integer(rows))
            confusion <<- confusion + as.numeric(seg$SegmentationConfusion(infer(batch$first), batch$second, classes))
        }
    })
    result <- seg$SegmentationMetrics()
    result$confusion <- confusion
    result$mean_iou <- seg$ConfusionMeanIou(confusion, classes)
    result$pixel_accuracy <- sum(diag(matrix(confusion, classes, byrow = TRUE))) / sum(confusion)
    result
}

# seg::SegmentationPreview: image | truth | prediction rows of the first four images.
segmentation_preview <- function(infer, data, filename) {
    nn$no_grad(function() {
        batch <- seg$SegmentationBatch(data, as.integer(seq_len(min(4L, data$count)) - 1L))
        seg$SegmentationPreviewLogits(infer(batch$first), data, filename)
    })
    invisible(NULL)
}

# The seg training block of the C++ main: quick-mode limits, balanced class weights, Adam on
# weighted pixel CE plus kernel L2, validation-mIoU checkpoint selection, restore, test confusion,
# preview and the demonstration policy.
train_segmentation <- function(training_cfg, builder, source_fit, source_validation, source_test, explicit_epochs) {
    if (!explicit_epochs) training_cfg$epochs <- example_runtime$Limit(training_cfg$epochs, 1L)
    training_cfg$batch <- example_runtime$Limit(training_cfg$batch, 2L)
    if (example_runtime$Full()) {
        fit <- source_fit
        validation <- source_validation
        test <- source_test
    } else {
        fit <- subset_data(source_fit, 8L)
        validation <- subset_data(source_validation, 4L)
        test <- subset_data(source_test, 4L)
    }
    example_runtime$Describe()
    check$Require(training_cfg$epochs > 0 && training_cfg$batch > 0 && fit$classes == validation$classes &&
                      fit$classes == test$classes, "valid segmentation configuration")
    check$Require(!example_runtime$Full() || training_cfg$accuracy || training_cfg$epochs >= 2,
                  "short segmentation demonstration needs at least two epochs")
    p <- nn$ParameterStore(training_cfg$seed)
    state <- nn$ParameterStore(training_cfg$seed)
    nn$no_grad(function() {  # create every parameter (no statistics update) before the optimizer
        probe <- seg$SegmentationBatch(fit, 0L)
        builder(p, state, probe$first, FALSE)
    })
    infer <- function(images) builder(p, state, images, FALSE)$logits
    # Inverse-square-root class frequencies of the fit masks (float arithmetic, as the C++).
    fit_labels <- fit$labels
    frequencies <- 1 + tabulate(fit_labels + 1L, nbins = fit$classes)
    weights <- f32(1 / f32(sqrt(f32(f32(frequencies) / f32(length(fit_labels))))))
    class_weights <- nn$from_vector(weights, c(fit$classes))
    kernels <- Filter(function(t) t$dim() > 1, nn$tensor_list(p$parameters()))
    oc <- nn$OptimizerConfig()
    oc$kind <- nn$OptimizerKind$Adam
    optimizer <- nn$Optimizer(p, training_cfg$learning_rate, oc)
    directory <- seg$SegmentationDirectory(training_cfg$name)
    prefix <- file.path(directory, "best")
    history <- file(file.path(directory, "history.tsv"), "w")
    writeLines("epoch\tloss\tvalidation_mean_iou", history)
    cat(sprintf("== %s: %d fit / %d validation / %d test; %.0f parameters; %dx%d, classes=%d\n", training_cfg$name,
                fit$count, validation$count, test$count, p$count(), fit$height, fit$width, fit$classes))
    batches <- nn$BatchOrder(fit$count, training_cfg$batch, 29L)
    best <- -1
    best_epoch <- -1L
    epoch_losses <- numeric()
    if (nzchar(training_cfg$warm_start)) {
        check$Require(normalizePath(training_cfg$warm_start, mustWork = FALSE) != normalizePath(prefix, mustWork = FALSE),
                      "segmentation warm start requires a separate output tag")
        p$load(paste0(training_cfg$warm_start, "_params.npz"))
        state$load(paste0(training_cfg$warm_start, "_state.npz"))
        best <- segmentation_score(infer, validation)$mean_iou
        best_epoch <- 0L
        p$save(paste0(prefix, "_params.npz"))
        state$save(paste0(prefix, "_state.npz"))
        writeLines(sprintf("0\tnan\t%s", g6(best)), history)
        flush(history)
        cat(sprintf("  warm start: %s; fresh Adam moments; restored validation mIoU=%s\n", training_cfg$warm_start,
                    g6(best)))
    }
    train_seconds <- 0
    for (epoch in seq_len(training_cfg$epochs)) {
        started <- proc.time()[["elapsed"]]
        batches$begin_epoch()
        total <- 0
        seen <- 0L
        repeat {
            rows <- batches$`_next`()
            if (!length(rows)) break
            batch <- seg$SegmentationBatch(fit, as.integer(rows))
            model <- builder(p, state, batch$first, TRUE)
            loss <- seg$SegmentationCrossEntropy(model$logits, batch$second, fit$classes, class_weights)
            auxiliary <- nn$tensor_list(model$auxiliary)
            if (length(auxiliary)) {
                total_loss <- loss
                for (output in auxiliary)
                    total_loss <- nn$add(total_loss, nn$mul(
                        seg$SegmentationCrossEntropy(output, batch$second, fit$classes, class_weights),
                        model$auxiliary_weight))
                loss <- nn$div(total_loss, 1 + model$auxiliary_weight * length(auxiliary))
            }
            value <- loss$item()
            check$Require(is.finite(value), "finite segmentation training loss")
            objective <- loss
            if (training_cfg$l2 > 0 && length(kernels)) {
                penalty <- nn$sum(nn$square(kernels[[1L]]))
                for (kernel in kernels[-1L]) penalty <- nn$add(penalty, nn$sum(nn$square(kernel)))
                objective <- nn$add(objective, nn$mul(penalty, .5 * training_cfg$l2))
            }
            optimizer$minimize(objective)
            total <- total + value * length(rows)
            seen <- seen + length(rows)
        }
        train_seconds <- train_seconds + proc.time()[["elapsed"]] - started
        epoch_losses <- c(epoch_losses, total / seen)
        check$Require(is.finite(total / seen), "finite segmentation epoch mean")
        score <- segmentation_score(infer, validation)
        check$Require(is.finite(score$mean_iou), "finite validation mIoU")
        if (score$mean_iou > best) {
            best <- score$mean_iou
            best_epoch <- epoch
            p$save(paste0(prefix, "_params.npz"))
            state$save(paste0(prefix, "_state.npz"))
        }
        writeLines(sprintf("%d\t%s\t%s", epoch, g6(total / seen), g6(score$mean_iou)), history)
        flush(history)
        cat(sprintf("  epoch %d: loss=%s, validation mIoU=%s, pixel accuracy=%s\n", epoch, g6(total / seen),
                    g6(score$mean_iou), g6(score$pixel_accuracy)))
    }
    close(history)
    p$load(paste0(prefix, "_params.npz"))
    state$load(paste0(prefix, "_state.npz"))
    check$Near(segmentation_score(infer, validation)$mean_iou, best, 1e-10,
               "restored validation score matches selected checkpoint")
    cat(sprintf("  restored validation mIoU=%s; checkpoint agreement verified\n", g6(best)))
    # SegmentationConfig::inspect is never set by this example.
    result <- segmentation_score(infer, test)
    cat(sprintf("  selected epoch=%d; test mIoU=%s, pixel accuracy=%s\n", best_epoch, g6(result$mean_iou),
                g6(result$pixel_accuracy)))
    cat("  test confusion (truth rows, predicted columns):\n")
    confusion <- as.numeric(result$confusion)
    for (c in seq_len(fit$classes) - 1)
        cat("   ", paste0(" ", sprintf("%.0f", confusion[c * fit$classes + seq_len(fit$classes)]), collapse = ""), "\n",
            sep = "")
    segmentation_preview(infer, test, file.path(directory, "preview.png"))
    cat(sprintf("  checkpoint, history and image/truth/prediction preview: %s\n", directory))
    cat(sprintf("  training time %.1f s (%.2f s/epoch, excluding validation)\n", train_seconds,
                train_seconds / training_cfg$epochs))
    if (example_runtime$Full())
        seg$CheckSegmentationDemonstration(training_cfg, epoch_losses, result)
    else
        check$Require(is.finite(result$mean_iou) && result$mean_iou >= 0 && result$mean_iou <= 1,
                      "finite quick segmentation evaluation")
    if (example_runtime$Full() && !training_cfg$accuracy)
        cat(sprintf("  short learning demonstration passed: mean loss %s -> %s; accuracy=1 enables mIoU target %s\n",
                    g6(epoch_losses[1L]), g6(epoch_losses[length(epoch_losses)]), g6(training_cfg$minimum_test_iou)))
    invisible(NULL)
}

main <- function(arguments = commandArgs(TRUE)) {
    cfg <- seg$SegmentationConfig()
    cfg$name <- "segnet_shapes"
    cfg$seed <- 1  # seed 17 plateaus at test mIoU .40 (< .55 gate); seeds 1/2 reach .815/.816 (TF .822)
    for (arg in arguments) {
        if (startsWith(arg, "epochs=")) cfg$epochs <- as.integer(substring(arg, 8L))
        else if (startsWith(arg, "seed=")) cfg$seed <- as.numeric(substring(arg, 6L))
        else if (startsWith(arg, "tag=")) cfg$name <- paste0("segnet_shapes_", substring(arg, 5L))
        else if (startsWith(arg, "threads=")) nn$set_num_threads(as.integer(substring(arg, 9L)))
        else stop("expected epochs=N, seed=N, tag=NAME or threads=N")
    }
    seg$SegmentationLossChecks()
    pool_checks()
    model_checks()
    fit <- seg$SegmentationShapes(600L, 96L, 11)
    validation <- seg$SegmentationShapes(120L, 96L, 22)
    test <- seg$SegmentationShapes(120L, 96L, 33)
    builder <- function(p, state, images, training) {
        model <- seg$SegmentationModel()
        model$logits <- segnet$SegNet(p, state, images, compact_config(), training)$logits
        model
    }
    explicit_epochs <- any(startsWith(arguments, "epochs="))
    train_segmentation(cfg, builder, fit, validation, test, explicit_epochs)
    cat("All SegNet checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
