# Image-classifier training pieces with per-image random crop/flip augmentation
# (examples/nn/common/image_training.hpp). Images stay NHWC; pixels map from
# [0,1] to [-1,1]. Training can zero-pad by four pixels, independently crop back
# and flip horizontally (nn$augment$random_pad_crop from nn$Generator(151)).
# The objective is (optionally label-smoothed) mean sparse cross-entropy plus
# 0.5 * l2 * sum(W^2) over parameters of rank >= 2, with optional weighted
# auxiliary heads, extra penalties or a custom loss (image_training_network).
# Model builders: model(params, state, x, training) -> logits. Training loops,
# checkpoint selection and reporting stay in each example.

# Normalized images and labels for 0-based `rows`; with a generator, pad-4
# random crops and horizontal flips per image. Returns list(x, y).
image_training_image_batch <- function(dataset, rows, augmentation = NULL) {
    if (dataset$images$dim() != 4L || dataset$images$numel() == 0)
        stop("image batching requires nonempty NHWC data")
    count <- as.numeric(dataset$images$shape())[1L]
    if (any(rows < 0 | rows >= count))
        stop("image batch row")
    x <- nn$take_rows(dataset$images, rows)
    if (!is.null(augmentation))
        x <- nn$augment$random_pad_crop(x, 4L, augmentation, TRUE, 0)
    list(x = nn$sub(nn$mul(x, 2), 1), y = nn$take_rows(dataset$labels, rows))
}

# Cross-entropy, accuracy and confusion of inference-mode parameters over
# `rows`, in batches of 256: list(loss, accuracy, confusion).
image_training_evaluate_images <- function(params, state, model, dataset, rows, classes) {
    result <- list(loss = nn$WeightedMean(), accuracy = nn$WeightedMean(), confusion = nn$ConfusionMatrix(classes))
    nn$no_grad(function() {
        for (start in seq.int(1L, length(rows), by = 256L)) {
            ids <- rows[start:min(start + 255L, length(rows))]
            batch <- image_training_image_batch(dataset, ids)
            logits <- model(params, state, batch$x, FALSE)
            prediction <- nn$argmax(logits, 1L)
            result$loss$add(nn$sparse_softmax_cross_entropy(logits, batch$y)$item(), length(ids))
            result$accuracy$add(nn$accuracy(prediction, batch$y), length(ids))
            result$confusion$add(batch$y, prediction)
        }
    })
    result
}

# image_training::ImageTrainingConfig (maximum_fit_rows = 0 uses every non-validation row).
image_training_config <- function(...) {
    cfg <- list(name = "", classes = 10L, epochs = 10L, batch = 128L, validation_rows = 5000L,
                maximum_fit_rows = 0L, learning_rate = 0.001, l2 = 1e-4, label_smoothing = 0,
                augment = TRUE, minimum_test_accuracy = 0.55, seed = 17)
    values <- list(...)
    unknown <- setdiff(names(values), names(cfg))
    if (length(unknown)) stop("unknown image training setting: ", unknown[1L])
    cfg[names(values)] <- values
    cfg
}

# Mean sparse cross-entropy with optional label smoothing.
image_training_cross_entropy <- function(logits, labels, classes, smoothing) {
    if (smoothing <= 0) return(nn$sparse_softmax_cross_entropy(logits, labels))
    targets <- nn$add(nn$mul(nn$one_hot(labels, classes, 1, 0, logits$dtype()), 1 - smoothing), smoothing / classes)
    nn$neg(nn$mean(nn$sum(nn$mul(targets, nn$log_softmax(logits, 1L)), 1L)))
}

# TrainingNetwork: training logits, list(list(aux_logits, weight)), extra scalar penalties, optional loss(labels).
image_training_network <- function(logits, auxiliary = list(), penalties = list(), loss = NULL)
    list(logits = logits, auxiliary = auxiliary, penalties = penalties, loss = loss)

# sum(W^2) over parameters of rank >= 2 in creation order, or NULL.
image_training_l2_penalty <- function(params) {
    penalty <- NULL
    for (p in nn$tensor_list(params$parameters())) if (p$dim() >= 2L) {
        term <- nn$sum(nn$square(p))
        penalty <- if (is.null(penalty)) term else nn$add(penalty, term)
    }
    penalty
}

# (Smoothed) cross-entropy of the training logits + weighted auxiliary heads +
# penalties + 0.5 * l2 * sum(W^2). `training(params, state, x)` optionally
# returns an image_training_network(); otherwise model(..., TRUE) gives logits.
image_training_objective <- function(params, state, model, x, y, cfg, training = NULL) {
    network <- if (is.null(training)) image_training_network(model(params, state, x, TRUE))
               else training(params, state, x)
    objective <- if (!is.null(network$loss)) network$loss(y)
                 else image_training_cross_entropy(network$logits, y, cfg$classes, cfg$label_smoothing)
    for (head in network$auxiliary) {
        weight <- head[[2L]]
        if (!is.finite(weight) || weight <= 0) stop("auxiliary loss weight must be positive and finite")
        objective <- nn$add(objective, nn$mul(image_training_cross_entropy(head[[1L]], y, cfg$classes,
                                                                           cfg$label_smoothing), weight))
    }
    for (penalty in network$penalties) objective <- nn$add(objective, penalty)
    if (cfg$l2 > 0) {
        penalty <- image_training_l2_penalty(params)
        if (!is.null(penalty)) objective <- nn$add(objective, nn$mul(penalty, 0.5 * cfg$l2))
    }
    objective
}

# Option defaults of image_training::ParseCifarOption, for nn_options().
image_training_cifar_option_defaults <- function(data)
    list(data = data, epochs = 10, batch = 128, lr = 0.001, fit = 0, augment = 1, threads = 0, seed = 17)

# Copies parsed ParseCifarOption values into cfg (threads = 0 keeps nn's default).
image_training_apply_cifar_options <- function(args, cfg) {
    if (!args$augment %in% c(0, 1)) stop("augment must be 0 or 1")
    cfg$epochs <- as.integer(args$epochs)
    cfg$batch <- as.integer(args$batch)
    cfg$learning_rate <- args$lr
    cfg$maximum_fit_rows <- as.integer(args$fit)
    cfg$augment <- args$augment == 1
    cfg$seed <- args$seed
    if (args$threads > 0) nn$set_num_threads(as.integer(args$threads))
    cfg
}

# The five official development batches and the separate test batch: list(development, test).
image_training_load_cifar10 <- function(directory) {
    development <- nn_components$dataset_assets$LoadCifar10(directory, "train")
    test <- nn_components$dataset_assets$LoadCifar10(directory, "test")
    if (!identical(as.numeric(development$images$shape()), c(50000, 32, 32, 3)) ||
        !identical(as.numeric(test$images$shape()), c(10000, 32, 32, 3)))
        stop("official CIFAR-10 split sizes")
    list(development = development, test = test)
}
