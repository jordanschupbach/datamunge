# Native R NN example. LeNet-5 architecture checks and MNIST fitting run in R.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# CMD: just r-example lenet5
# LeNet-5: sparse convolutions, learned subsampling and a fixed RBF head.
#
# Port of tflow's lenet5. Preparation (in ~/projects/tflow_examples):
# python3 scripts/fetch_datasets.py mnist --split both.
# Preserve the 1998 architecture's C1/S2/C3/S4/C5/F6 dimensions, 60 sparse C3
# connections, scaled tanh activations and exactly 60,000 learned parameters.
# Use our own fixed seven-segment 7x12 digit prototypes for the RBF output.
# Adam and cross entropy on negative squared distances are modern training
# adaptations; this is not a reproduction of the original training recipe.
# Reference: https://cognitivemedium.com/assets/rmnist/LeCun1998.pdf
#
# Check stage shapes/counts, C3's absent-channel derivatives and the RBF
# classifier's prototype predictions before fitting the official MNIST data.
# Fit 55k rows, select on 5k validation rows, then evaluate all 10k test rows.
# Default eight epochs, lr=.001, batch=256; require 97.5% test accuracy.
# Weights initialise from nn's generator (seed 17), so the trajectory matches
# TF only statistically. Options: data=PATH epochs=8 batch=256 lr=.001 threads=N.
# Quick mode uses 64 fit / 32 validation / 32 test rows, batch <= 16 and one
# epoch unless epochs= is given. Checkpoint and validation history: output/nn/models/lenet5/.

nn_helper("checks")
nn_helper("lenet")
nn_helper("mnist")

g6 <- function(value) sprintf("%.6g", value)

architecture_checks <- function() {
    p <- nn$ParameterStore(17L)
    pixels <- nn$ones(c(2L, 784L))
    network <- lenet_lenet5(p, pixels)
    nn_require(p$count() == 60000, "LeNet-5 has exactly 60000 learned parameters")
    stage_names <- c("c1", "s2", "c3", "s4", "c5", "f6")
    counts <- c(156, 12, 1516, 32, 48120, 10164)
    for (stage in seq_along(stage_names)) {
        count <- 0
        for (name in p$names())
            if (startsWith(name, paste0(stage_names[stage], "/"))) count <- count + p$at(name)$numel()
        nn_require(count == counts[stage], paste0("LeNet stage parameter count: ", stage_names[stage]))
    }
    connections <- sum(lengths(lenet_c3_connections()))
    nn_require(connections == 60, "C3 has 60 channel connections")
    # Map 0 reads channels 0,1,2 only. Its derivative must be exactly zero for
    # channels 3,4,5, while at least one derivative in each connected map lives.
    c3_input <- nn$full(c(1L, 14L, 14L, 6L), 0.2)
    c3_input$requires_grad_()
    c3 <- lenet_c3(p, "c3", c3_input)
    probe <- nn$sum(nn$slice_axis(c3, 3L, 0L, 1L))
    gradient <- nn$grad(probe, list(c3_input))[[1L]]$values()
    prototype_logits <- lenet_rbf_logits(lenet_digit_codes())

    stages <- list(list(network$image, c(2, 32, 32, 1)), list(network$c1, c(2, 28, 28, 6)),
                   list(network$s2, c(2, 14, 14, 6)), list(network$c3, c(2, 10, 10, 16)),
                   list(network$s4, c(2, 5, 5, 16)), list(network$c5, c(2, 1, 1, 120)),
                   list(network$f6, c(2, 84)), list(network$logits, c(2, 10)))
    for (stage in stages)
        nn_require(identical(as.numeric(stage[[1L]]$shape()), stage[[2L]]), "LeNet stage shape")
    nn_near(network$image$at(c(0L, 0L, 0L, 0L)), -0.1, 1e-7, "padding background normalization")
    nn_near(network$image$at(c(0L, 2L, 2L, 0L)), 1.175, 1e-7, "foreground normalization")
    # Values are NHWC with 6 channels innermost: channel = (index - 1) %% 6.
    magnitude <- vapply(0:5, function(channel) sum(abs(gradient[seq(channel + 1L, length(gradient), by = 6L)])),
                        numeric(1))
    for (channel in 0:5) {
        if (channel < 3)
            nn_require(magnitude[channel + 1L] > 0, "connected C3 channel has a gradient")
        else
            nn_near(magnitude[channel + 1L], 0, 0, "disconnected C3 channel has no gradient")
    }
    for (a in 0:9) for (b in 0:9) {
        logit <- prototype_logits$at(c(a, b))
        if (a == b)
            nn_near(logit, 0, 0, "RBF self distance")
        else
            nn_require(logit < 0, "distinct RBF prototypes classify themselves uniquely")
    }
    cat("  architecture checks: stage shapes/counts, sparse C3 derivatives and RBF prototypes passed\n")
}

lenet5_logits <- function(store, x, training = FALSE) lenet_lenet5(store, x)$logits

main <- function(arguments = commandArgs(TRUE)) {
    architecture_checks()
    name <- "lenet5"
    default_epochs <- 8L
    minimum_test_accuracy <- 0.975
    args <- nn_options(arguments, list(data = nn_dataset("mnist"), epochs = 0L, batch = 256L, lr = 0.001,
                                       threads = 0L, full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    Sys.setenv(DATAMUNGE_EXAMPLES_FULL = if (isTRUE(args$full)) "1" else "0")
    explicit_epochs <- args$epochs != 0L
    epochs <- if (explicit_epochs) args$epochs else default_epochs
    batch <- args$batch
    rate <- args$lr
    if (args$threads > 0L) nn$set_num_threads(args$threads)
    nn_require(epochs > 0L && batch > 0L && is.finite(rate) && rate > 0,
               "positive epochs, batch and learning rate required")
    if (!explicit_epochs) epochs <- nn_limit(epochs, 1L)  # an explicit budget is honored
    batch <- nn_limit(batch, 16L)
    nn_describe()
    nn_require(file.exists(file.path(args$data, "train-images-idx3-ubyte")) ||
                   file.exists(file.path(args$data, "train-images-idx3-ubyte.gz")),
               paste0("MNIST missing in ", args$data, "; run python3 scripts/fetch_datasets.py mnist --split both ",
                      "in tflow_examples, or set data="))
    train <- nn$load_mnist(args$data, "train")
    test <- nn$load_mnist(args$data, "test")
    nn_require(identical(as.numeric(train$images$shape()), c(60000, 28, 28, 1)) &&
                   identical(as.numeric(test$images$shape()), c(10000, 28, 28, 1)),
               "official MNIST dimensions required")
    train_x <- nn$reshape(train$images, c(60000L, 784L))
    test_x <- nn$reshape(test$images, c(10000L, 784L))

    order <- mnist_development_order(60000L, 2026L)
    fit_rows <- order[seq_len(55000L)]
    validation_rows <- order[55001:60000]
    fit_rows <- head(fit_rows, nn_limit(length(fit_rows), 64L))
    validation_rows <- head(validation_rows, nn_limit(length(validation_rows), 32L))
    test_rows <- seq_len(nn_limit(10000L, 32L)) - 1L

    l2 <- 1e-4
    store <- nn$ParameterStore(17L)
    nn$no_grad(function() lenet5_logits(store, nn$slice_axis(train_x, 0L, 0L, 1L), TRUE))
    config <- nn$OptimizerConfig()
    config$kind <- nn$OptimizerKind$Adam
    optimizer <- nn$Optimizer(store, rate, config)
    weights <- Filter(function(p) p$dim() >= 2L, nn$tensor_list(store$parameters()))

    model_dir <- nn_output_dir(file.path("models", name))
    checkpoint <- file.path(model_dir, "best.npz")
    history <- file(file.path(model_dir, "history.tsv"), "w")
    writeLines("epoch\tvalidation_loss\tvalidation_accuracy", history)
    cat(sprintf("== %s: %d fit / %d validation / %d test\n", name, length(fit_rows), length(validation_rows),
                length(test_rows)))
    cat(sprintf("  parameters=%.0f; batch=%d; lr=%s\n", store$count(), batch, g6(rate)))

    batches <- nn$BatchOrder(length(fit_rows), batch, 29L)
    best <- Inf
    best_epoch <- -1L
    started <- proc.time()[["elapsed"]]
    for (epoch in seq_len(epochs)) {
        epoch_start <- proc.time()[["elapsed"]]
        batches$begin_epoch()
        samples <- 0L
        repeat {
            ids <- batches$`_next`()
            if (!length(ids)) break
            samples <- samples + length(ids)
            ids <- fit_rows[ids + 1L]
            x <- nn$take_rows(train_x, ids)
            y <- nn$take_rows(train$labels, ids)
            objective <- nn$sparse_softmax_cross_entropy(lenet5_logits(store, x, TRUE), y)
            if (l2 > 0) {
                penalty <- nn$sum(nn$square(weights[[1L]]))
                for (weight in weights[-1L]) penalty <- nn$add(penalty, nn$sum(nn$square(weight)))
                objective <- nn$add(objective, nn$mul(penalty, 0.5 * l2))
            }
            optimizer$minimize(objective)
        }
        nn_require(samples == length(fit_rows), "each selected training row used once per epoch")
        validation <- mnist_evaluate(store, lenet5_logits, train_x, train$labels, validation_rows)
        loss <- validation$loss$value()
        accuracy <- validation$accuracy$value()
        if (loss < best) {
            best <- loss
            best_epoch <- epoch
            store$save(checkpoint)
        }
        writeLines(sprintf("%d\t%s\t%s", epoch, g6(loss), g6(accuracy)), history)
        seconds <- proc.time()[["elapsed"]] - epoch_start
        cat(sprintf("  epoch %d: validation NLL=%s, accuracy=%s (%.3g s)\n", epoch, g6(loss), g6(accuracy), seconds))
    }
    train_seconds <- proc.time()[["elapsed"]] - started
    close(history)
    store$load(checkpoint)
    restored <- mnist_evaluate(store, lenet5_logits, train_x, train$labels, validation_rows)
    nn_near(restored$loss$value(), best, 1e-7, "best checkpoint matches selected validation loss")
    result <- mnist_evaluate(store, lenet5_logits, test_x, test$labels, test_rows)
    cat(sprintf("  selected epoch=%d; test NLL=%s; accuracy=%s\n", best_epoch, g6(result$loss$value()),
                g6(result$accuracy$value())))
    cells <- mnist_print_confusion(result$confusion, "test confusion (truth rows):")
    nn_require(sum(cells) == length(test_rows), "all selected test images evaluated")
    nn_near(result$accuracy$value(), sum(diag(cells)) / length(test_rows), 1e-7, "confusion matches accuracy")
    nn_training_quality(result$accuracy$value() >= minimum_test_accuracy,
                        "classifier did not reach demonstration accuracy threshold")
    cat(sprintf("  training time %.1f s (%.2f s/epoch)\n", train_seconds, train_seconds / epochs))
    cat(sprintf("  best checkpoint and history: %s\nAll %s checks passed.\n", normalizePath(model_dir), name))
    0L
}

if (sys.nframe() == 0L) quit(status = main())
