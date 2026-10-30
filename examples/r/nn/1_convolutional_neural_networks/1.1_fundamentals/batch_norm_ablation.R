# Native R NN example. Paired MNIST CNN batch-norm ablation trained and checked in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.1_fundamentals/batch_norm_ablation.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

# Paired MNIST CNN experiments with and without batch normalization.
#
# Port of tflow's batch_norm_ablation. Use the same 12,000 fitting rows,
# 3,000 validation rows, initial convolution and dense weights, six epochs,
# batch order, Adam(.001) and kernel L2(.0001). Three initialization seeds
# (11,23,47) produce paired runs. Batch norm adds gamma/beta after each
# convolution, before ReLU, with momentum .9 and eps .001. Keep convolution
# biases in both variants to isolate normalization.
# Select checkpoints on validation NLL; evaluate the official 10k test set
# after selection. Report paired differences and sample standard deviations;
# three runs on a small MNIST training subset do not establish a general
# advantage or disadvantage for batch normalization. Do not require BN to win.
#
# Check identical starting weights, one moving-statistics update per batch,
# inference invariance to batch companions, no statistics mutation during
# evaluation, and checkpoint restoration into a fresh model including state.
# The split (std::mt19937_64(2026), reproduced by nn$Mt19937_64) and batch
# orders (nn$BatchOrder) are identical to TF's; weights come from nn's
# generator per seed, so the paired numbers match TF only statistically.
# Batch norm is nn$batch_norm (moving statistics in a separate state store);
# checkpoints are .npz pairs seed<S>_{plain,bn}_{params,state}.npz.
# Learning curves, selected checkpoints and summary: output/nn/models/bn_ablation/.
# Preparation: python3 scripts/fetch_datasets.py mnist --split both (in tflow).
# Options: data=PATH epochs=6 threads=N (quick mode: 64 fit / 32 validation /
# 32 test rows, batch 16, one epoch unless epochs= is given).

nn_helper("checks")
nn_helper("mnist")

g6 <- function(value) sprintf("%.6g", value)

bn_options <- function() {
    o <- nn$BatchNormOptions()
    o$momentum <- 0.9
    o$eps <- 0.001
    o
}

# The paired CNN: conv(16)-[BN]-ReLU-pool, conv(32)-[BN]-ReLU-pool, Dense(64)-ReLU, 10 logits.
architecture <- function(use_bn, p, state, x, training) {
    h <- nn$reshape(x, c(-1L, 28L, 28L, 1L))
    for (block in 1:2) {
        width <- if (block == 1L) 16L else 32L
        h <- nn$conv2d(p, paste0("conv", block), h, width, 3L)
        if (use_bn) h <- nn$batch_norm(p, state, paste0("bn", block), h, training, bn_options())
        h <- nn$max_pool2d(nn$relu(h), 2L, 2L)
    }
    h <- nn$dense(p, "dense", nn$reshape(h, c(-1L, 7L * 7L * 32L)), 64L, nn$Activation$Relu)
    nn$dense(p, "output", h, 10L)
}

# Batch mean/variance (unbiased, as TF FusedBatchNorm reports) that a training
# step on `x` feeds into each moving average, from the current weights.
moments <- function(p, x) nn$no_grad(function() {
    result <- list()
    h <- nn$reshape(x, c(-1L, 28L, 28L, 1L))
    for (block in 1:2) {
        h <- nn$conv2d(p, paste0("conv", block), h, if (block == 1L) 16L else 32L, 3L)
        r <- nn$batch_norm_train(h, p$at(paste0("bn", block, "/gamma")), p$at(paste0("bn", block, "/beta")),
                                 bn_options()$eps)
        result <- c(result, list(r$mean$clone(), r$variance$clone()))
        h <- nn$max_pool2d(nn$relu(r$y), 2L, 2L)
    }
    result
})

# Parameters, batch-norm state, Adam and the L2-regularized kernels of one variant.
new_model <- function(use_bn, seed) {
    model <- new.env()
    model$use_bn <- use_bn
    model$params <- nn$ParameterStore(seed)
    model$state <- nn$ParameterStore(0L)
    nn$no_grad(function() architecture(use_bn, model$params, model$state, nn$zeros(c(1L, 784L)), FALSE))
    config <- nn$OptimizerConfig()
    config$kind <- nn$OptimizerKind$Adam
    model$optimizer <- nn$Optimizer(model$params, 0.001, config)
    model$weights <- Filter(function(w) w$dim() >= 2L, nn$tensor_list(model$params$parameters()))
    model$builder <- function(p, x, training) architecture(model$use_bn, p, model$state, x, training)
    model
}

model_probabilities <- function(model, x)
    nn$no_grad(function() nn$softmax(architecture(model$use_bn, model$params, model$state, x, FALSE)))

model_save <- function(model, prefix) {
    model$params$save(paste0(prefix, "_params.npz"))
    if (model$use_bn) model$state$save(paste0(prefix, "_state.npz"))
}

model_load <- function(model, prefix) {
    model$params$load(paste0(prefix, "_params.npz"))
    if (model$use_bn) model$state$load(paste0(prefix, "_state.npz"))
}

snapshot <- function(store) lapply(nn$tensor_list(store$parameters()), function(t) t$detach()$clone())

same <- function(a, b, tolerance, message) {
    nn_require(identical(as.numeric(a$shape()), as.numeric(b$shape())), paste(message, "shape"))
    x <- a$contiguous()$values()
    y <- b$contiguous()$values()
    if (tolerance == 0)
        nn_require(identical(x, y), message)
    else
        nn_require(all(is.finite(x) & is.finite(y) & abs(x - y) <= tolerance), message)
}

describe_values <- function(values) c(mean = mean(values), sd = sd(values))

main <- function(arguments = commandArgs(TRUE)) {
    args <- nn_options(arguments, list(data = nn_dataset("mnist"), epochs = 0L, threads = 0L,
                                       full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    Sys.setenv(DATAMUNGE_EXAMPLES_FULL = if (isTRUE(args$full)) "1" else "0")
    if (args$threads > 0L) nn$set_num_threads(args$threads)
    nn_require(args$epochs >= 0L, "positive epochs required")
    epochs <- if (args$epochs > 0L) args$epochs else nn_limit(6L, 1L)
    training <- nn_components$dataset_assets$LoadMnist(args$data, "train")
    test <- nn_components$dataset_assets$LoadMnist(args$data, "test")
    nn_require(as.numeric(training$images$shape())[1L] == 60000 && as.numeric(test$images$shape())[1L] == 10000,
               "official MNIST row counts")
    train_x <- nn$reshape(training$images, c(60000L, 784L))
    test_x <- nn$reshape(test$images, c(10000L, 784L))
    test_rows <- seq_len(nn_limit(10000L, 32L)) - 1L
    order <- mnist_development_order(60000L, 2026L)
    fit <- order[1:12000]
    validation <- order[12001:15000]
    fit <- head(fit, nn_limit(length(fit), 64L))
    validation <- head(validation, nn_limit(length(validation), 32L))
    nn_describe()
    directory <- nn_output_dir("models/bn_ablation")
    curves <- file(file.path(directory, "history.tsv"), "w")
    summary_file <- file(file.path(directory, "summary.tsv"), "w")
    writeLines("seed\tbatch_norm\tepoch\tvalidation_nll\tvalidation_accuracy\ttrain_seconds", curves)
    writeLines("seed\tbatch_norm\tselected_epoch\ttest_nll\ttest_accuracy\ttrain_seconds", summary_file)
    accuracies <- list(numeric(), numeric())
    started <- proc.time()[["elapsed"]]
    for (seed in c(11L, 23L, 47L)) {
        models <- list(new_model(FALSE, seed), new_model(TRUE, seed))
        for (name in models[[1L]]$params$names())
            same(models[[1L]]$params$at(name), models[[2L]]$params$at(name), 0, "identical paired initialization")
        cat(sprintf("== seed %d: parameters %.0f / %.0f; identical starting weights\n", seed,
                    models[[1L]]$params$count(), models[[2L]]$params$count()))
        best <- c(Inf, Inf)
        seconds <- c(0, 0)
        selected <- c(0L, 0L)
        checkpoints <- file.path(directory, paste0("seed", seed, c("_plain", "_bn")))
        batches <- nn$BatchOrder(length(fit), nn_limit(128L, 16L), seed + 1000L)
        first_update <- TRUE
        for (epoch in seq_len(epochs)) {
            batches$begin_epoch()
            samples <- 0L
            repeat {
                rows <- batches$`_next`()
                if (!length(rows)) break
                samples <- samples + length(rows)
                rows <- fit[rows + 1L]
                x <- nn$take_rows(train_x, rows)
                y <- nn$take_rows(training$labels, rows)
                if (first_update) {
                    old_state <- snapshot(models[[2L]]$state)
                    expected_moments <- moments(models[[2L]]$params, x)
                }
                for (variant in 1:2) {
                    # One explicit training step: forward, mean cross-entropy plus
                    # 0.5 * l2 * sum(W^2) over the kernels, then the Adam update.
                    start <- proc.time()[["elapsed"]]
                    model <- models[[variant]]
                    objective <- nn$sparse_softmax_cross_entropy(
                        architecture(model$use_bn, model$params, model$state, x, TRUE), y)
                    penalty <- nn$sum(nn$square(model$weights[[1L]]))
                    for (weight in model$weights[-1L]) penalty <- nn$add(penalty, nn$sum(nn$square(weight)))
                    model$optimizer$minimize(nn$add(objective, nn$mul(penalty, 0.5 * 1e-4)))
                    seconds[variant] <- seconds[variant] + proc.time()[["elapsed"]] - start
                }
                if (first_update) {
                    updated <- snapshot(models[[2L]]$state)
                    nn_require(length(updated) == length(expected_moments), "one mean/variance pair per batch norm")
                    for (i in seq_along(updated)) {
                        u <- updated[[i]]$values()
                        o <- old_state[[i]]$values()
                        m <- expected_moments[[i]]$values()
                        nn_require(all(abs(u - (0.9 * o + 0.1 * m)) <= 2e-6),
                                   "exactly one moving-statistics update from pre-update activations")
                    }
                    first_update <- FALSE
                }
            }
            nn_require(samples == length(fit), "all fitting rows used once per epoch")
            for (variant in 1:2) {
                model <- models[[variant]]
                metrics <- mnist_evaluate(model$params, model$builder, train_x, training$labels, validation)
                nll <- metrics$loss$value()
                accuracy <- metrics$accuracy$value()
                if (nll < best[variant]) {
                    best[variant] <- nll
                    selected[variant] <- epoch
                    model_save(model, checkpoints[variant])
                }
                writeLines(sprintf("%d\t%d\t%d\t%s\t%s\t%s", seed, variant - 1L, epoch, g6(nll), g6(accuracy),
                                   g6(seconds[variant])), curves)
                flush(curves)
                cat(sprintf("  epoch %d %s: validation NLL=%s, accuracy=%s\n", epoch,
                            if (variant == 2L) "BN" else "plain", g6(nll), g6(accuracy)))
            }
        }
        for (variant in 1:2) {
            model <- models[[variant]]
            label <- if (variant == 2L) "BN" else "plain"
            model_load(model, checkpoints[variant])
            validation_result <- mnist_evaluate(model$params, model$builder, train_x, training$labels, validation)
            nn_near(validation_result$loss$value(), best[variant], 1e-7, "restored validation loss")
            result <- mnist_evaluate(model$params, model$builder, test_x, test$labels, test_rows)
            nn_training_quality(result$accuracy$value() >= 0.95, "both models learn the classification task")
            accuracies[[variant]] <- c(accuracies[[variant]], result$accuracy$value())
            writeLines(sprintf("%d\t%d\t%d\t%s\t%s\t%s", seed, variant - 1L, selected[variant],
                               g6(result$loss$value()), g6(result$accuracy$value()), g6(seconds[variant])),
                       summary_file)
            flush(summary_file)
            cat(sprintf("  selected %s epoch=%d; test NLL=%s, accuracy=%s\n", label, selected[variant],
                        g6(result$loss$value()), g6(result$accuracy$value())))
            cells <- mnist_print_confusion(result$confusion, paste(label, "test confusion (truth rows):"))
            nn_require(sum(cells) == length(test_rows), "all selected test images evaluated")
            nn_near(result$accuracy$value(), sum(diag(cells)) / sum(cells), 1e-7, "confusion matches accuracy")
        }
        bn <- models[[2L]]
        before_evaluation <- snapshot(bn$state)
        batch_x <- nn$take_rows(test_x, 0:7)
        single_x <- nn$take_rows(test_x, 0L)
        batch_result <- model_probabilities(bn, batch_x)
        single_result <- model_probabilities(bn, single_x)
        for (c in 0:9)
            nn_near(batch_result$at(c(0L, c)), single_result$at(c(0L, c)), 1e-6,
                    "inference predictions do not depend on batch companions")
        after_evaluation <- snapshot(bn$state)
        for (i in seq_along(before_evaluation))
            same(before_evaluation[[i]], after_evaluation[[i]], 0, "evaluation leaves moments unchanged")
        fresh <- new_model(TRUE, 999L)
        model_load(fresh, checkpoints[2L])
        same(model_probabilities(fresh, batch_x), batch_result, 1e-7, "fresh model checkpoint predictions")
        fresh_state <- snapshot(fresh$state)
        for (i in seq_along(fresh_state))
            same(fresh_state[[i]], before_evaluation[[i]], 0, "checkpoint contains moving moments")
    }
    for (variant in 1:2) {
        stats <- describe_values(accuracies[[variant]])
        cat(sprintf("  %s test accuracy mean=%s; sample SD=%s\n", if (variant == 2L) "BN" else "plain",
                    g6(stats[["mean"]]), g6(stats[["sd"]])))
    }
    stats <- describe_values(accuracies[[2L]] - accuracies[[1L]])
    cat(sprintf("  paired BN-plain accuracy difference mean=%s; sample SD=%s\n", g6(stats[["mean"]]), g6(stats[["sd"]])))
    cat("  six epochs on 12000 fitting rows; descriptive results from three paired seeds\n")
    close(curves)
    close(summary_file)
    cat(sprintf("  total time %.1f s\n", proc.time()[["elapsed"]] - started))
    cat("All batch-normalization ablation checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
