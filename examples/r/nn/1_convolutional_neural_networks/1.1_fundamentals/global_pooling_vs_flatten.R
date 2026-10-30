# Native R NN example. Global average pooling vs flatten heads, written out with datamunge::nn ops.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.1_fundamentals/global_pooling_vs_flatten.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)
nn_helper("checks")

# Global average pooling vs flatten: invariance, location and learned heads.
#
# Port of tflow's global_pooling_vs_flatten. For [N,H,W,C] feature maps and K
# classes, a GAP-linear head learns C*K+K parameters while flatten-linear learns
# H*W*C*K+K. GAP discards spatial arrangement: every permutation of a map's
# spatial sites has the same logits. Verify uniform spatial input gradients and
# the class-activation-map identity logit_k = mean_hw(sum_c(feature_hwc * W_ck)) + bias_k.
# Two controlled, head-only learning tasks use 8x8 two-channel feature maps.
# Appearance: the active channel is the label; fit half the spatial sites and
# test the other half. Location: the label is left/right and the active channel
# is fixed; fit all sites and test a new positive amplitude. Both heads use
# Adam(.05), the same examples and 300 full-batch steps of mean sparse
# cross-entropy. Each head's 2x2 test confusion matrix (truth rows) is printed.

g6 <- function(x) sprintf("%.6g", x)

# float32 rounding through a tensor round trip.
as_float32 <- function(x) nn$from_vector(x, length(x))$values()

algebra_checks <- function() {
    weights <- nn$from_vector(c(.4, -.7, .2, .3, -.5, .6), c(3L, 2L))
    bias <- nn$from_vector(c(.1, -.2), 2L)
    # std::sin(int(i) * .17f): the product is a float32.
    index <- 0:(2 * 4 * 5 * 3 - 1)
    values <- sin(as_float32(index * as_float32(.17)))
    x <- nn$from_vector(values, c(2L, 4L, 5L, 3L))
    x$requires_grad_()
    logits <- nn$bias_add(nn$matmul(nn$global_avg_pool2d(x), weights), bias)
    maps <- nn$reshape(nn$matmul(nn$reshape(x, c(-1L, 3L)), weights), c(2L, 4L, 5L, 2L))
    from_maps <- nn$bias_add(nn$global_avg_pool2d(maps), bias)
    # Reversing both spatial axes is a nontrivial permutation that keeps channel identity.
    permuted <- nn$bias_add(nn$matmul(nn$global_avg_pool2d(nn$reverse(x, c(1L, 2L))), weights), bias)
    objective <- nn$sum(nn$slice_axis(logits, 1L, 0L, 1L))
    gradient <- nn$grad(objective, list(x))[[1L]]$values()
    a <- logits$values()
    b <- from_maps$values()
    c3 <- permuted$values()
    for (i in 1:4) {
        nn_near(a[i], b[i], 1e-7, "class-activation-map identity")
        nn_near(a[i], c3[i], 1e-7, "spatial permutation invariance")
    }
    w <- weights$values()
    for (i in seq_along(gradient))
        nn_near(gradient[i], w[((i - 1) %% 3) * 2 + 1] / 20, 1e-8, "uniform spatial GAP gradient")
    cat("  algebra: spatial permutation, class-activation maps and uniform input derivatives passed\n")
}

head_logits <- function(parameters, inputs, global_average) {
    width <- if (global_average) 2L else 128L
    features <- if (global_average) nn$global_avg_pool2d(nn$reshape(inputs, c(-1L, 8L, 8L, 2L))) else inputs
    weights <- parameters$get("head/W", c(width, 2L), nn$init$zeros())
    bias <- parameters$get("head/b", 2L, nn$init$zeros())
    nn$bias_add(nn$matmul(features, weights), bias)
}

# An impulse of amplitude 64 gives one unit in the global average. Test location
# maps use amplitude 80; appearance uses exactly unseen sites.
make_dataset <- function(location, test) {
    examples <- list()
    for (y in 0:7) for (x in 0:7) {
        if (location) {
            examples[[length(examples) + 1L]] <- c(y, x, 0, as.numeric(x >= 4))
        } else if ((y + x) %% 2 == as.integer(test)) {
            for (label in 0:1) examples[[length(examples) + 1L]] <- c(y, x, label, label)
        }
    }
    n <- length(examples)
    features <- numeric(n * 128)
    labels <- numeric(n)
    for (i in seq_len(n)) {
        e <- examples[[i]]
        features[(i - 1) * 128 + (e[1] * 8 + e[2]) * 2 + e[3] + 1] <- if (location && test) 80 else 64
        labels[i] <- e[4]
    }
    list(x = nn$from_vector(features, c(n, 128L)), y = nn$from_vector_int64(labels, n))
}

learn_heads <- function() {
    for (location in c(FALSE, TRUE)) {
        train <- make_dataset(location, FALSE)
        test_set <- make_dataset(location, TRUE)
        # Explicitly prove that location labels have identical GAP features.
        if (location) {
            values <- matrix(train$x$values(), ncol = 128L, byrow = TRUE)
            for (row in seq_len(nrow(values))) {
                nn_near(sum(values[row, seq(1L, 128L, by = 2L)]), 64, 0, "location examples share GAP feature zero")
                nn_near(sum(values[row, seq(2L, 128L, by = 2L)]), 0, 0, "location examples share GAP feature one")
            }
        }
        for (global_average in c(FALSE, TRUE)) {
            model <- nn$ParameterStore(13L)
            nn$no_grad(function() head_logits(model, train$x, global_average))
            nn_require(model$count() == if (global_average) 6 else 258, "GAP vs flatten head parameter count")
            config <- nn$OptimizerConfig()
            config$kind <- nn$OptimizerKind$Adam
            optimizer <- nn$Optimizer(model, .05, config)
            for (step in seq_len(300L)) {
                loss <- nn$sparse_softmax_cross_entropy(head_logits(model, train$x, global_average), train$y)
                optimizer$minimize(loss)
            }
            nn$no_grad(function() {
                evaluate <- function(x, y) {
                    logits <- head_logits(model, x, global_average)
                    c(nll = nn$sparse_softmax_cross_entropy(logits, y)$item(), accuracy = nn$accuracy(logits, y))
                }
                training <- evaluate(train$x, train$y)
                test <- evaluate(test_set$x, test_set$y)
                expected <- if (location) (if (global_average) .5 else 1) else (if (global_average) 1 else .5)
                nn_near(test[["accuracy"]], expected, 0, "controlled head generalization result")
                if (global_average && location)
                    nn_near(test[["nll"]], log(2), 1e-6, "indistinguishable balanced classes have log(2) loss")
                else
                    nn_near(training[["accuracy"]], 1, 0, "learnable training task fitted")
                if (!global_average && !location) {
                    weights <- model$at("head/W")$values()
                    for (y in 0:7) for (x in 0:7) if ((y + x) %% 2 == 1)
                        for (channel in 0:1) for (label in 0:1)
                            nn_near(weights[((y * 8 + x) * 2 + channel) * 2 + label + 1], 0, 0,
                                    "unseen flatten coordinates receive no gradient")
                }
                cat(sprintf("  %s %s: parameters=%d; training accuracy=%s; test accuracy=%s; test NLL=%s\n",
                            if (location) "location" else "appearance", if (global_average) "GAP" else "flatten",
                            as.integer(model$count()), g6(training[["accuracy"]]), g6(test[["accuracy"]]),
                            g6(test[["nll"]])))
                confusion <- nn$ConfusionMatrix(2L)
                confusion$add(test_set$y, nn$argmax(head_logits(model, test_set$x, global_average), 1L))
                cells <- as.integer(confusion$to_tensor()$values())
                nn_near(confusion$accuracy(), test[["accuracy"]], 1e-12, "confusion matches test accuracy")
                cat(sprintf("    test confusion (truth rows): [%d %d; %d %d]\n", cells[1], cells[2], cells[3], cells[4]))
            })
        }
    }
}

main <- function(arguments = commandArgs(TRUE)) {
    nn_options(arguments, list(full = nn_full()))
    algebra_checks()
    learn_heads()
    cat("All global-pooling vs flatten checks passed.\n")
    0L
}

if (sys.nframe() == 0L) quit(status = main())
