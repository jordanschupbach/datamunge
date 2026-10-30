# Native R NN example. Model composition and training run in R.
# CMD: nix develop .#r --command Rscript examples/r/nn/1_convolutional_neural_networks/1.1_fundamentals/cnn_mnist.R
# Epochs are honored in both modes; --quick reduces the dataset, --full uses all MNIST.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT", getwd()), "examples/r/nn/common.R")
if (!file.exists(.nn_common))
    .nn_common <- system.file("examples/nn/common.R", package = "datamungerNN", mustWork = TRUE)
source(.nn_common, local = TRUE)
rm(.nn_common)

cnn_mnist_logits <- function(parameters, inputs, training = FALSE) {
    options <- nn$ConvLayerOptions()
    options$activation <- nn$Activation$Relu
    image <- nn$reshape(inputs, c(-1L, 28L, 28L, 1L))
    conv1 <- nn$conv2d(parameters, "conv1", image, 16L, 3L, options)
    pool1 <- nn$max_pool2d(conv1, 2L, 2L)
    conv2 <- nn$conv2d(parameters, "conv2", pool1, 32L, 3L, options)
    pool2 <- nn$max_pool2d(conv2, 2L, 2L)
    hidden <- nn$dense(parameters, "dense", nn$reshape(pool2, c(-1L, 7L * 7L * 32L)), 64L, nn$Activation$Relu)
    nn$dense(parameters, "output", hidden, 10L)
}

cnn_mnist_evaluate <- function(parameters, images, labels, rows) {
    nn$no_grad(function() {
        total <- correct <- count <- 0
        confusion <- nn$ConfusionMatrix(10L)
        for (start in seq.int(1L, length(rows), by = 512L)) {
            ids <- rows[start:min(start + 511L, length(rows))]
            targets <- nn$take_rows(labels, ids)
            logits <- cnn_mnist_logits(parameters, nn$take_rows(images, ids), FALSE)
            total <- total + length(ids) * nn$sparse_softmax_cross_entropy(logits, targets)$item()
            confusion$add(targets, nn$argmax(logits, 1L))
            correct <- correct + sum(nn$argmax(logits, 1L)$values() == targets$values())
            count <- count + length(ids)
        }
        structure(c(loss = total/count, accuracy = correct/count),
            confusion = matrix(confusion$to_tensor()$values(), nrow=10L, byrow=TRUE))
    })
}

main <- function(arguments = commandArgs(TRUE)) {
    name <- "cnn_mnist"
    epochs <- 100L
    learning_rate <- 0.001
    minimum_accuracy <- 0.98
    args <- nn_options(arguments, list(data = file.path(nn_data_root(), "mnist"), epochs = as.numeric(epochs), batch = 256,
        lr = learning_rate, threads = 2, full = Sys.getenv("DATAMUNGE_EXAMPLES_FULL") == "1"))
    if (length(args$epochs) != 1L || !is.finite(args$epochs) || args$epochs < 1L ||
        args$epochs != floor(args$epochs))
        stop("epochs must be a positive whole number")
    if (length(args$batch) != 1L || !is.finite(args$batch) || args$batch < 1L ||
        args$batch != floor(args$batch))
        stop("batch must be a positive whole number")
    if (!is.finite(args$lr) || args$lr <= 0)
        stop("lr must be positive and finite")
    if (!is.finite(args$threads) || args$threads < 1L || args$threads != floor(args$threads))
        stop("threads must be a positive integer")
    nn$set_num_threads(args$threads)
    train <- nn_components$dataset_assets$LoadMnist(args$data, "train")
    test <- nn_components$dataset_assets$LoadMnist(args$data, "test")
    stopifnot(identical(as.numeric(train$images$shape()), c(60000, 28, 28, 1)), identical(as.numeric(test$images$shape()),
        c(10000, 28, 28, 1)))
    train_x <- nn$reshape(train$images, c(60000L, 784L))
    test_x <- nn$reshape(test$images, c(10000L, 784L))
    split <- nn$BatchOrder(60000L, 60000L, 2026L)
    split$begin_epoch()
    ordering <- split$`_next`()
    fit <- ordering[seq_len(55000L)]
    validation <- ordering[55001L:60000L]
    testing <- 0L:9999L
    if (!args$full) {
        fit <- head(fit, 64L)
        validation <- head(validation, 32L)
        testing <- head(testing, 32L)
        args$batch <- min(args$batch, 16L)
    }
    cat(sprintf("%s mode: %d epochs; %d fit / %d validation / %d test images; batch=%d\n",
        if (args$full) "Full MNIST" else "Quick dataset", args$epochs, length(fit),
        length(validation), length(testing), args$batch))
    if (!args$full)
        cat("Quick mode limits data and batch size only. Use --full for all MNIST images.\n")
    parameters <- nn$ParameterStore(17L)
    nn$no_grad(function() cnn_mnist_logits(parameters, nn$slice_axis(train_x, 0L, 0L, 1L), TRUE))
    config <- nn$OptimizerConfig()
    config$kind <- nn$OptimizerKind$Adam
    optimizer <- nn$Optimizer(parameters, args$lr, config)
    weights <- Filter(function(p) p$dim() >= 2L, nn$tensor_list(parameters$parameters()))
    directory <- file.path(nn_output_root(args$full), "models", name)
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    history <- file(file.path(directory, "history.tsv"), "w")
    on.exit(close(history), add = TRUE)
    writeLines("epoch\tvalidation_loss\tvalidation_accuracy", history)
    batches <- nn$BatchOrder(length(fit), args$batch, 29L)
    best <- Inf
    selected <- 0L
    for (epoch in seq_len(args$epochs)) {
        batches$begin_epoch()
        seen <- 0L
        repeat {
            ids <- batches$`_next`()
            if (!length(ids))
                break
            rows <- fit[ids + 1L]
            loss <- nn$sparse_softmax_cross_entropy(cnn_mnist_logits(parameters, nn$take_rows(train_x,
                rows), TRUE), nn$take_rows(train$labels, rows))
            penalty <- nn$sum(nn$square(weights[[1L]]))
            if (length(weights) > 1L)
                for (weight in weights[-1L]) penalty <- nn$add(penalty, nn$sum(nn$square(weight)))
            optimizer$minimize(nn$add(loss, nn$mul(penalty, 5e-05)))
            seen <- seen + length(rows)
        }
        stopifnot(seen == length(fit))
        metric <- cnn_mnist_evaluate(parameters, train_x, train$labels, validation)
        writeLines(sprintf("%d\t%.17g\t%.17g", epoch, metric[1L], metric[2L]), history)
        flush(history)
        if (metric[1L] < best) {
            best <- metric[1L]
            selected <- epoch
            parameters$save(file.path(directory, "best.npz"))
        }
        cat(sprintf("epoch%d/%d: validation NLL=%.7g,accuracy=%.7g\n", epoch, args$epochs, metric[1L], metric[2L]))
    }
    parameters$load(file.path(directory, "best.npz"))
    metric <- cnn_mnist_evaluate(parameters, test_x, test$labels, testing)
    if (args$full)
        stopifnot(metric[2L] >= minimum_accuracy)
    cat(sprintf("%s: test NLL=%.7g,accuracy=%.7g;%d parameters\n", name, metric[1L], metric[2L], parameters$count()))
    confusion <- attr(metric, "confusion")
    stopifnot(sum(confusion) == length(testing))
    dimnames(confusion) <- list(truth=0L:9L, prediction=0L:9L)
    write.table(confusion, file.path(directory,"confusion.tsv"), sep="\t", quote=FALSE, col.names=NA)
    cat("Confusion matrix (rows=truth, columns=prediction):\n")
    print(confusion)
    0L
}

if (sys.nframe() == 0L) quit(status = main())

