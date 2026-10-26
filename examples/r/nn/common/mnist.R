# MNIST evaluation helpers for the example-local classifier workflows
# (examples/nn/common/mnist.hpp). The official 60,000 training rows are split
# once with std::mt19937_64(2026) (nn$Mt19937_64 reproduces the shuffle).
# Metrics weight batch means by row count. Training loops stay in each example.

# 0..count-1 shuffled by std::shuffle with std::mt19937_64(seed).
mnist_development_order <- function(count = 60000L, seed = 2026L)
    as.numeric(nn$Mt19937_64(seed)$shuffle(seq_len(count) - 1L))

# mnist::Evaluate over 0-based `rows` in batches of 512. `logits_of(store, x, training)`
# builds [batch, 10] logits. Returns list(loss, accuracy, confusion) with
# WeightedMean metrics and a 10-class ConfusionMatrix.
mnist_evaluate <- function(store, logits_of, images, labels, rows) {
    result <- list(loss = nn$WeightedMean(), accuracy = nn$WeightedMean(), confusion = nn$ConfusionMatrix(10L))
    nn$no_grad(function() {
        for (start in seq.int(1L, length(rows), by = 512L)) {
            ids <- rows[start:min(start + 511L, length(rows))]
            x <- nn$take_rows(images, ids)
            y <- nn$take_rows(labels, ids)
            logits <- logits_of(store, x, FALSE)
            prediction <- nn$argmax(logits, 1L)
            result$loss$add(nn$sparse_softmax_cross_entropy(logits, y)$item(), length(ids))
            result$accuracy$add(nn$accuracy(prediction, y), length(ids))
            result$confusion$add(y, prediction)
        }
    })
    result
}

# ConfusionMatrix -> integer matrix (rows = truth, columns = prediction).
mnist_confusion_cells <- function(confusion) {
    classes <- confusion$classes()
    cells <- matrix(as.integer(confusion$to_tensor()$values()), nrow = classes, byrow = TRUE)
    dimnames(cells) <- list(truth = seq_len(classes) - 1L, prediction = seq_len(classes) - 1L)
    cells
}

# Prints the matrix in the C++ examples' layout (truth rows); returns the cells invisibly.
mnist_print_confusion <- function(confusion, title = "test confusion (truth rows):") {
    cells <- mnist_confusion_cells(confusion)
    cat("  ", title, "\n", sep = "")
    for (i in seq_len(nrow(cells)))
        cat("   ", paste0(" ", formatC(cells[i, ], width = 4L), collapse = ""), "\n", sep = "")
    invisible(cells)
}
