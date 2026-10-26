# Runtime checks, quick/full mode and output paths (examples/nn/common/{checks,runtime,paths}.hpp).
nn_full <- function() identical(Sys.getenv("DATAMUNGE_EXAMPLES_FULL"), "1")
nn_limit <- function(requested, quick_limit) if (nn_full()) requested else min(requested, quick_limit)
nn_describe <- function() if (!nn_full())
    cat("Quick demonstration: bounded training; convergence metrics require DATAMUNGE_EXAMPLES_FULL=1.\n")
nn_require <- function(condition, message) if (!isTRUE(condition)) stop(message, call. = FALSE)
nn_training_quality <- function(condition, message) {
    if (nn_full()) nn_require(condition, message)
    else cat("  full-training quality check deferred:", message, "\n")
}
nn_near <- function(actual, expected, tolerance, message) {
    if (!is.finite(actual) || !is.finite(expected) || abs(actual - expected) > tolerance)
        stop(sprintf("%s: got %.17g, expected %.17g", message, actual, expected), call. = FALSE)
}
nn_throws <- function(code, message) {
    result <- tryCatch({ code; NULL }, error = function(e) conditionMessage(e))
    if (is.null(result)) stop(message, ": expected an exception", call. = FALSE)
    result
}
nn_header <- function(title) cat("\n== ", title, " ", strrep("=", max(0L, 60L - nchar(title))), "\n", sep = "")
nn_dataset <- function(name) file.path(nn_data_root(), name)
nn_output_dir <- function(relative) {
    directory <- file.path(nn_output_root(nn_full()), relative)
    dir.create(directory, recursive = TRUE, showWarnings = FALSE)
    directory
}
nn_tf_models <- function(name) file.path(dirname(nn_data_root()), "models", name)
