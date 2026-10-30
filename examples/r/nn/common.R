# R data, evaluation and training helpers; numerical ops use the SWIG NN API.
suppressPackageStartupMessages(library(datamungerNN))

nn_data_root <- function() nn_components$dataset_assets$DataRoot()

nn_output_root <- function(full = FALSE) Sys.getenv("DATAMUNGE_OUTPUT_DIR", file.path("output/nn/r",
    if (full) "full" else "quick"))

nn_options <- function(arguments, defaults) {
    result <- defaults
    for (argument in arguments) {
        if (argument %in% c("--quick", "--full")) {
            result$full <- argument == "--full"
            next
        }
        equal <- regexpr("=", argument, fixed = TRUE)[1L]
        key <- if (equal > 0L) substr(argument, 1L, equal - 1L) else argument
        if (equal < 1L || !key %in% names(result))
            stop("unknown option: ", argument)
        value <- substring(argument, equal + 1L)
        result[[key]] <- if (is.integer(result[[key]]))
            as.integer(value)
        else if (is.numeric(result[[key]]))
            as.numeric(value)
        else value
    }
    result
}

nn_arrays <- function(parameters) setNames(lapply(parameters$names(), function(name) parameters$at(name)$detach()$clone()),
    parameters$names())


# Source examples/r/nn/common/<name>.R (shared model pieces and data readers,
# mirroring examples/nn/common/*.hpp) into the calling script's environment.
# Helpers never own an end-to-end experiment: loops stay in each example.
nn_helper <- function(name, envir = parent.frame()) {
    file <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT", getwd()), "examples/r/nn/common", paste0(name, ".R"))
    if (!file.exists(file)) file <- system.file("examples/nn/common", paste0(name, ".R"), package = "datamungerNN", mustWork = TRUE)
    sys.source(file, envir = envir)
    invisible(envir)
}
