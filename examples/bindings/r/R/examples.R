# All experiments execute in this R process through SWIG.
nn_example_names <- function() .nn_catalog

nn_example_run <- function(name, arguments = character()) {
  if (length(name) != 1L || is.na(name) || !name %in% .nn_catalog)
    stop("unknown NN example: ", paste(name, collapse = ", "))
  if (!is.character(arguments) || anyNA(arguments))
    stop("arguments must be a character vector without missing values")
  keys <- c("DATAMUNGE_OUTPUT_DIR", "DATAMUNGE_EXAMPLES_FULL")
  previous <- Sys.getenv(keys, unset = NA_character_)
  on.exit({
    for (key in keys) {
      if (is.na(previous[[key]])) Sys.unsetenv(key)
      else do.call(Sys.setenv, setNames(list(previous[[key]]), key))
    }
  }, add = TRUE)
  full <- identical(Sys.getenv("DATAMUNGE_EXAMPLES_FULL"), "1")
  if ("--full" %in% arguments) full <- TRUE
  if ("--quick" %in% arguments) full <- FALSE
  Sys.setenv(DATAMUNGE_EXAMPLES_FULL = if (full) "1" else "0")
  if (!nzchar(Sys.getenv("DATAMUNGE_OUTPUT_DIR")))
    Sys.setenv(DATAMUNGE_OUTPUT_DIR = file.path("output/nn/r", if (full) "full" else "quick"))
  arguments <- arguments[!arguments %in% c("--full", "--quick")]
  if (name %in% .nn_native_catalog) {
    root <- Sys.getenv("DATAMUNGE_EXAMPLES_ROOT", getwd())
    while (!dir.exists(file.path(root,"examples/r/nn")) && dirname(root)!=root) root <- dirname(root)
    old_directory <- setwd(root)
    on.exit(setwd(old_directory),add=TRUE)
    files <- list.files(file.path(root,"examples/r/nn"), pattern=paste0("^",name,"[.]R$"),recursive=TRUE,full.names=TRUE)
    if (length(files)!=1L) stop("Run from the repository or set DATAMUNGE_EXAMPLES_ROOT")
    environment <- new.env(parent=globalenv())
    sys.source(files[[1L]],envir=environment)
    return(as.integer(environment$main(arguments)))
  }
  as.integer(run_nn_example(name, arguments))
}
