# Example dispatch is R code. Each sourced example owns its model and loops.
.nn_example_directory <- function() {
  root <- Sys.getenv("DATAMUNGE_EXAMPLES_ROOT", "")
  if (nzchar(root)) {
    directory <- file.path(root, "examples/r/nn")
    if (!dir.exists(directory)) stop("No R NN examples in DATAMUNGE_EXAMPLES_ROOT")
    return(directory)
  }
  root <- getwd()
  repeat {
    directory <- file.path(root, "examples/r/nn")
    if (file.exists(file.path(directory, "common.R"))) return(directory)
    parent <- dirname(root)
    if (identical(parent, root)) break
    root <- parent
  }
  system.file("examples/nn", package = "datamungerNN", mustWork = TRUE)
}
.nn_example_files <- function() {
  files <- list.files(.nn_example_directory(), "[.]R$", recursive = TRUE, full.names = TRUE)
  native <- vapply(files, function(file)
    startsWith(readLines(file, n = 1L, warn = FALSE), "# Native R NN example"), logical(1))
  files <- files[native]
  names(files) <- tools::file_path_sans_ext(basename(files))
  if (anyDuplicated(names(files))) stop("Duplicate R NN example names")
  files
}
nn_example_names <- function() names(.nn_example_files())
.nn_execute_example <- function(name, arguments) {
  file <- .nn_example_files()[[name]]
  environment <- new.env(parent = globalenv())
  sys.source(file, envir = environment)
  as.integer(environment$main(arguments))
}
nn_example_run <- function(name, arguments = character()) {
  if (length(name) != 1L || is.na(name) || !name %in% nn_example_names())
    stop("unknown native R NN example: ", paste(name, collapse = ", "),
         "; use nn_example_names() for R implementations (unported examples remain C++ references)")
  if (!is.character(arguments) || anyNA(arguments)) stop("arguments must be a character vector without missing values")
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
  .nn_execute_example(name, arguments[!arguments %in% c("--quick", "--full")])
}
