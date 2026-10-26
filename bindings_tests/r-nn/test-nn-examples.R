testthat::test_that("the R catalogue contains actual R implementations", {
  names <- nn_example_names()
  expected <- datamungerNN:::.nn_example_files()
  testthat::expect_setequal(names, names(expected))
  testthat::expect_true(all(c("cnn_mnist", "convolution_im2col", "undercomplete_autoencoder") %in% names))
  testthat::expect_false(exists("run_nn_example", asNamespace("datamungerNN"), inherits = FALSE))
  testthat::expect_error(nn_example_run("alexnet"), "unported examples remain C[+][+] references")
})

testthat::test_that("native numerical fixtures run and restore the environment", {
  keys <- c("DATAMUNGE_OUTPUT_DIR", "DATAMUNGE_EXAMPLES_FULL")
  before <- Sys.getenv(keys, unset = NA_character_)
  testthat::expect_identical(nn_example_run("convolution_im2col", "--full"), 0L)
  testthat::expect_identical(Sys.getenv(keys, unset = NA_character_), before)
  testthat::expect_error(nn_example_run("missing"), "unknown native R NN example")
  testthat::expect_error(nn_example_run("cnn_mnist", NA_character_), "missing")
})

testthat::test_that("mode flags and errors preserve the caller's environment", {
  keys <- c("DATAMUNGE_OUTPUT_DIR", "DATAMUNGE_EXAMPLES_FULL")
  before <- Sys.getenv(keys, unset = NA_character_)
  on.exit({
    for (key in keys) {
      if (is.na(before[[key]])) Sys.unsetenv(key)
      else do.call(Sys.setenv, setNames(list(before[[key]]), key))
    }
  })
  Sys.unsetenv(keys)
  testthat::local_mocked_bindings(
    .nn_execute_example = function(name, arguments) {
      testthat::expect_identical(name, "cnn_mnist")
      testthat::expect_identical(arguments, "epochs=1")
      testthat::expect_identical(Sys.getenv("DATAMUNGE_EXAMPLES_FULL"), "1")
      testthat::expect_identical(Sys.getenv("DATAMUNGE_OUTPUT_DIR"), "output/nn/r/full")
      stop("fixture failure")
    },
    .package = "datamungerNN"
  )
  testthat::expect_error(nn_example_run("cnn_mnist", c("--full", "epochs=1")), "fixture failure")
  testthat::expect_true(all(is.na(Sys.getenv(keys, unset = NA_character_))))
})

testthat::test_that("installed R examples run outside a checkout", {
  previous <- getwd()
  directory <- tempfile("r-examples-installed-")
  dir.create(directory)
  root <- Sys.getenv("DATAMUNGE_EXAMPLES_ROOT", unset = NA_character_)
  on.exit({
    setwd(previous)
    if (is.na(root)) Sys.unsetenv("DATAMUNGE_EXAMPLES_ROOT")
    else Sys.setenv(DATAMUNGE_EXAMPLES_ROOT = root)
  }, add = TRUE)
  Sys.unsetenv("DATAMUNGE_EXAMPLES_ROOT")
  setwd(directory)
  testthat::expect_true(all(c("cnn_mnist", "convolution_im2col", "undercomplete_autoencoder") %in% nn_example_names()))
  testthat::expect_identical(nn_example_run("convolution_im2col"), 0L)
})

testthat::test_that("the just R command graph invokes only R build and runtime tools", {
  if (!nzchar(Sys.which("just"))) testthat::skip("just not installed")
  root <- normalizePath(testthat::test_path("../.."))
  previous <- setwd(root)
  on.exit(setwd(previous))
  commands <- system2("just", c("--dry-run", "run-nn-r", "cnn_mnist", "epochs=2"), stdout=TRUE, stderr=TRUE)
  testthat::expect_null(attr(commands,"status"))
  testthat::expect_false(any(grepl("python|[.]py([[:space:]]|$)",commands,ignore.case=TRUE)))
  testthat::expect_true(any(grepl("R CMD INSTALL",commands,fixed=TRUE)))
  testthat::expect_true(any(grepl("Rscript --vanilla -e",commands,fixed=TRUE)))
})
