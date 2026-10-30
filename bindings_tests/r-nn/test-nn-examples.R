testthat::test_that("the R catalogue contains actual R implementations", {
  names <- nn_example_names()
  expected <- datamungerNN:::.nn_example_files()
  testthat::expect_setequal(names, names(expected))
  testthat::expect_true(all(c("cnn_mnist", "convolution_im2col", "undercomplete_autoencoder") %in% names))
  testthat::expect_false(exists("run_nn_example", asNamespace("datamungerNN"), inherits = FALSE))
  testthat::expect_true(all(c("alexnet", "xception", "densenet", "efficientnet_v2") %in% names))
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

testthat::test_that("shared example components are bound and grouped by C++ namespace", {
  for (name in c("check", "example_runtime", "paths", "image_training", "resnet", "resnet_demo", "attention_cnn", "densenet"))
    testthat::expect_true(is.environment(nn_components[[name]]), info = name)
  cfg <- nn_components$image_training$ImageTrainingConfig()
  testthat::expect_identical(as.integer(c(cfg$classes, cfg$epochs, cfg$batch)), c(10L, 10L, 128L))
  cfg$epochs <- 3L
  testthat::expect_identical(as.integer(cfg$epochs), 3L)
  testthat::expect_error(nn_components$check$Require(FALSE, "broken"), "broken")
  previous_full <- Sys.getenv("DATAMUNGE_EXAMPLES_FULL", unset = NA_character_)
  Sys.setenv(DATAMUNGE_EXAMPLES_FULL = "0")
  testthat::expect_false(nn_components$example_runtime$Full())
  testthat::expect_identical(as.integer(nn_components$example_runtime$Limit(12L, 1L)), 1L)
  if (is.na(previous_full)) Sys.unsetenv("DATAMUNGE_EXAMPLES_FULL") else Sys.setenv(DATAMUNGE_EXAMPLES_FULL = previous_full)
  fixture <- nn$ImageDataset()
  fixture$images <- nn$ones(c(2L, 32L, 32L, 3L))
  fixture$labels <- nn$from_vector_int64(c(2, 7), 2L)
  batch <- nn_components$image_training$ImageBatch(fixture, c(1L, 0L))
  testthat::expect_identical(as.numeric(batch$second$values()), c(7, 2))
  testthat::expect_true(all(batch$first$values() == 1))
  model <- nn_components$resnet_demo$CompactResNet(18L)
  params <- nn$ParameterStore(17L)
  state <- nn$ParameterStore(17L)
  logits <- nn$no_grad(function() model$call(params, state, nn$zeros(c(2L, 32L, 32L, 3L)), FALSE))
  testthat::expect_identical(as.numeric(logits$shape()), c(2, 10))
  testthat::expect_identical(as.numeric(params$count()), 701466)
  testthat::expect_true("resnet_basic" %in% nn_example_names())
})
