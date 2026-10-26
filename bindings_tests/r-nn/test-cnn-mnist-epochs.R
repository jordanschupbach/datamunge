testthat::test_that("MNIST honors edited defaults and explicit epochs in quick mode", {
  root <- normalizePath(testthat::test_path("../.."))
  previous_directory <- setwd(root)
  on.exit(setwd(previous_directory), add = TRUE)
  example <- new.env()
  source("examples/r/nn/1_convolutional_neural_networks/1.1_fundamentals/cnn_mnist.R", local = example)

  # Use small synthetic tensors while exercising the actual batching, optimizer,
  # checkpoint and history code. Dataset readers retain their official metadata.
  core <- datamungerNN::nn
  api <- list2env(as.list(core), parent = emptyenv())
  example$nn <- api
  api$load_mnist <- function(directory, split) {
    count <- if (split == "train") 60000L else 10000L
    rows <- if (split == "train") 64L else 32L
    list(images = list(shape = function() c(count, 28, 28, 1), rows = rows),
         labels = core$zeros(rows, core$DType$Int64))
  }
  api$reshape <- function(tensor, shape) {
    if (is.list(tensor) && !is.null(tensor$rows)) return(core$zeros(c(tensor$rows, 784L)))
    core$reshape(tensor, shape)
  }
  api$BatchOrder <- function(samples, batch, seed) {
    if (samples == 60000L)
      return(list(begin_epoch = function() invisible(NULL),
                  `_next` = function() rep(0L:63L, length.out = 60000L)))
    core$BatchOrder(samples, batch, seed)
  }
  example$cnn_mnist_logits <- function(parameters, inputs, training = FALSE)
    core$dense(parameters, "output", inputs, 10L)

  # Simulate editing epochs in the script, without passing epochs= on the CLI.
  expressions <- as.list(body(example$main))
  assignment <- which(vapply(expressions, function(expr)
    is.call(expr) && identical(expr[[1L]], as.name("<-")) &&
      identical(expr[[2L]], as.name("epochs")), logical(1)))
  expressions[[assignment]][[3L]] <- 3L
  body(example$main) <- as.call(expressions)

  output <- tempfile("cnn-mnist-epochs-")
  previous_output <- Sys.getenv("DATAMUNGE_OUTPUT_DIR", unset = NA_character_)
  Sys.setenv(DATAMUNGE_OUTPUT_DIR = output)
  on.exit({
    if (is.na(previous_output)) Sys.unsetenv("DATAMUNGE_OUTPUT_DIR")
    else Sys.setenv(DATAMUNGE_OUTPUT_DIR = previous_output)
  }, add = TRUE)
  printed <- capture.output(testthat::expect_identical(example$main("--quick"), 0L))
  history <- read.delim(file.path(output, "models/cnn_mnist/history.tsv"))
  testthat::expect_identical(history$epoch, 1L:3L)
  testthat::expect_true(any(grepl("3 epochs; 64 fit", printed, fixed = TRUE)))
  testthat::expect_true(any(grepl("epoch3/3", printed, fixed = TRUE)))

  capture.output(testthat::expect_identical(example$main(c("--quick", "epochs=2")), 0L))
  history <- read.delim(file.path(output, "models/cnn_mnist/history.tsv"))
  testthat::expect_identical(history$epoch, 1L:2L)
  testthat::expect_error(example$main("epochs=0"), "epochs must be")
  testthat::expect_error(example$main("epochs=2.5"), "epochs must be")
  testthat::expect_error(example$main("batch=2.5"), "batch must be")
  testthat::expect_error(example$main("threads=1.5"), "threads must be")
  testthat::expect_error(example$main("lr=NaN"), "lr must be")
})
