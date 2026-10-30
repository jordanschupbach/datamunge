testthat::test_that("component enum values and fixed-array accessors survive SWIG", {
  cfg <- nn_components$attention_cnn$AttentionResNetConfig()
  cfg$attention <- nn_components$attention_cnn$ChannelAttention_kEca
  testthat::expect_identical(cfg$attention, nn_components$attention_cnn$ChannelAttention_kEca)
  layout <- nn_components$dpn$DPNLayoutFor(92L)
  testthat::expect_true(layout$blocks(0L) >= 1L)
  testthat::expect_true(layout$inc(0L) >= 1L)
  unit <- nn_components$repvgg$RepVGGBlockSpec()
  unit$set_input_channels(17L)
  testthat::expect_equal(unit$input_channels(), 17L)
})

testthat::test_that("image resizing accepts integer or numeric pairs and differentiates in C++", {
  x <- nn$from_vector(c(0, 10, 20, 30), c(1L, 2L, 2L, 1L))$requires_grad_()
  resized <- nn$resize_bilinear(x, c(4L, 4L), FALSE, TRUE)
  testthat::expect_equal(resized$shape(), c(1, 4, 4, 1))
  testthat::expect_equal(resized$values(), c(0, 2.5, 7.5, 10, 5, 7.5, 12.5, 15,
                                          15, 17.5, 22.5, 25, 20, 22.5, 27.5, 30))
  testthat::expect_equal(nn$grad(nn$sum(resized), list(x))[[1L]]$values(), rep(4, 4))
  testthat::expect_equal(nn$resize_nearest(x, c(1, 1))$values(), 0)
  testthat::expect_error(nn$resize_bilinear(x, c(4.5, 4)), "integer dimensions")
  testthat::expect_error(nn$resize_bilinear(x, 4L), "integer dimensions")
})

testthat::test_that("native attention CNN loops honor epochs and restore their checkpoints", {
  root <- normalizePath(testthat::test_path("../.."))
  output <- tempfile("cifar-workflows-")
  dir.create(output)
  previous <- Sys.getenv(c("DATAMUNGE_OUTPUT_DIR", "DATAMUNGE_EXAMPLES_FULL"), unset = NA_character_)
  previous_threads <- nn$get_num_threads()
  on.exit({
    nn$set_num_threads(previous_threads)
    for (key in names(previous)) {
      if (is.na(previous[[key]])) Sys.unsetenv(key)
      else do.call(Sys.setenv, setNames(list(previous[[key]]), key))
    }
  }, add = TRUE)
  Sys.setenv(DATAMUNGE_OUTPUT_DIR = output, DATAMUNGE_EXAMPLES_FULL = "0")
  development <- nn$ImageDataset()
  development$images <- nn$zeros(c(5002L, 32L, 32L, 3L))
  development$labels <- nn$zeros(5002L, nn$DType$Int64)
  test <- nn$ImageDataset()
  test$images <- nn$ones(c(3L, 32L, 32L, 3L))
  test$labels <- nn$from_vector_int64(c(1, 2, 3), 3L)
  models <- c(senet = "se_resnet50_compact_cifar10", eca_net = "eca_resnet50_compact_cifar10",
              alexnet = "alexnet_compact_cifar10", xception = "xception_compact_cifar10",
              densenet = "densenet_bc40_k12_cifar10", ghostnet = "ghostnet_cifar10",
              mobilenet_v2 = "mobilenet_v2_cifar10", mnasnet = "mnasnet_a1_cifar10",
              convnext = "convnext_atto_cifar10", efficientnet = "efficientnet_b0_cifar10",
              efficientnet_v2 = "efficientnet_v2_b0_cifar10",
              preactivation_resnet = "preactivation_resnet164_compact_cifar10",
              pyramidnet = "pyramidnet110_a48_cifar10", res2net = "res2net50_compact_cifar10",
              resnext = "resnext29_8x4d_compact_cifar10", revnet = "revnet38_cifar10",
              cbam = "cbam_resnet50_compact_cifar10", non_local = "nonlocal_resnet50_compact_cifar10",
              octconv = "oct_resnet50_compact_cifar10", repvgg = "repvgg_a0_cifar10",
              sknet = "sknet50_compact_cifar10")
  for (name in names(models)) {
    example <- new.env()
    file <- datamungerNN:::.nn_example_files()[[name]]
    sys.source(file, example)
    # Run real models/autograd/Adam/checkpoints on two fit and three test rows;
    # full-size diagnostic derivatives are exercised separately by port validation.
    example$full_checks <- function() invisible(NULL)
    key <- paste0(name, "_checks")
    if (exists(key, example, inherits = FALSE)) {
      diagnostics <- as.list(example[[key]])
      for (check in grep("Checks?$", names(diagnostics), value = TRUE))
        diagnostics[[check]] <- function(...) invisible(NULL)
      example[[key]] <- list2env(diagnostics, parent = emptyenv())
    }
    example$image_training <- list2env(as.list(example$image_training), parent = emptyenv())
    example$image_training$LoadCifar10 <- function(path) list(first = development, second = test)
    printed <- capture.output(testthat::expect_identical(
      example$main(c("epochs=3", "fit=2", "batch=2", "threads=2")), 0L))
    model <- models[[name]]
    directory <- file.path(output, "models", model)
    history <- read.delim(file.path(directory, "history.tsv"))
    testthat::expect_identical(history$epoch, 1L:3L)
    testthat::expect_true(file.exists(file.path(directory, "best_params.npz")))
    testthat::expect_identical(file.exists(file.path(directory, "best_state.npz")), !name %in% c("alexnet", "convnext"))
    testthat::expect_true(any(grepl("confusion (truth rows)", printed, fixed = TRUE)))
  }
})
