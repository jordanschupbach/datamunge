testthat::test_that("normalization and pooling validate actual samples across RNG libraries", {
    for (name in c("batch_norm", "normalization_layers", "pooling")) {
        args <- if (name == "batch_norm") "epochs=5" else character()
        testthat::expect_identical(nn_example_run(name, args), 0L)
    }
})
