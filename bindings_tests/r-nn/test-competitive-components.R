testthat::test_that("C++ competitive masks preserve signed ties and exact gradients", {
    ae <- datamungerNN::nn_components$ae
    nn <- datamungerNN::nn
    x <- nn$from_vector(c(3, 1, 1, 0, 2, 4, 0, 0, -1, -2, 5, 5, -4, -1, -2, -3), c(4L, 4L))$requires_grad_()
    expected <- c(1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0)
    mask <- ae$CompetitiveMask(x$detach(), FALSE, 2L)
    testthat::expect_equal(mask$values(), expected)
    testthat::expect_equal(nn$grad(nn$sum(nn$mul(x, mask)), list(x))[[1L]]$values(), expected)
    testthat::expect_equal(ae$CompetitiveMask(x$detach(), TRUE)$values(),
                           c(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0))
    testthat::expect_silent(ae$CompetitiveChecks())
    testthat::expect_true(ae$SameTensorBits(nn$from_vector(0, 1L), nn$from_vector(0, 1L)))
    testthat::expect_false(ae$SameTensorBits(nn$from_vector(0, 1L), nn$from_vector(-0, 1L)))
    testthat::expect_false(ae$SameTensorBits(nn$zeros(c(1L, 2L)), nn$zeros(c(2L, 1L))))
})

testthat::test_that("C++ competitive output graphs work with R-owned Adam", {
    ae <- datamungerNN::nn_components$ae
    nn <- datamungerNN::nn
    for (lifetime in c(FALSE, TRUE)) {
        params <- nn$ParameterStore(1931L)
        x <- nn$from_vector(((seq_len(4L * 784L) - 1L) * 13L %% 31L) / 30, c(4L, 784L))
        nn$no_grad(function() ae$CompetitiveAutoencoder(params, x, lifetime, FALSE))
        testthat::expect_equal(params$count(), 101200)
        testthat::expect_equal(params$size(), 4L)
        before <- params$at("encoder/W")$values()
        cfg <- nn$OptimizerConfig()
        cfg$kind <- nn$OptimizerKind$Adam
        optimizer <- nn$Optimizer(params, 0.003, cfg)
        result <- ae$CompetitiveAutoencoder(params, x, lifetime, FALSE)
        reconstruction <- result$training_reconstruction
        rm(result)
        invisible(gc())
        loss <- ae$ReconstructionMse(reconstruction, x)
        optimizer$minimize(loss)
        testthat::expect_true(is.finite(loss$item()) && loss$item() > 0)
        testthat::expect_false(identical(params$at("encoder/W")$values(), before))
        nn$no_grad(function() {
            output <- ae$CompetitiveAutoencoder(params, x, lifetime, FALSE)
            testthat::expect_equal(output$code$shape(), c(4L, 64L))
            testthat::expect_equal(output$reconstruction$shape(), c(4L, 784L))
            probe <- nn$tensor_vector(list(output$raw, output$mask, output$training_code,
                                           output$training_reconstruction))
            testthat::expect_silent(ae$CheckCompetitiveSupport(probe, lifetime, FALSE))
        })
    }
})
