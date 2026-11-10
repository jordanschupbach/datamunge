testthat::test_that("C++ competitive masks preserve signed ties and exact gradients", {
    ae <- datamungerNN::nn_components$ae
    nn <- datamungerNN::nn
    x <- nn$from_vector(c(3, 1, 1, 0, 2, 4, 0, 0, -1, -2, 5, 5, -4, -1, -2, -3), c(4L, 4L))$requires_grad_()
    expected <- c(1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0)
    mask <- nn$CompetitiveMask(x$detach(), FALSE, 2L)
    testthat::expect_equal(mask$values(), expected)
    testthat::expect_equal(nn$grad(nn$sum(nn$mul(x, mask)), list(x))[[1L]]$values(), expected)
    testthat::expect_equal(nn$CompetitiveMask(x$detach(), TRUE)$values(),
                           c(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0))
    testthat::expect_equal(nn$grad(nn$sum(nn$mul(x,nn$CompetitiveMask(x,TRUE))),list(x))[[1L]]$values(),
                           c(1,0,0,0,0,1,0,0,0,0,1,1,0,0,0,0))
    testthat::expect_true(ae$SameTensorBits(nn$from_vector(0, 1L), nn$from_vector(0, 1L)))
    testthat::expect_false(ae$SameTensorBits(nn$from_vector(0, 1L), nn$from_vector(-0, 1L)))
    testthat::expect_false(ae$SameTensorBits(nn$zeros(c(1L, 2L)), nn$zeros(c(2L, 1L))))
})

testthat::test_that("C++ competitive output graphs work with R-owned Adam", {
    ae <- datamungerNN::nn_components$ae
    nn <- datamungerNN::nn
    for (lifetime in c(FALSE, TRUE)) {
        params <- nn$ParameterStore(1931L)
        x <- nn$from_vector((((seq_len(4L * 784L) - 1L) * 13L) %% 31L) / 30, c(4L, 784L))
        testthat::expect_true(all(x$values() >= 0 & x$values() <= 1))
        nn$no_grad(function() nn$CompetitiveAutoencoder(params, x, lifetime, FALSE))
        testthat::expect_equal(params$count(), 101200)
        testthat::expect_equal(params$size(), 4L)
        before <- params$at("encoder/W")$values()
        cfg <- nn$OptimizerConfig()
        cfg$kind <- nn$OptimizerKind$Adam
        optimizer <- nn$Optimizer(params, 0.003, cfg)
        result <- nn$CompetitiveAutoencoder(params, x, lifetime, FALSE)
        reconstruction <- result$training_reconstruction
        rm(result)
        invisible(gc())
        loss <- ae$ReconstructionMse(reconstruction, x)
        optimizer$minimize(loss)
        testthat::expect_true(is.finite(loss$item()) && loss$item() > 0)
        testthat::expect_false(identical(params$at("encoder/W")$values(), before))
        nn$no_grad(function() {
            output <- nn$CompetitiveAutoencoder(params, x, lifetime, FALSE)
            testthat::expect_equal(output$code$shape(), c(4L, 64L))
            testthat::expect_equal(output$reconstruction$shape(), c(4L, 784L))
            probe <- nn$tensor_vector(list(output$raw, output$mask, output$training_code,
                                           output$training_reconstruction))
            values <- matrix(output$mask$values(),nrow=4L,byrow=TRUE)
            testthat::expect_true(if(lifetime) all(colSums(values)==1) else all(rowSums(values)==8))
        })
    }
})

testthat::test_that("canonical competition owns endpoints and preserves signed winners", {
    nn <- datamungerNN::nn
    scores <- nn$from_vector(c(-4,-1,-2,-3),c(1L,4L))$requires_grad_()
    mask <- nn$CompetitiveMask(scores,FALSE,2L)
    testthat::expect_equal(mask$values(),c(0,1,1,0))
    testthat::expect_false(mask$requires_grad())
    testthat::expect_equal(nn$grad(nn$sum(nn$mul(scores,mask)),list(scores))[[1L]]$values(),c(0,1,1,0))
    p <- nn$ParameterStore(1931L)
    x <- nn$from_vector(c(.1,.2,.8,.7,.3,.4),c(2L,3L))
    result <- nn$CompetitiveAutoencoder(p,x,FALSE,FALSE,3L,4L,2L)
    testthat::expect_equal(p$count(),31)
    testthat::expect_equal(sum(result$mask$values()),4)
    testthat::expect_equal(result$code$values(),result$training_code$values())
    testthat::expect_equal(result$reconstruction$values(),result$training_reconstruction$values())
    derivatives <- nn$grad(nn$mse(result$training_reconstruction,x),p$parameters())
    testthat::expect_length(derivatives,4L)
    testthat::expect_true(all(vapply(derivatives,function(g)all(is.finite(g$values())),logical(1))))
    reconstruction <- result$reconstruction
    expected <- reconstruction$values()
    rm(result)
    invisible(gc())
    testthat::expect_equal(reconstruction$values(),expected)
    testthat::expect_error(nn$CompetitiveMask(nn$full(c(1L,4L),NaN),FALSE,2L),"finite")
})
