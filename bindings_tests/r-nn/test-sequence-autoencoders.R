testthat::test_that("native recurrent outputs own histories and keep the complete graph", {
    nn <- datamungerNN::nn
    p <- nn$ParameterStore(1871L)
    images <- nn$from_vector((((0:1567)*13) %% 31)/30,c(2L,28L,28L,1L))
    testthat::expect_true(all(images$values() >= 0 & images$values() <= 1))
    x <- nn$ImageRowSequence(images)
    testthat::expect_equal(x$shape(),c(2L,28L,28L))
    testthat::expect_equal(x$values(),images$values())
    result <- nn$RecurrentAutoencoder(p,x)
    testthat::expect_equal(p$size(),8L)
    testthat::expect_equal(p$count(),4956)
    testthat::expect_equal(result$encoder_states$shape(),c(2L,28L,32L))
    testthat::expect_equal(result$decoder_states$shape(),c(2L,28L,32L))
    code <- result$code; reconstruction <- result$reconstruction
    expected <- reconstruction$values()
    rm(result);invisible(gc())
    testthat::expect_equal(reconstruction$values(),expected)
    testthat::expect_equal(nn$RecurrentDecode(p,code)$reconstruction$values(),expected)
    gradients <- nn$grad(nn$mse(reconstruction,x),p$parameters())
    testthat::expect_length(gradients,8L)
    testthat::expect_true(all(vapply(gradients,function(g) all(is.finite(g$values())) && any(g$values()!=0),logical(1))))
    testthat::expect_false(nn$RecurrentEncode(p,x)$reconstruction$defined())
    testthat::expect_false(nn$RecurrentDecode(p,code)$encoder_states$defined())
})

testthat::test_that("native sequence gradients reach every encoder step and validate shapes", {
    nn <- datamungerNN::nn
    x <- nn$from_vector_double(c(.1,.6,.7,.2,.4,.8),c(1L,3L,2L))$requires_grad_()
    p <- nn$ParameterStore(17L,nn$DType$Float64)
    code <- nn$RecurrentEncode(p,x,3L,2L,2L)$code
    gradient <- nn$grad(nn$sum(code),list(x))[[1L]]
    for(t in 0:2) testthat::expect_true(any(nn$select(gradient,1L,t)$values()!=0))
    testthat::expect_equal(nn$grad(nn$sum(nn$ImageRowSequence(x,3L,2L)),list(x))[[1L]]$values(),rep(1,6))
    empty <- nn$ParameterStore(1L)
    testthat::expect_error(nn$RecurrentAutoencoder(empty,nn$zeros(c(1L,6L)),3L,2L,2L),"B,T,F")
    testthat::expect_error(nn$RecurrentAutoencoder(empty,nn$zeros(c(1L,3L,2L)),3L,2L,6L),"0<U")
    testthat::expect_equal(empty$size(),0L)
})
