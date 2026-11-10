testthat::test_that("native shared weight fields own lifetimes and retain both paths", {
    nn <- datamungerNN::nn
    p <- nn$ParameterStore(1853L)
    x <- nn$from_vector(c(0,0,1,0,0,1),c(2L,3L))
    out <- nn$TiedAutoencoder(p,x,3L,2L)
    code <- out$code;prediction <- out$reconstruction;decoder <- out$decoder_weight
    rm(out);invisible(gc())
    testthat::expect_equal(p$size(),3L)
    testthat::expect_equal(p$count(),11)
    testthat::expect_false(p$contains("decoder/W"))
    testthat::expect_equal(nn$TiedAutoencoderDecode(p,code,3L,2L)$values(),prediction$values())
    gradients <- nn$grad(nn$mse(prediction,x),p$parameters())
    testthat::expect_length(gradients,3L)
    testthat::expect_true(all(vapply(gradients,function(g) all(is.finite(g$values())),logical(1))))
    options <- nn$OptimizerConfig();options$kind <- nn$OptimizerKind$Adam
    optimizer <- nn$Optimizer(p,.003,options)
    initial <- p$at("encoder/W")$values()
    optimizer$minimize(nn$mse(nn$TiedAutoencoder(p,x,3L,2L)$reconstruction,x))
    testthat::expect_false(identical(initial,p$at("encoder/W")$values()))
    expected <- nn$transpose(p$at("encoder/W"),c(1L,0L))$contiguous()$values()
    testthat::expect_equal(nn$TiedAutoencoder(p,x,3L,2L)$decoder_weight$contiguous()$values(),expected)
    testthat::expect_equal(decoder$contiguous()$values(),expected)
})
testthat::test_that("native standalone decoder uses only matrix and decoder bias", {
    nn <- datamungerNN::nn
    for(tied in c(TRUE,FALSE)) {
        p <- nn$ParameterStore(1853L)
        code <- nn$from_vector(c(-.2,.3,.5,0),c(2L,2L))$requires_grad_()
        y <- nn$TiedAutoencoderDecode(p,code,3L,2L,tied)
        testthat::expect_equal(p$size(),2L)
        testthat::expect_equal(p$count(),9)
        testthat::expect_false(p$contains("encoder/b"))
        testthat::expect_equal(p$contains("encoder/W"),tied)
        gradients <- nn$grad(nn$sum(y),list(code,p$at(if(tied) "encoder/W" else "decoder/W"),p$at("decoder/b")))
        testthat::expect_length(gradients,3L)
        testthat::expect_true(all(vapply(gradients,function(g) any(g$values()!=0),logical(1))))
    }
    p <- nn$ParameterStore(1L)
    testthat::expect_error(nn$TiedAutoencoder(p,nn$zeros(c(1L,3L)),3L,3L),"latent")
    testthat::expect_error(nn$TiedAutoencoder(p,nn$zeros(c(0L,3L)),3L,2L),"nonempty")
    testthat::expect_equal(p$size(),0L)
    nn$TiedAutoencoder(p,nn$zeros(c(1L,3L)),3L,2L,FALSE)
    testthat::expect_error(nn$TiedAutoencoder(p,nn$zeros(c(1L,3L)),3L,2L),"decoder/W")
})
