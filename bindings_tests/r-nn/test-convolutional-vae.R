library(testthat)
nn <- datamungerNN::nn

test_that("Convolutional VAE fields and decoder stages own their lifetime", {
  p <- nn$ParameterStore(2011L);x <- nn$full(c(2L,4L,4L,1L),.3);epsilon <- nn$full(c(2L,1L),.7)
  out <- nn$ConvolutionalVariationalAutoencoder(p,x,epsilon,1L,2L,1L)
  expect_equal(p$size(),14);expect_equal(p$count(),84)
  expect_equal(out$pool2$shape(),c(2,1,1,1));expect_equal(out$decoder1$shape(),c(2,2,2,2))
  probability <- out$probability;mu <- out$mu;z <- out$z;loss <- out$loss;base <- out$decoder_base
  expect_equal(loss$item(),out$unweighted_loss$item());saved <- probability$values();rm(out);gc()
  decoded <- nn$ConvolutionalVaeDecode(p,z,4L,4L,1L,2L,1L)
  expect_equal(decoded$probability$values(),saved);expect_equal(decoded$decoder_base$values(),base$values())
  retained <- decoded$decoder1;rm(decoded);gc();expect_equal(retained$shape(),c(2,2,2,2))
  zero <- nn$ConvolutionalVariationalAutoencoder(p,x,nn$zeros(c(2L,1L)),1L,2L,1L)
  expect_equal(zero$mu$values(),mu$values());expect_equal(zero$z$values(),zero$mu$values())
  gradients <- nn$grad(loss,p$parameters());expect_length(gradients,14)
  config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam
  optimizer <- nn$Optimizer(p,.003,config);before <- p$flatten();optimizer$step(nn$tensor_vector(gradients))
  expect_equal(optimizer$iterations(),1);expect_false(identical(before,p$flatten()))
})
test_that("Default geometry and code-only decoder have the original census", {
  p <- nn$ParameterStore(2011L);out <- nn$ConvolutionalVariationalAutoencoder(p,nn$zeros(c(1L,28L,28L,1L)),nn$zeros(c(1L,2L)))
  expect_equal(p$count(),2117);expect_equal(p$size(),14);expect_equal(out$probability$shape(),c(1,784))
  decoder <- nn$ParameterStore(7L);d <- nn$ConvolutionalVaeDecode(decoder,nn$zeros(c(1L,2L)))
  expect_equal(decoder$count(),957);expect_equal(decoder$size(),6);expect_equal(d$spatial_logits$shape(),c(1,28,28,1))
})
test_that("Malformed spatial geometry is rejected before creating leaves", {
  p <- nn$ParameterStore(1L)
  for(shape in list(c(0L,4L,4L,1L),c(1L,5L,4L,1L),c(1L,4L,6L,1L),c(1L,4L,4L,2L),c(4L,4L,1L)))
    expect_error(nn$ConvolutionalVariationalAutoencoder(p,nn$zeros(shape),nn$zeros(c(1L,2L))))
  expect_error(nn$ConvolutionalVaeDecode(p,nn$zeros(c(1L,2L)),6L,4L));expect_equal(p$size(),0)
  t <- nn$DType$Float64;q <- nn$ParameterStore(2011L,t)
  o <- nn$ConvolutionalVariationalAutoencoder(q,nn$full(c(2L,4L,8L,1L),.3,t),nn$ones(c(2L,1L),t),1L,2L,1L)
  expect_equal(o$spatial_logits$shape(),c(2,4,8,1));expect_equal(o$probability$dtype(),t)
})
