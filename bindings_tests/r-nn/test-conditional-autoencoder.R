library(testthat)
nn <- datamungerNN::nn

test_that("Conditional fields own inherited endpoints and caller-owned Adam", {
  p <- nn$ParameterStore(2111L)
  x <- nn$full(c(2L,3L),.3);epsilon <- nn$full(c(2L,2L),.7)
  labels <- nn$cast(nn$from_vector(c(0L,2L),c(2L)),nn$DType$Int64)
  out <- nn$ConditionalVariationalAutoencoder(p,x,labels,epsilon,3L,3L,2L,2L)
  expect_equal(p$size(),10);expect_equal(p$count(),47)
  expect_equal(out$condition$values(),c(1,0,0,0,0,1))
  expect_equal(out$encoder_input$shape(),c(2,6));expect_equal(out$decoder_input$shape(),c(2,5))
  probability <- out$probability;mu <- out$mu;z <- out$z;loss <- out$loss
  expect_equal(loss$item(),out$unweighted_loss$item());saved <- probability$values();rm(out);gc()
  expect_equal(nn$sigmoid(nn$ConditionalVaeDecode(p,z,labels,3L,3L,2L,2L))$values(),saved)
  zero <- nn$ConditionalVariationalAutoencoder(p,x,labels,nn$zeros(c(2L,2L)),3L,3L,2L,2L)
  expect_equal(zero$mu$values(),mu$values());expect_equal(zero$z$values(),zero$mu$values())
  gradients <- nn$grad(loss,p$parameters());expect_length(gradients,10)
  config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam
  optimizer <- nn$Optimizer(p,.003,config);before <- p$flatten()
  optimizer$step(nn$tensor_vector(gradients));expect_equal(optimizer$iterations(),1)
  expect_false(identical(before,p$flatten()))
})
test_that("Discrete class validation precedes every new leaf", {
  p <- nn$ParameterStore(1L)
  for(labels in list(nn$from_vector_int64(c(-1,0),c(2L)),nn$from_vector_int64(c(0,3),c(2L)),nn$zeros(c(2L)),nn$zeros(c(1L),nn$DType$Int32))) {
    expect_error(nn$ConditionalVariationalAutoencoder(p,nn$zeros(c(2L,3L)),labels,nn$zeros(c(2L,2L)),3L,3L,2L,2L))
    expect_error(nn$ConditionalVaeDecode(p,nn$zeros(c(2L,2L)),labels,3L,3L,2L,2L))
  }
  expect_equal(p$size(),0)
})
test_that("Float64 one-hot dtype and standalone decoder need no image", {
  t <- nn$DType$Float64;p <- nn$ParameterStore(2111L,t)
  labels <- nn$from_vector_int64(c(0,1),c(2L))
  out <- nn$ConditionalVariationalAutoencoder(p,nn$zeros(c(2L,1L),t),labels,nn$full(c(2L,1L),.5,t),1L,2L,1L,1L)
  expect_equal(out$condition$dtype(),t);expect_equal(p$count(),14)
  decoder <- nn$ParameterStore(99L)
  logits <- nn$ConditionalVaeDecode(decoder,nn$zeros(c(2L,2L)),labels,3L,3L,2L,2L)
  expect_equal(decoder$size(),4);expect_equal(decoder$count(),21);expect_equal(logits$shape(),c(2,3))
})
