library(testthat)
nn <- datamungerNN::nn

test_that("Conditional coupling owns endpoints and inverts with fixed context", {
  t <- nn$DType$Float64;p <- nn$ParameterStore(3049L,t)
  z <- nn$from_vector_double(c(.4,-.2),c(1L,2L));context <- nn$ones(c(1L,1L),t)
  nn$ConditionalAffinePosteriorFlow(p,"f",z,context,FALSE,FALSE,2L,2L)
  p$set("f/heads/W",nn$zeros(c(2L,2L),t));p$set("f/heads/b",nn$from_vector_double(c(atanh(.4),.3),c(2L)))
  for(swap in c(FALSE,TRUE)){
    out <- nn$ConditionalAffinePosteriorFlow(p,"f",z,context,swap,FALSE,2L,2L)
    back <- nn$ConditionalAffinePosteriorFlow(p,"f",out$value,context,swap,TRUE,2L,2L)
    expect_equal(back$value$values(),z$values(),tolerance=1e-13)
    expect_equal(out$log_determinant$item(),.2);expect_equal(back$log_determinant$item(),-.2)
  }
  q <- nn$ParameterStore(3049L,t);live <- context$clone()$requires_grad_()
  out <- nn$ConditionalAffinePosteriorFlow(q,"g",z,live,FALSE,FALSE,2L,2L)
  value <- out$value;ld <- out$log_determinant;rm(out);gc()
  gradients <- nn$grad(nn$add(nn$sum(value),nn$sum(ld)),nn$tensor_vector(list(live)))
  expect_true(any(abs(gradients[[1]]$values())>1e-12))
})

test_that("All named flow VAE fields own lifetime and host owns Adam", {
  p <- nn$ParameterStore(3001L);x <- nn$full(c(2L,3L),.3);e <- nn$full(c(2L,2L),.7)
  out <- nn$FlowVariationalAutoencoder(p,x,e,3L,2L,2L,2L)
  expect_equal(p$count(),63);expect_equal(p$size(),18)
  arrays <- nn$named_list(out$Named());expect_length(arrays,18)
  expect_equal(names(arrays)[c(1,18)],c("context","inverse_logdet"))
  loss <- out$loss;z <- out$z;probability <- out$probability;rm(out);gc()
  expect_equal(nn$sigmoid(nn$GaussianVaeDecode(p,z,3L,2L,2L))$values(),probability$values())
  expect_equal(nn$sub(arrays$logq0,arrays$logdet)$values(),arrays$logq$values())
  expect_equal(nn$sub(arrays$logq,arrays$logp)$values(),arrays$logratio$values())
  expect_equal(arrays$inverse_z0$values(),arrays$z0$values(),tolerance=1e-6)
  gradients <- nn$grad(loss,p$parameters());expect_length(gradients,18)
  config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam
  optimizer <- nn$Optimizer(p,.003,config);before <- p$flatten()
  optimizer$step(nn$tensor_vector(gradients));expect_equal(optimizer$iterations(),1)
  expect_false(identical(p$flatten(),before))
})

test_that("Default census, Float64 and malformed inputs retain native contracts", {
  p <- nn$ParameterStore(3001L)
  out <- nn$FlowVariationalAutoencoder(p,nn$zeros(c(1L,784L)),nn$zeros(c(1L,4L)))
  expect_equal(p$count(),52672);expect_equal(p$size(),18);expect_equal(out$z0$values(),out$mu$values())
  decoder <- nn$ParameterStore(1L);nn$GaussianVaeDecode(decoder,nn$zeros(c(20L,4L)))
  expect_equal(decoder$count(),26032);expect_equal(decoder$size(),4)
  t <- nn$DType$Float64;q <- nn$ParameterStore(3001L,t)
  f <- nn$FlowVariationalAutoencoder(q,nn$full(c(2L,3L),.3,t),nn$ones(c(2L,2L),t),3L,2L,2L,2L)
  expect_equal(f$loss$dtype(),t);expect_true(is.finite(f$loss$item()))
  empty <- nn$ParameterStore(1L);x <- nn$zeros(c(2L,3L));e <- nn$zeros(c(2L,2L))
  expect_error(nn$FlowVariationalAutoencoder(empty,x,e,3L,3L,2L,2L))
  expect_error(nn$FlowVariationalAutoencoder(empty,x,e,3L,2L,2L,0L))
  expect_error(nn$FlowVariationalAutoencoder(empty,nn$full(c(2L,3L),1.1),e,3L,2L,2L,2L))
  expect_error(nn$ConditionalAffinePosteriorFlow(empty,"",e,x,FALSE,FALSE,2L,2L));expect_equal(empty$size(),0)
})
