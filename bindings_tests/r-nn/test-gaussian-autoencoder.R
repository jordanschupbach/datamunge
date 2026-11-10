library(testthat)
nn <- datamungerNN::nn

test_that("Vanilla dimensions preserve compatibility ownership gradients and caller Adam", {
  t <- nn$DType$Float64;p <- nn$ParameterStore(1951L,t)
  image <- nn$full(c(2L,784L),.25,t)$requires_grad_();noise <- nn$full(c(2L,2L),.2,t)$requires_grad_()
  canonical <- nn$BetaVariationalAutoencoder(p,image,noise,1,784L,2L,64L)
  compatibility <- datamungerNN::nn_components$ae$VariationalAutoencoder(p,image,noise)
  fields <- c("mu","log_variance","z","logits","probability","reconstruction_nll","kl","loss")
  saved <- lapply(fields,function(name)do.call("$",list(compatibility,name)));names(saved) <- fields
  rm(compatibility);gc();expect_equal(p$size(),10);expect_equal(p$count(),101652)
  for(name in fields)expect_equal(saved[[name]]$values(),do.call("$",list(canonical,name))$values(),tolerance=0)
  expect_equal(nn$sigmoid(nn$GaussianVaeDecode(p,saved$z,784L,2L,64L))$values(),saved$probability$values(),tolerance=0)
  derivatives <- nn$grad(canonical$loss,nn$tensor_vector(c(nn$tensor_list(p$parameters()),list(image,noise))))
  expect_length(derivatives,12);expect_true(all(vapply(derivatives,function(a)all(is.finite(a$values())),logical(1))))
  expect_true(all(vapply(tail(derivatives,2),function(a)any(abs(a$values())>1e-12),logical(1))))
  before <- p$flatten();config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam
  optimizer <- nn$Optimizer(p,.003,config);optimizer$step(nn$tensor_vector(derivatives[seq_len(10)]))
  expect_equal(optimizer$iterations(),1);expect_false(identical(p$flatten(),before))
})

test_that("Gaussian endpoints preserve ownership, paired controls and explicit Adam", {
  p <- nn$ParameterStore(2237L)
  x <- nn$full(c(2L,3L),.3);e <- nn$full(c(2L,2L),.7)
  out <- nn$BetaVariationalAutoencoder(p,x,e,4,3L,2L,2L)
  expect_equal(p$size(),10);expect_equal(p$count(),35)
  probability <- out$probability;mu <- out$mu;loss <- out$loss
  saved <- probability$values();rm(out);gc()
  one <- nn$BetaVariationalAutoencoder(p,x,e,1,3L,2L,2L)
  expect_equal(one$probability$values(),saved)
  expect_equal(one$mu$values(),mu$values())
  expect_equal(loss$item()-one$loss$item(),3*nn$mean(one$kl)$item(),tolerance=2e-6)
  expect_equal(nn$sigmoid(nn$GaussianVaeDecode(p,one$z,3L,2L,2L))$values(),saved)
  zero <- nn$BetaVariationalAutoencoder(p,x,nn$zeros(c(2L,2L)),4,3L,2L,2L)
  expect_equal(zero$z$values(),zero$mu$values())
  gradients <- nn$grad(loss,p$parameters());expect_length(gradients,10)
  config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam
  optimizer <- nn$Optimizer(p,.003,config);before <- p$flatten()
  optimizer$step(nn$tensor_vector(gradients))
  expect_equal(optimizer$iterations(),1);expect_false(identical(before,p$flatten()))
})
test_that("Native generic factor probes use only fitting labels for fitting", {
  fit <- nn$from_vector_double(c(-1,-1,0,0,1,1),c(3L,2L))
  test <- nn$from_vector_double(c(2,2,3,3),c(2L,2L))
  y <- nn$from_vector_double(c(1,3,5),c(3L,1L));yt <- nn$from_vector_double(c(7,9),c(2L,1L))
  lv <- nn$zeros(c(3L,2L),nn$DType$Float64)
  a <- nn$LatentFactorDiagnostics(fit,test,y,yt,lv)
  b <- nn$LatentFactorDiagnostics(fit,test,y,nn$from_vector_double(c(-100,-99),c(2L,1L)),lv)
  expect_length(a,13);expect_equal(a$best_axis_by_fit$values(),0)
  for(name in c("latent_mean","latent_std","factor_mean","ridge_coefficients","best_axis_by_fit","ridge_prediction","single_prediction")) expect_equal(a[[name]]$values(),b[[name]]$values())
  expect_gt(a$ridge_r2$item(),.99999);expect_lt(b$ridge_r2$item(),0)
  prediction <- a$ridge_prediction;rm(a);gc();expect_false(prediction$requires_grad())
})
test_that("Gaussian validation precedes parameter creation", {
  p <- nn$ParameterStore(1L)
  for(beta in c(0,-1,Inf,NaN,1e-100,1e100)) expect_error(nn$BetaVariationalAutoencoder(p,nn$zeros(c(2L,3L)),nn$zeros(c(2L,2L)),beta,3L,2L,2L))
  expect_equal(p$size(),0)
  expect_equal(nn$DiagonalGaussianKl(nn$ones(c(2L,3L)),nn$zeros(c(2L,3L)))$values(),c(1.5,1.5))
})
