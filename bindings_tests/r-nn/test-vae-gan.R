library(testthat)
nn <- datamungerNN::nn

test_that("Feature NLL sums coordinates and detaches real targets", {
  t <- nn$DType$Float64;real <- nn$ones(c(2L,3L),t)$requires_grad_();rec <- nn$zeros(c(2L,3L),t)$requires_grad_()
  loss <- nn$VaeGanFeatureNll(real,rec)
  expect_equal(loss$values(),c(1.5,1.5))
  expect_equal(nn$grad(nn$sum(loss),nn$tensor_vector(list(rec)))[[1]]$values(),rep(-1,6))
  expect_error(nn$grad(nn$sum(nn$VaeGanFeatureNll(real,rec)),nn$tensor_vector(list(real))))
  expect_error(nn$VaeGanFeatureNll(real,nn$zeros(c(2L,2L),t)))
})

test_that("All nineteen fields own lifetime and explicit continuous paths", {
  t <- nn$DType$Float64;e <- nn$ParameterStore(3301L,t);g <- nn$ParameterStore(3307L,t);d <- nn$ParameterStore(3313L,t)
  x <- nn$full(c(2L,3L),.3,t)$requires_grad_();epsilon <- nn$full(c(2L,2L),.2,t)$requires_grad_();prior <- nn$full(c(3L,2L),-.4,t)$requires_grad_()
  out <- nn$VaeGan(e,g,d,x,epsilon,prior,3L,2L,2L,2L)
  names <- c("mu","log_variance","z","probability","prior_probability","real_feature","reconstruction_feature","prior_feature","real_logit","reconstruction_logit","prior_logit","feature_nll","kl","pixel_nll","mse","encoder_loss","generator_loss","discriminator_loss","generator_adversarial")
  fields <- lapply(names,function(name)do.call("$",list(out,name)));names(fields) <- names;rm(out);gc()
  expect_length(fields,19);expect_equal(e$count()+g$count()+d$count(),46)
  expect_equal(fields$probability$shape(),c(2,3));expect_equal(fields$prior_probability$shape(),c(3,3))
  expect_true(all(vapply(fields,function(a)all(is.finite(a$values())),logical(1))))
  derivatives <- nn$grad(fields$generator_loss,nn$tensor_vector(list(x,epsilon,prior)))
  expect_length(derivatives,3);expect_true(all(vapply(derivatives,function(a)any(abs(a$values())>1e-12),logical(1))))
  expect_equal(nn$VaeGanDecode(g,fields$z,3L,2L,2L)$values(),fields$probability$values(),tolerance=0)
  other <- nn$VaeGan(e,g,d,x,epsilon,nn$ones(c(4L,2L),t),3L,2L,2L,2L)
  expect_equal(other$probability$values(),fields$probability$values(),tolerance=0)
  expect_equal(other$encoder_loss$item(),fields$encoder_loss$item(),tolerance=0)
})

test_that("Standalone networks preserve default census and own fields", {
  e <- nn$ParameterStore(3301L);g <- nn$ParameterStore(3307L);d <- nn$ParameterStore(3313L)
  nn$VaeGan(e,g,d,nn$zeros(c(2L,784L)),nn$zeros(c(2L,4L)),nn$zeros(c(3L,4L)))
  expect_equal(c(e$count(),g$count(),d$count()),c(12696,13408,6289))
  expect_equal(c(e$size(),g$size(),d$size()),c(6,4,4))
  code <- nn$full(c(2L,4L),-.2)$requires_grad_();generator <- nn$ParameterStore(3307L)
  pixels <- nn$VaeGanDecode(generator,code);expect_equal(pixels$shape(),c(2,784));expect_equal(generator$size(),4)
  expect_true(any(abs(nn$grad(nn$sum(pixels),nn$tensor_vector(list(code)))[[1]]$values())>1e-12))
  x <- nn$full(c(2L,3L),-.2)$requires_grad_();out <- nn$VaeGanDiscriminator(nn$ParameterStore(3313L),x,3L,2L)
  feature <- out$feature;logit <- out$logit;rm(out);gc()
  expect_equal(feature$shape(),c(2,2));expect_equal(logit$shape(),c(2,1))
  expect_true(any(abs(nn$grad(nn$sum(logit),nn$tensor_vector(list(x)))[[1]]$values())>1e-12))
})

test_that("Three host-owned Adam states update only their recomputed selected groups", {
  t <- nn$DType$Float64;stores <- lapply(c(3301L,3307L,3313L),function(seed)nn$ParameterStore(seed,t))
  x <- nn$full(c(2L,3L),.3,t);epsilon <- nn$full(c(2L,2L),.2,t);prior <- nn$full(c(3L,2L),-.4,t)
  forward <- function()nn$VaeGan(stores[[1]],stores[[2]],stores[[3]],x,epsilon,prior,3L,2L,2L,2L)
  forward();config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam
  optimizers <- lapply(stores,function(p)nn$Optimizer(p,.003,config))
  for(group in c(3L,1L,2L)){
    before <- lapply(stores,function(p)p$flatten());out <- forward()
    objective <- list(out$encoder_loss,out$generator_loss,out$discriminator_loss)[[group]]
    optimizers[[group]]$step(nn$tensor_vector(nn$grad(objective,stores[[group]]$parameters())))
    changed <- vapply(seq_along(stores),function(k)!identical(stores[[k]]$flatten(),before[[k]]),logical(1))
    expect_equal(changed,seq_along(stores)==group)
  }
  expect_equal(vapply(optimizers,function(opt)opt$iterations(),numeric(1)),rep(1,3))
})

test_that("Malformed inputs stores and widths fail before creation", {
  e <- nn$ParameterStore(3301L);g <- nn$ParameterStore(3307L);d <- nn$ParameterStore(3313L)
  x <- nn$zeros(c(2L,3L));epsilon <- nn$zeros(c(2L,2L));prior <- nn$zeros(c(3L,2L))
  expect_error(nn$VaeGan(e,g,d,nn$full(c(2L,3L),1.1),epsilon,prior,3L,2L,2L,2L))
  expect_error(nn$VaeGan(e,e,d,x,epsilon,prior,3L,2L,2L,2L))
  expect_error(nn$VaeGan(e,g,d,x,nn$zeros(c(1L,2L)),prior,3L,2L,2L,2L))
  expect_error(nn$VaeGan(e,g,d,x,epsilon,nn$zeros(c(0L,2L)),3L,2L,2L,2L))
  expect_error(nn$VaeGan(e,g,d,x,epsilon,prior,3L,2L,2L,0L))
  expect_equal(e$size()+g$size()+d$size(),0)
})
