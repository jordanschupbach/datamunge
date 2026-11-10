library(testthat)
nn <- datamungerNN::nn

test_that("Gumbel relaxation keeps live inputs and temperature-independent hard IDs", {
  logits <- nn$from_vector_double(c(.1,-.4,.7),c(1L,3L))$requires_grad_()
  noise <- nn$from_vector_double(c(.3,-.2,.6,-.1,.4,.2),c(2L,3L))$requires_grad_()
  out <- nn$GumbelCategorical(logits,noise,.7);q <- out$probability;y <- out$relaxed;ids <- out$indices
  kl <- nn$CategoricalUniformKl(q,out$log_probability)
  expect_equal(q$shape(),c(1,3));expect_equal(y$shape(),c(2,3))
  expected <- exp(c(.1,-.4,.7));expected <- expected/sum(expected)
  expect_equal(q$values(),expected,tolerance=1e-13);expect_equal(kl$item(),sum(expected*log(3*expected)),tolerance=1e-13)
  objective <- nn$add(nn$mul(nn$sum(nn$square(y)),.5),nn$mul(nn$sum(kl),.3));rm(out);gc()
  g <- nn$grad(objective,nn$tensor_vector(list(logits,noise)))
  expect_true(all(vapply(g,function(x)any(abs(x$values())>1e-12),logical(1))))
  cold <- nn$GumbelCategorical(logits,noise,.25);warm <- nn$GumbelCategorical(logits,noise,2)
  expect_equal(ids$values(),cold$indices$values());expect_equal(ids$values(),warm$indices$values())
  expect_false(identical(cold$relaxed$values(),warm$relaxed$values()));expect_false(ids$requires_grad())
})

test_that("Explicit banks own lifetime and retain full unsigned32 seeds", {
  bank <- nn$GumbelNoise(2L,3L,4294967295,1L,3L)
  uniform <- bank$uniform;gumbel <- bank$gumbel;noise <- bank$noise;rm(bank);gc()
  expect_equal(noise$shape(),c(3,2,1,3));expect_false(noise$requires_grad())
  expect_equal(noise$values(),nn$GumbelNoise(2L,3L,4294967295,1L,3L)$noise$values())
  expect_equal(gumbel$values(),-log(-log(uniform$values())),tolerance=1e-14)
  expect_equal(noise$values(),nn$cast(gumbel,nn$DType$Float32)$values())
})

test_that("Every categorical field owns lifetime and R owns Adam", {
  t <- nn$DType$Float64;p <- nn$ParameterStore(2903L,t);x <- nn$full(c(2L,3L),.3,t)
  noise <- nn$select(nn$GumbelNoise(2L,1L,2939L,2L,2L)$gumbel,0L,0L)
  out <- nn$GumbelCategoricalAutoencoder(p,x,noise,.7,3L,2L,2L,2L)
  expect_equal(p$count(),39);expect_equal(p$size(),8)
  fields <- lapply(c("logits","log_probability","probability","relaxed","indices","hard","code","soft_prediction","hard_prediction","nll","hard_nll","kl","loss","hard_objective"),function(name)do.call("$",list(out,name)))
  names(fields) <- c("logits","log_probability","probability","relaxed","indices","hard","code","soft_prediction","hard_prediction","nll","hard_nll","kl","loss","hard_objective")
  rm(out);gc();expect_equal(fields$indices$dtype(),nn$DType$Int32);expect_equal(fields$hard$dtype(),t)
  expect_false(fields$hard$requires_grad());expect_equal(fields$code$shape(),c(2,4))
  expect_equal(nn$GumbelDecodeSoft(p,fields$relaxed,3L,2L,2L,2L)$values(),fields$soft_prediction$values())
  expect_equal(nn$GumbelDecodeIds(p,fields$indices,3L,2L,2L,2L)$values(),fields$hard_prediction$values())
  expect_equal(nn$GumbelDecodeIds(p,nn$cast(fields$indices,nn$DType$Int64),3L,2L,2L,2L)$values(),fields$hard_prediction$values())
  gradients <- nn$grad(fields$loss,p$parameters());expect_length(gradients,8)
  config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam
  optimizer <- nn$Optimizer(p,.003,config);before <- p$flatten();optimizer$step(nn$tensor_vector(gradients))
  expect_equal(optimizer$iterations(),1);expect_false(identical(before,p$flatten()))
})

test_that("Default census and malformed calls preserve empty stores", {
  p <- nn$ParameterStore(2903L);nn$GumbelCategoricalAutoencoder(p,nn$zeros(c(1L,784L)),nn$zeros(c(1L,4L,8L)))
  expect_equal(p$count(),53104);expect_equal(p$size(),8)
  decoder <- nn$ParameterStore(1L);nn$GumbelDecodeSoft(decoder,nn$full(c(2L,4L,8L),.125))
  expect_equal(decoder$count(),26928);expect_equal(decoder$size(),4)
  empty <- nn$ParameterStore(1L);x <- nn$zeros(c(2L,3L));noise <- nn$zeros(c(2L,2L,2L))
  for(temp in c(0,-1,NaN,Inf,1e-300))expect_error(nn$GumbelCategoricalAutoencoder(empty,x,noise,temp,3L,2L,2L,2L))
  expect_error(nn$GumbelDecodeSoft(empty,nn$zeros(c(1L,2L,2L)),3L,2L,2L,2L))
  for(bad in c(-1L,2L))expect_error(nn$GumbelDecodeIds(empty,nn$from_vector_int64(c(bad,0L),c(1L,2L)),3L,2L,2L,2L))
  expect_equal(empty$size(),0)
})
