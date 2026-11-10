library(testthat)
nn <- datamungerNN::nn
sentence_targets <- function() nn$from_vector_int64(c(3,5,8,11,13,2,0,4,7,10,12,16,2,0),c(2L,7L))

test_that("All eighteen sentence fields own lifetime with live BPTT and noise", {
  t <- nn$DType$Float64;p <- nn$ParameterStore(3109L,t)
  e <- nn$full(c(2L,4L),.2,t)$requires_grad_()
  out <- nn$SequenceVariationalAutoencoder(p,sentence_targets(),e)
  names <- c("decoder_inputs","encoder_state","encoder_states","mu","log_variance","z",
             "decoder_initial","decoder_states","logits","probability","predicted_ids","mask",
             "token_nll","nll","kl","correct","valid_tokens","loss")
  fields <- lapply(names,function(name)do.call("$",list(out,name)));names(fields) <- names
  rm(out);gc()
  expect_equal(p$size(),22);expect_equal(p$count(),3243)
  expect_equal(fields$logits$shape(),c(2,7,19));expect_equal(fields$valid_tokens$values(),c(6,6))
  expect_true(all(vapply(fields,function(a)all(is.finite(a$values())),logical(1))))
  g <- nn$grad(fields$loss,nn$tensor_vector(c(nn$tensor_list(p$parameters()),list(e))))
  expect_length(g,23);expect_true(any(abs(g[[23]]$values())>1e-10))
  teacher <- nn$SequenceTeacher(p,fields$z,fields$decoder_inputs)
  expect_equal(teacher$probability$values(),fields$probability$values())
})

test_that("PAD encoder and five prefix fields survive garbage collection", {
  p <- nn$ParameterStore(3109L);out <- nn$SequenceEncoder(p,sentence_targets())
  state <- out$state;states <- out$states;rm(out);gc()
  expect_equal(p$size(),7);expect_equal(p$count(),1352);expect_equal(states$shape(),c(2,7,16))
  padded <- nn$concat(nn$tensor_vector(list(sentence_targets(),nn$zeros(c(2L,1L),nn$DType$Int64))),1L)
  expect_equal(nn$SequenceEncoder(p,padded)$state$values(),state$values())
  decoder <- nn$ParameterStore(3109L);z <- nn$full(c(2L,4L),.2)$requires_grad_()
  out <- nn$SequenceTeacher(decoder,z,nn$SequenceDecoderInputs(sentence_targets()))
  fields <- lapply(c("initial","states","logits","probability","predicted_ids"),function(name)do.call("$",list(out,name)))
  rm(out);gc();expect_equal(decoder$size(),11);expect_equal(decoder$count(),1755)
  expect_true(all(vapply(fields,function(a)all(is.finite(a$values())),logical(1))))
  g <- nn$grad(nn$sum(fields[[3]]),nn$tensor_vector(list(z)))
  expect_true(any(abs(g[[1]]$values())>1e-10))
})

test_that("Explicit generation fields and optimizer remain independent of containers", {
  p <- nn$ParameterStore(3109L);z <- nn$zeros(c(2L,4L))
  out <- nn$SequenceGenerate(p,z,TRUE,nn$full(c(2L,7L),.5,nn$DType$Float64))
  fields <- lapply(c("ids","logits","probability","ended"),function(name)do.call("$",list(out,name)))
  rm(out);gc();expect_true(all(vapply(fields,function(a)all(is.finite(a$values())),logical(1))))
  expect_equal(fields[[1]]$shape(),c(2,7));expect_false(fields[[1]]$requires_grad())
  out <- nn$SequenceVariationalAutoencoder(p,sentence_targets(),z);before <- p$flatten()
  expect_equal(nn$CollapseBeta(TRUE,0L),0);expect_equal(nn$CollapseBeta(TRUE,17L),1)
  expect_equal(nn$AnnealedSequenceObjective(out,0)$item(),nn$mean(out$nll)$item())
  config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam
  opt <- nn$Optimizer(p,.003,config);opt$step(nn$tensor_vector(nn$grad(out$loss,p$parameters())))
  expect_equal(opt$iterations(),1);expect_false(identical(p$flatten(),before))
})

test_that("Owned host summaries preserve signed finite-mixture estimates", {
  t <- nn$DType$Float64;mu <- nn$zeros(c(8L,4L),t);lv <- nn$zeros(c(8L,4L),t)
  out <- nn$CollapseStatistics(mu,lv,nn$tensor_vector(list(mu)))
  unit <- out$unit_kl;variance <- out$mu_variance;rm(out);gc()
  expect_equal(unit,rep(0,4));expect_equal(variance,rep(0,4))
  out <- nn$CollapseStatistics(mu,nn$full(c(8L,4L),log(2),t),nn$tensor_vector(list(mu)))
  expect_gt(out$analytic_kl,0);expect_lt(out$aggregate_kl,0);expect_lt(out$sampled_conditional_kl,0)
  expect_equal(out$active_units,0)
  expect_equal(out$sampled_conditional_kl,out$mixture_information+out$aggregate_kl)
})

test_that("Invalid sentence IDs and uniforms fail before leaves", {
  p <- nn$ParameterStore(1L)
  expect_error(nn$SequenceVariationalAutoencoder(p,nn$zeros(c(2L,7L)),nn$zeros(c(2L,4L))))
  expect_error(nn$SequenceVariationalAutoencoder(p,nn$full(c(2L,7L),19,nn$DType$Int32),nn$zeros(c(2L,4L))))
  expect_error(nn$SequenceGenerate(p,nn$zeros(c(2L,4L)),TRUE,nn$zeros(c(2L,7L),nn$DType$Float64)))
  expect_equal(p$size(),0);expect_error(nn$CollapseBeta(TRUE,-1L))
})

test_that("Native midpoint uniforms preserve the exact local unsigned32 stream", {
  raw <- c(3499211612,581869302,3890346734,3586334585,545404204,4161255391)
  u <- nn$MidpointUniformNoise(c(3L,2L),5489)
  expect_equal(u$values(),(raw+.5)/4294967296,tolerance=0)
  expect_equal(u$dtype(),nn$DType$Float64);expect_false(u$requires_grad())
  expect_equal(nn$MidpointUniformNoise(6L,5489)$values(),u$values(),tolerance=0)
  for(seed in c(0,2147483648,4294967295)){
    v <- nn$MidpointUniformNoise(c(20L,7L),seed)
    expect_true(all(v$values()>0 & v$values()<1))
    expect_equal(v$values(),nn$MidpointUniformNoise(c(20L,7L),seed)$values(),tolerance=0)
  }
  expect_equal(nn$MidpointUniformNoise(c(2L,0L,3L),0)$numel(),0)
  expect_length(nn$MidpointUniformNoise(integer(),5489)$shape(),0)
  expect_error(nn$MidpointUniformNoise(c(0L,-1L),1))
})
