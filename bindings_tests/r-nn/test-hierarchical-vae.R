library(testthat)
nn <- datamungerNN::nn

test_that("Residual posterior owns results and all conditional KL paths remain live", {
  t <- nn$DType$Float64;pm <- nn$full(c(2L,3L),.4,t)$requires_grad_();pl <- nn$full(c(2L,3L),-.2,t)$requires_grad_();zero <- nn$zeros(c(2L,3L),t)
  q <- nn$HierarchyPosterior(pm,pl,zero,zero);mu <- q$mu;lv <- q$log_variance;rm(q);gc()
  expect_equal(mu$values(),pm$values());expect_equal(lv$values(),pl$values())
  expect_equal(nn$ConditionalGaussianKl(mu,lv,pm,pl)$values(),c(0,0))
  qm <- nn$full(c(2L,3L),-.3,t)$requires_grad_();ql <- nn$full(c(2L,3L),.1,t)$requires_grad_()
  g <- nn$grad(nn$sum(nn$ConditionalGaussianKl(qm,ql,pm,pl)),nn$tensor_vector(list(qm,ql,pm,pl)))
  expect_length(g,4);expect_true(all(vapply(g,function(x)any(abs(x$values())>1e-12),logical(1))))
})

test_that("All nineteen fields own lifetime and both explicit groups decode", {
  t <- nn$DType$Float64
  for(nvae in c(FALSE,TRUE)){
    p <- nn$ParameterStore(2609L,t);state <- nn$ParameterStore(0L,t)
    out <- nn$HierarchicalVariationalAutoencoder(p,state,nn$full(c(2L,16L),.3,t),nn$ones(c(2L,2L),t),nvae,FALSE,FALSE,4L,4L,1L,1L)
    names <- c("encoder_feature","top_feature","top_mu","top_lv","top_z","prior_mu","prior_lv","head_mu","head_lv","lower_mu","lower_lv","lower_z","probability","nll","kl_lower","kl_top","loss","regularizer","train_loss")
    fields <- lapply(names,function(name)do.call("$",list(out,name)));names(fields) <- names;rm(out);gc()
    expect_equal(p$size(),if(nvae)37 else 26);expect_equal(p$count(),if(nvae)96 else 70)
    expect_equal(state$size(),if(nvae)6 else 0);expect_equal(fields$encoder_feature$shape(),c(2,1,1,1))
    expect_equal(fields$train_loss$item(),fields$loss$item()+fields$regularizer$item())
    expect_equal(nn$HierarchyDecode(p,state,fields$lower_z,fields$top_z,nvae,4L,4L,1L)$values(),fields$probability$values())
    gradients <- nn$grad(fields$train_loss,p$parameters());expect_length(gradients,p$size())
  }
})

test_that("Training moments, moving updates and host optimizer have separate controls", {
  t <- nn$DType$Float64;p <- nn$ParameterStore(2609L,t);state <- nn$ParameterStore(0L,t)
  x <- nn$full(c(2L,16L),.3,t);e <- nn$from_vector_double(c(.3,-.6,-.2,.8),c(2L,2L))
  nn$HierarchicalVariationalAutoencoder(p,state,x,e,TRUE,FALSE,FALSE,4L,4L,1L,1L)
  weights <- p$flatten();before <- state$flatten()
  probe <- nn$HierarchicalVariationalAutoencoder(p,state,x,e,TRUE,TRUE,FALSE,4L,4L,1L,1L)
  expect_identical(state$flatten(),before)
  update <- nn$HierarchicalVariationalAutoencoder(p,state,x,e,TRUE,TRUE,TRUE,4L,4L,1L,1L)
  expect_identical(p$flatten(),weights);expect_false(identical(state$flatten(),before))
  expect_equal(probe$probability$values(),update$probability$values());saved <- state$flatten()
  gradients <- nn$grad(update$train_loss,p$parameters());config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam
  optimizer <- nn$Optimizer(p,.003,config);optimizer$step(nn$tensor_vector(gradients))
  expect_equal(optimizer$iterations(),1);expect_false(identical(p$flatten(),weights));expect_identical(state$flatten(),saved)
})

test_that("Default census, ancestral results and malformed calls preserve native contracts", {
  for(nvae in c(FALSE,TRUE)){
    p <- nn$ParameterStore(2609L);state <- nn$ParameterStore(0L)
    nn$HierarchicalVariationalAutoencoder(p,state,nn$zeros(c(2L,784L)),nn$zeros(c(2L,51L)),nvae)
    expect_equal(p$count(),if(nvae)2027 else 1977);expect_equal(p$size(),if(nvae)37 else 26)
    weights <- p$flatten();stats <- state$flatten()
    prior <- nn$HierarchyPrior(p,state,nn$ones(c(2L,2L)),nn$zeros(c(2L,49L)),nvae)
    z <- prior$lower_z;pm <- prior$prior_mu;probability <- prior$probability;top <- prior$top_z;rm(prior);gc()
    expect_equal(z$values(),pm$values());expect_equal(nn$HierarchyDecode(p,state,z,top,nvae)$values(),probability$values())
    expect_identical(p$flatten(),weights);expect_identical(state$flatten(),stats)
  }
  p <- nn$ParameterStore(1L);state <- nn$ParameterStore(0L);x <- nn$zeros(c(2L,16L));e <- nn$zeros(c(2L,2L))
  expect_error(nn$HierarchicalVariationalAutoencoder(p,p,x,e,TRUE,FALSE,FALSE,4L,4L,1L,1L))
  expect_error(nn$HierarchicalVariationalAutoencoder(p,state,x,e,TRUE,FALSE,TRUE,4L,4L,1L,1L))
  expect_error(nn$HierarchicalVariationalAutoencoder(p,state,x,e,FALSE,FALSE,FALSE,5L,4L,1L,1L))
  expect_error(nn$HierarchyPrior(p,state,nn$zeros(c(2L,1L)),nn$zeros(c(1L,1L)),FALSE,4L,4L,1L))
  expect_equal(p$size(),0);expect_equal(state$size(),0)
})
