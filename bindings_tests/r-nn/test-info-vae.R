library(testthat)
nn <- datamungerNN::nn

test_that("Signed unequal-count IMQ owns all seven endpoints and sample gradients", {
  t <- nn$DType$Float64;x <- nn$from_vector_double(c(0,1),c(2L,1L))$requires_grad_()
  y <- nn$from_vector_double(c(0,2,3),c(3L,1L))$requires_grad_()
  out <- nn$ImqMmd(x,y);fields <- list(out$xx,out$yy,out$xy,out$within_x,out$within_y,out$cross,out$value);rm(out);gc()
  expect_equal(fields[[3]]$shape(),c(2,3));expect_equal(fields[[4]]$item(),127/170)
  expect_equal(fields[[7]]$item(),fields[[4]]$item()+fields[[5]]$item()-2*fields[[6]]$item())
  expect_equal(nn$ImqMmd(x,x)$value$item(),-43/170)
  expect_equal(nn$ImqKernel(nn$zeros(c(2L,1L),t),nn$ones(c(3L,1L),t))$values(),rep(127/170,6))
  g <- nn$grad(fields[[7]],nn$tensor_vector(list(x,y)));expect_length(g,2)
  expect_true(all(vapply(g,function(a)any(abs(a$values())>1e-12),logical(1))))
})

test_that("InfoVAE coefficient endpoints and three objective gradients are exact", {
  t <- nn$DType$Float64;nll <- nn$from_vector_double(c(4,6),c(2L))$requires_grad_()
  kl <- nn$from_vector_double(c(2,4),c(2L))$requires_grad_();mmd <- nn$full(integer(),.25,t)$requires_grad_()
  loss <- nn$InfoVaeObjective(nll,kl,mmd)
  expect_equal(loss$item(),8.875);expect_equal(nn$InfoVaeObjective(nll,kl,mmd,1)$item(),7.5)
  expect_equal(nn$InfoVaeObjective(nll,kl,mmd,0,1)$item(),8)
  expect_equal(nn$InfoVaeObjective(nll,kl,nn$neg(mmd))$item(),4.125)
  g <- nn$grad(loss,nn$tensor_vector(list(nll,kl,mmd)))
  expect_equal(g[[1]]$values(),c(.5,.5));expect_equal(g[[2]]$values(),c(.25,.25));expect_equal(g[[3]]$item(),9.5)
})

test_that("All sixteen fields own lifetime and Float64 image noise and prior paths stay live", {
  t <- nn$DType$Float64
  for(alpha in c(.5,1)){
    p <- nn$ParameterStore(3901L,t);x <- nn$full(c(2L,6L),.3,t)$requires_grad_()
    e <- nn$full(c(2L,2L),.2,t)$requires_grad_();prior <- nn$full(c(3L,2L),-.4,t)$requires_grad_()
    out <- nn$InfoVariationalAutoencoder(p,x,e,prior,alpha,10,6L,2L,3L)
    names <- c("mu","log_variance","z","logits","probability","nll","kl","negative_elbo","mmd_xx","mmd_yy","mmd_xy","mmd_within_x","mmd_within_y","mmd_cross","mmd","loss")
    fields <- lapply(names,function(name)do.call("$",list(out,name)));names(fields) <- names;rm(out);gc()
    expect_equal(p$size(),10);expect_equal(p$count(),70);expect_equal(fields$mmd_xy$shape(),c(2,3))
    expect_equal(fields$loss$dtype(),t);before <- p$flatten()
    expect_equal(nn$sigmoid(nn$InfoVaeDecode(p,fields$z,6L,2L,3L))$values(),fields$probability$values())
    expect_identical(p$flatten(),before)
    g <- nn$grad(fields$loss,nn$tensor_vector(c(nn$tensor_list(p$parameters()),list(x,e,prior))))
    expect_length(g,13);expect_true(all(vapply(g,function(a)all(is.finite(a$values())),logical(1))))
    expect_true(all(vapply(tail(g,3),function(a)any(abs(a$values())>1e-12),logical(1))))
  }
})

test_that("Default census prior-independent decoding and caller Adam preserve native behaviour", {
  p <- nn$ParameterStore(3901L);x <- nn$full(c(2L,784L),.3);e <- nn$ones(c(2L,4L));prior <- nn$zeros(c(3L,4L))
  out <- nn$InfoVariationalAutoencoder(p,x,e,prior);other <- nn$InfoVariationalAutoencoder(p,x,e,nn$ones(c(4L,4L)))
  expect_equal(p$size(),10);expect_equal(p$count(),26104)
  expect_equal(out$probability$values(),other$probability$values());expect_false(identical(out$loss$item(),other$loss$item()))
  before <- p$flatten();g <- nn$grad(out$loss,p$parameters());config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam
  optimizer <- nn$Optimizer(p,.003,config);optimizer$step(nn$tensor_vector(g))
  expect_equal(optimizer$iterations(),1);expect_false(identical(p$flatten(),before))
  decoder <- nn$ParameterStore(1L);nn$InfoVaeDecode(decoder,nn$zeros(c(1L,4L)))
  expect_equal(decoder$size(),4);expect_equal(decoder$count(),13408)
})

test_that("Malformed controls samples and pixels fail before creation", {
  p <- nn$ParameterStore(1L);x <- nn$full(c(2L,6L),.3);e <- nn$ones(c(2L,2L));prior <- nn$zeros(c(3L,2L))
  expect_error(nn$InfoVariationalAutoencoder(p,x,e,prior,-.1,10,6L,2L,3L))
  expect_error(nn$InfoVariationalAutoencoder(p,x,e,prior,.5,.9,6L,2L,3L))
  expect_error(nn$InfoVariationalAutoencoder(p,x,e,nn$zeros(c(1L,2L)),.5,10,6L,2L,3L))
  expect_error(nn$InfoVariationalAutoencoder(p,nn$full(c(2L,6L),1.1),e,prior,.5,10,6L,2L,3L))
  expect_equal(p$size(),0);expect_error(nn$ImqMmd(nn$zeros(c(1L,2L)),prior))
  expect_error(nn$ImqKernel(e,nn$ones(c(2L,2L),nn$DType$Float64)))
})
