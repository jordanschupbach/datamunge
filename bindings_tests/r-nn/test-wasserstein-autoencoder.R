library(testthat)
nn <- datamungerNN::nn

test_that("Pixel sums and signed objective preserve both paths", {
  t <- nn$DType$Float64;x <- nn$full(c(2L,4L),.2,t)$requires_grad_();y <- nn$full(c(2L,4L),.7,t)$requires_grad_()
  cost <- nn$WaeSquaredPixelCost(x,y);expect_equal(cost$values(),c(1,1),tolerance=1e-14)
  g <- nn$grad(nn$mean(cost),nn$tensor_vector(list(x,y)))
  expect_equal(g[[1]]$values(),rep(-.5,8),tolerance=1e-14);expect_equal(g[[2]]$values(),rep(.5,8),tolerance=1e-14)
  c <- nn$from_vector_double(c(2,4),c(2L))$requires_grad_();mm <- nn$full(integer(),-.1,t)$requires_grad_()
  loss <- nn$WaeObjective(c,mm);expect_equal(loss$item(),2)
  g <- nn$grad(loss,nn$tensor_vector(list(c,mm)));expect_equal(g[[1]]$values(),c(.5,.5));expect_equal(g[[2]]$item(),10)
  expect_equal(nn$grad(nn$WaeObjective(c,mm,0),nn$tensor_vector(list(mm)))[[1]]$item(),0)
  expect_lt(nn$WaeObjective(nn$zeros(c(2L)),nn$full(integer(),-.1))$item(),0)
})

test_that("Default census code-only restore and caller-owned Adam agree", {
  p <- nn$ParameterStore(3701L);o <- nn$WassersteinAutoencoder(p,nn$full(c(2L,784L),.3),nn$zeros(c(3L,4L)))
  expect_equal(p$size(),8);expect_equal(p$count(),26036);expect_equal(o$code$shape(),c(2,4));expect_equal(o$mmd$xy$shape(),c(2,3))
  expect_equal(o$probability$values(),nn$sigmoid(nn$WaeDecode(p,o$code))$values(),tolerance=0)
  d <- nn$ParameterStore(1L);nn$WaeDecode(d,nn$zeros(c(20L,4L)));expect_equal(d$size(),4);expect_equal(d$count(),13408)
  for(name in c("decoder/W","decoder/b","output/W","output/b"))d$set(name,p$at(name))
  expect_equal(o$probability$values(),nn$sigmoid(nn$WaeDecode(d,o$code))$values(),tolerance=0)
  before <- p$flatten();cfg <- nn$OptimizerConfig();cfg$kind <- nn$OptimizerKind$Adam;adam <- nn$Optimizer(p,.003,cfg)
  adam$step(nn$tensor_vector(nn$grad(o$loss,p$parameters())));expect_equal(adam$iterations(),1);expect_false(identical(before,p$flatten()))
})

test_that("Nested owning MMD records survive collection with live Float64 paths", {
  t <- nn$DType$Float64;p <- nn$ParameterStore(3701L,t);x <- nn$transpose(nn$full(c(6L,2L),.3,t))$requires_grad_();prior <- nn$transpose(nn$full(c(2L,3L),-.2,t))$requires_grad_()
  expect_false(x$is_contiguous());expect_false(prior$is_contiguous());o <- nn$WassersteinAutoencoder(p,x,prior,10,6L,2L,3L)
  names <- c("code","logits","probability","squared_cost","reconstruction_loss","loss");fields <- lapply(names,function(name)do.call("$",list(o,name)));names(fields) <- names
  nested <- o$mmd;rm(o);gc();kn <- c("xx","yy","xy","within_x","within_y","cross","value");kernels <- lapply(kn,function(name)do.call("$",list(nested,name)));names(kernels) <- kn;rm(nested);gc()
  expect_equal(p$count(),62);expect_equal(kernels$xy$shape(),c(2,3));expect_true(all(vapply(c(fields,kernels),function(a)all(is.finite(a$values())),logical(1))))
  g <- nn$grad(fields$loss,nn$tensor_vector(c(nn$tensor_list(p$parameters()),list(x,prior))));expect_length(g,10)
  expect_true(all(vapply(g,function(a)all(is.finite(a$values())),logical(1))));expect_true(all(vapply(g[9:10],function(a)any(abs(a$values())>1e-12),logical(1))))
  expect_equal(fields$probability$values(),nn$sigmoid(nn$WaeDecode(p,fields$code,6L,2L,3L))$values(),tolerance=0)
})

test_that("Unequal prior samples change only regularization", {
  p <- nn$ParameterStore(3701L);x <- nn$full(c(3L,784L),.3);a <- nn$WassersteinAutoencoder(p,x,nn$zeros(c(4L,4L)));before <- p$flatten()
  b <- nn$WassersteinAutoencoder(p,x,nn$ones(c(5L,4L)));expect_equal(a$code$values(),b$code$values(),tolerance=0);expect_equal(a$probability$values(),b$probability$values(),tolerance=0);expect_false(identical(a$mmd$value$item(),b$mmd$value$item()))
  small <- nn$WassersteinAutoencoder(p,nn$take_rows(x,c(0L,1L)),nn$zeros(c(7L,4L)));expect_equal(small$probability$values(),nn$take_rows(a$probability,c(0L,1L))$values(),tolerance=0)
  z <- nn$WassersteinAutoencoder(p,x,nn$zeros(c(4L,4L)),0);expect_equal(z$loss$item(),z$reconstruction_loss$item());expect_identical(before,p$flatten())
})

test_that("Invalid counts controls values and widths fail before creation", {
  p <- nn$ParameterStore(1L);x <- nn$full(c(2L,6L),.3);prior <- nn$zeros(c(3L,2L))
  for(w in c(-1,Inf,NaN,1e100,1e-100))expect_error(nn$WassersteinAutoencoder(p,x,prior,w,6L,2L,3L))
  expect_error(nn$WassersteinAutoencoder(p,nn$zeros(c(1L,6L)),prior,0,6L,2L,3L));expect_error(nn$WassersteinAutoencoder(p,x,nn$zeros(c(1L,2L)),0,6L,2L,3L))
  expect_error(nn$WassersteinAutoencoder(p,nn$full(c(2L,6L),1.1),prior,10,6L,2L,3L));expect_error(nn$WaeDecode(p,nn$zeros(c(2L,3L)),6L,2L,3L));expect_equal(p$size(),0)
  expect_error(nn$WaeSquaredPixelCost(nn$zeros(c(2L,3L)),nn$full(c(2L,3L),-.1)))
  expect_error(nn$WaeObjective(nn$from_vector_double(c(-1,2),c(2L)),nn$full(integer(),.1,nn$DType$Float64)))
})
