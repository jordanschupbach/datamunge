library(testthat)
nn <- datamungerNN::nn

test_that("Ordered ring centres seeds and final cast are explicit", {
  t <- nn$DType$Float64;centre <- nn$GaussianRingPoints(4L,5507L,4L,2,0,t)
  expect_equal(centre$values(),c(2,0,0,2,-2,0,0,-2),tolerance=1e-14)
  first <- nn$GaussianRingPoints(19L,5507L)
  expect_equal(first$values(),nn$GaussianRingPoints(19L,5507L)$values(),tolerance=0)
  expect_false(identical(first$values(),nn$GaussianRingPoints(19L,5519L)$values()))
  widened <- nn$GaussianRingPoints(19L,5507L,8L,2,.08,t)
  expect_equal(first$values(),nn$cast(widened,nn$DType$Float32)$values(),tolerance=0)
  expect_false(first$requires_grad())
  expect_error(nn$GaussianRingPoints(0L));expect_error(nn$GaussianRingPoints(3L,1L,0L))
  expect_error(nn$GaussianRingPoints(3L,1L,8L,-1));expect_error(nn$GaussianRingPoints(3L,1L,8L,2,Inf))
  expect_error(nn$GaussianRingPoints(3L,1L,8L,NaN));expect_error(nn$GaussianRingPoints(3L,1L,8L,2,.08,nn$DType$Int32))
  expect_error(nn$GaussianRingPoints(3L,1L,1L,1e300,0))
})

test_that("Default census unequal pools and both explicit pair parts match", {
  g <- nn$ParameterStore(6007L);e <- nn$ParameterStore(6011L);d <- nn$ParameterStore(6029L)
  x <- nn$full(c(2L,2L),.3);z <- nn$full(c(3L,2L),-.2);out <- nn$BidirectionalGan(g,e,d,x,z)
  expect_equal(c(g$count(),e$count(),d$count()),c(82,82,97));expect_equal(c(g$size(),e$size(),d$size()),c(4,4,4))
  expect_equal(out$fake$shape(),c(3,2));expect_equal(out$encoded$shape(),c(2,2))
  expect_equal(nn$BiganMap(g,z)$values(),out$fake$values(),tolerance=0);expect_equal(nn$BiganMap(e,x)$values(),out$encoded$values(),tolerance=0)
  pair <- nn$BiganJointScore(d,x,out$encoded)
  expect_equal(pair$values(),out$real_score$values(),tolerance=0)
  expect_false(identical(nn$BiganJointScore(d,x,nn$zeros(c(2L,2L)))$values(),pair$values()))
  expect_false(identical(nn$BiganJointScore(d,nn$zeros(c(2L,2L)),out$encoded)$values(),pair$values()))
})

test_that("Eight fields survive parent and store collection with live continuous paths", {
  t <- nn$DType$Float64;g <- nn$ParameterStore(6007L,t);e <- nn$ParameterStore(6011L,t);d <- nn$ParameterStore(6029L,t)
  x <- nn$full(c(2L,3L),-.3,t)$requires_grad_();z <- nn$full(c(4L,2L),.2,t)$requires_grad_()
  out <- nn$BidirectionalGan(g,e,d,x,z,3L)
  names <- c("fake","encoded","real_score","fake_score","discriminator_loss","eg_loss","reconstruction","latent_cycle")
  fields <- lapply(names,function(name)do.call("$",list(out,name)));names(fields) <- names;rm(out,g,e,d);gc()
  expect_length(fields,8);expect_equal(fields$reconstruction$shape(),c(2,3));expect_equal(fields$latent_cycle$shape(),c(4,2))
  expect_true(all(vapply(fields,function(a)all(is.finite(a$values())),logical(1))))
  derivatives <- nn$grad(fields$eg_loss,list(x,z))
  expect_true(all(vapply(derivatives,function(a)any(abs(a$values())>1e-12),logical(1))))
})

test_that("Three separate Adam states write D then E then G and preserve E-independent G partial", {
  t <- nn$DType$Float64;stores <- lapply(c(6007L,6011L,6029L),function(seed)nn$ParameterStore(seed,t))
  x <- nn$full(c(2L,3L),.3,t);z <- nn$full(c(3L,2L),-.2,t)
  forward <- function()nn$BidirectionalGan(stores[[1]],stores[[2]],stores[[3]],x,z,3L)
  forward();config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam;config$beta1 <- .5
  opts <- lapply(stores,function(p)nn$Optimizer(p,.0002,config));old_g <- NULL
  for(group in c(3L,2L,1L)) {
    before <- lapply(stores,function(p)p$flatten())
    if(group==2L)old_g <- lapply(nn$grad(forward()$eg_loss,stores[[1]]$parameters()),function(a)a$values())
    if(group==1L)expect_identical(lapply(nn$grad(forward()$eg_loss,stores[[1]]$parameters()),function(a)a$values()),old_g)
    out <- forward();loss <- if(group==3L)out$discriminator_loss else out$eg_loss
    opts[[group]]$step(nn$tensor_vector(nn$grad(loss,stores[[group]]$parameters())))
    expect_equal(vapply(seq_along(stores),function(k)!identical(stores[[k]]$flatten(),before[[k]]),logical(1)),seq_along(stores)==group)
  }
  expect_equal(vapply(opts,function(opt)opt$iterations(),numeric(1)),c(1,1,1))
})

test_that("Malformed coordinates stores and dimensions fail before leaves", {
  t <- nn$DType$Float64;g <- nn$ParameterStore(1L,t);e <- nn$ParameterStore(2L,t);d <- nn$ParameterStore(3L,t)
  x <- nn$zeros(c(2L,3L),t);z <- nn$zeros(c(3L,2L),t)
  for(bad in list(nn$zeros(c(0L,3L),t),nn$zeros(c(2L,0L),t),nn$zeros(c(2L,3L)),nn$full(c(2L,3L),NaN,t)))expect_error(nn$BidirectionalGan(g,e,d,bad,z,2L))
  for(bad in list(nn$zeros(c(0L,2L),t),nn$zeros(c(3L,0L),t),nn$zeros(c(3L,2L)),nn$full(c(3L,2L),Inf,t)))expect_error(nn$BidirectionalGan(g,e,d,x,bad,2L))
  expect_error(nn$BidirectionalGan(g,g,d,x,z,2L));expect_error(nn$BidirectionalGan(g,e,g,x,z,2L));expect_error(nn$BidirectionalGan(g,e,d,x,z,0L))
  expect_error(nn$BiganJointScore(d,x,z,2L));expect_error(nn$BiganMap(g,z,0L,2L));expect_equal(g$size()+e$size()+d$size(),0)
})
