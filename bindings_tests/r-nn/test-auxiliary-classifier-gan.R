library(testthat)
nn <- datamungerNN::nn

test_that("Stable extreme raw logits retain small real and strong generator gradients", {
  for(t in c(nn$DType$Float32,nn$DType$Float64)) {
    real <- nn$full(c(2L,1L),80,t)$requires_grad_();fake <- nn$full(c(3L,1L),-80,t)$requires_grad_()
    out <- nn$LogisticGanObjectives(real,fake);d <- out$discriminator;g <- out$generator;rm(out);gc()
    expect_equal(g$item(),80)
    derivatives <- nn$grad(d,list(real,fake))
    expect_true(all(derivatives[[1]]$values()<0));expect_equal(derivatives[[1]]$values()/(-.5*exp(-80)),rep(1,2),tolerance=1e-6)
    expect_true(all(derivatives[[2]]$values()>0));expect_equal(derivatives[[2]]$values()/(exp(-80)/3),rep(1,3),tolerance=1e-6)
    expect_equal(nn$grad(g,list(fake))[[1]]$values(),rep(-1/3,3),tolerance=1e-7)
  }
})

test_that("Default census and class targets never feed the image-only discriminator", {
  g <- nn$ParameterStore(5209L);d <- nn$ParameterStore(5219L)
  x <- nn$full(c(2L,784L),.3);z <- nn$full(c(2L,16L),-.2);labels <- nn$zeros(c(2L),nn$DType$Int32)
  a <- nn$AuxiliaryClassifierGan(g,d,x,z,labels);b <- nn$AuxiliaryClassifierGan(g,d,x,z,nn$ones(c(2L),nn$DType$Int64))
  expect_equal(c(g$count(),d$count()),c(26736,25483));expect_equal(c(g$size(),d$size()),c(4,6))
  expect_equal(d$names(),c("hidden/W","hidden/b","class/W","class/b","source/W","source/b"))
  expect_equal(a$real_logit$values(),b$real_logit$values(),tolerance=0)
  expect_equal(a$real_class_probability$values(),b$real_class_probability$values(),tolerance=0)
  expect_false(identical(a$fake$values(),b$fake$values()));expect_false(identical(a$real_class_loss$item(),b$real_class_loss$item()))
  expect_equal(nn$ConditionalGanGenerate(g,z,labels)$values(),a$fake$values(),tolerance=0)
  out <- nn$AuxiliaryGanDiscriminate(d,x);source <- out$source;logits <- out$class_logits;probability <- out$class_probability;rm(out,d);gc()
  expect_equal(source$values(),a$real_logit$values(),tolerance=0);expect_equal(logits$shape(),c(2,10));expect_true(all(is.finite(probability$values())))
})

test_that("All fifteen live fields own lifetime after collection of parent and stores", {
  t <- nn$DType$Float64;g <- nn$ParameterStore(5209L,t);d <- nn$ParameterStore(5219L,t)
  x <- nn$full(c(2L,3L),.3,t)$requires_grad_();z <- nn$full(c(2L,2L),-.2,t)$requires_grad_()
  out <- nn$AuxiliaryClassifierGan(g,d,x,z,nn$zeros(c(2L),nn$DType$Int64),3L,2L,2L)
  names <- c("fake","real_logit","fake_logit","real_probability","fake_probability","discriminator_loss","generator_loss","real_class_logits","fake_class_logits","real_class_probability","fake_class_probability","real_class_loss","fake_class_loss","source_discriminator_loss","source_generator_loss")
  fields <- lapply(names,function(name)do.call("$",list(out,name)));names(fields) <- names;rm(out,g,d);gc()
  expect_length(fields,15);expect_equal(fields$fake$shape(),c(2,3));expect_true(all(vapply(fields,function(a)all(is.finite(a$values())),logical(1))))
  expect_true(any(abs(nn$grad(fields$generator_loss,list(z))[[1]]$values())>1e-12))
})

test_that("R owns two independent Adam states and recomputes D then G", {
  t <- nn$DType$Float64;stores <- lapply(c(5209L,5219L),function(seed)nn$ParameterStore(seed,t))
  x <- nn$full(c(2L,3L),.3,t);z <- nn$full(c(2L,2L),-.2,t);labels <- nn$zeros(c(2L),nn$DType$Int32)
  forward <- function()nn$AuxiliaryClassifierGan(stores[[1]],stores[[2]],x,z,labels,3L,2L,2L)
  forward();config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam;config$beta1 <- .5
  opts <- lapply(stores,function(p)nn$Optimizer(p,.0002,config))
  for(group in c(2L,1L)) {
    before <- lapply(stores,function(p)p$flatten());out <- forward();loss <- if(group==1L)out$generator_loss else out$discriminator_loss
    opts[[group]]$step(nn$tensor_vector(nn$grad(loss,stores[[group]]$parameters())))
    expect_equal(vapply(seq_along(stores),function(k)!identical(stores[[k]]$flatten(),before[[k]]),logical(1)),seq_along(stores)==group)
  }
  expect_equal(vapply(opts,function(opt)opt$iterations(),numeric(1)),c(1,1))
})

test_that("Coarse pooling and malformed targets retain canonical behavior", {
  t <- nn$DType$Float64;x <- nn$ones(c(2L,784L),t)$requires_grad_();coarse <- nn$GanCoarsePixels(x)
  expect_equal(coarse$shape(),c(2,16));expect_equal(coarse$values(),rep(.25,32),tolerance=1e-14)
  expect_equal(nn$grad(nn$sum(coarse),list(x))[[1]]$values(),rep(1/196,1568),tolerance=1e-14)
  g <- nn$ParameterStore(1L);d <- nn$ParameterStore(2L);x <- nn$zeros(c(2L,3L));z <- nn$zeros(c(2L,2L));labels <- nn$zeros(c(2L),nn$DType$Int32)
  for(bad in list(nn$zeros(c(2L)),nn$zeros(c(1L),nn$DType$Int32),nn$full(c(2L),-1,nn$DType$Int64),nn$full(c(2L),2,nn$DType$Int32))) {
    expect_error(nn$AuxiliaryClassifierGan(g,d,x,z,bad,3L,2L,2L));expect_error(nn$ConditionalGanGenerate(g,z,bad,3L,2L,2L));expect_error(nn$AuxiliaryClassLoss(nn$zeros(c(2L,2L)),bad))
  }
  expect_error(nn$AuxiliaryClassifierGan(g,g,x,z,labels,3L,2L,2L));expect_error(nn$AuxiliaryClassifierGan(g,d,x,nn$zeros(c(1L,2L)),labels,3L,2L,2L))
  expect_error(nn$AuxiliaryClassifierGan(g,d,nn$full(c(2L,3L),NaN),z,labels,3L,2L,2L));expect_error(nn$AuxiliaryClassifierGan(g,d,x,z,labels,3L,2L,0L))
  expect_error(nn$LogisticGanObjectives(nn$zeros(c(2L,1L)),nn$zeros(c(2L,2L))));expect_error(nn$GanCoarsePixels(nn$zeros(c(2L,783L))))
  expect_equal(g$size()+d$size(),0)
})
