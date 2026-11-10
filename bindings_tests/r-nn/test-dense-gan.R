library(testthat)
nn <- datamungerNN::nn

test_that("Raw least-squares and hinge losses retain pool means and zero margin subgradients", {
 t <- nn$DType$Float64
 real <- nn$from_vector_double(c(0,1,2),c(3L,1L))$requires_grad_();fake <- nn$from_vector_double(c(-2,-1,0,1),c(4L,1L))$requires_grad_()
 ls <- nn$LeastSquaresGanObjectives(real,fake)
 expect_equal(ls$discriminator$item(),.5*(2/3+1.5));expect_equal(ls$generator$item(),1.75)
 expect_equal(nn$grad(ls$discriminator,list(real))[[1]]$values(),c(-1/3,0,1/3))
 h <- nn$HingeGanObjectives(real,fake);expect_equal(h$discriminator$item(),1/3+.75);expect_equal(h$generator$item(),.5)
 expect_equal(nn$grad(h$discriminator,list(real,fake))[[1]]$values(),c(-1/3,0,0))
 expect_equal(nn$grad(nn$HingeGanObjectives(real,fake)$discriminator,list(fake))[[1]]$values(),c(0,0,.25,.25))
 expect_equal(nn$grad(nn$HingeGanObjectives(real,fake)$generator,list(fake))[[1]]$values(),rep(-.25,4))
})

test_that("Three native objectives share maps and own five endpoints with unequal pools", {
 for(kind in c("Logistic","LeastSquares","Hinge")) {
  objective <- nn$DenseGanObjective[[kind]];g <- nn$ParameterStore(4801L);d <- nn$ParameterStore(4807L)
  x <- nn$full(c(2L,784L),.3);z <- nn$full(c(3L,16L),-.2);out <- nn$DenseGan(g,d,x,z,objective)
  expect_equal(c(g$count(),d$count(),g$size(),d$size()),c(26416,25153,4,4))
  expect_equal(nn$MlpGanGenerate(g,z)$values(),out$fake$values(),tolerance=0)
  expect_equal(nn$MlpGanDiscriminate(d,x)$values(),out$real_score$values(),tolerance=0)
  names <- c("fake","real_score","fake_score","discriminator_loss","generator_loss")
  fields <- lapply(names,function(name)do.call("$",list(out,name)));rm(out,g,d);gc()
  expect_equal(fields[[1]]$shape(),c(3,784));expect_true(all(vapply(fields,function(a)all(is.finite(a$values())),logical(1))))
 }
})

test_that("Four host-owned loops select D then recompute through updated D for G", {
 for(kind in c("Logistic","LeastSquares","Hinge","Conditional")) {
  t <- nn$DType$Float64;g <- nn$ParameterStore(5101L,t);d <- nn$ParameterStore(5113L,t)
  x <- nn$full(c(2L,3L),.3,t);z <- nn$full(c(2L,2L),-.2,t)$requires_grad_();labels <- nn$from_vector_int64(c(0,1),c(2L))
  forward <- function(){if(kind=="Conditional")nn$ConditionalGan(g,d,x,z,labels,3L,2L,3L) else nn$DenseGan(g,d,x,z,nn$DenseGanObjective[[kind]],3L,3L)}
  expect_true(any(abs(nn$grad(forward()$generator_loss,list(z))[[1]]$values())>1e-12))
  config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam;config$beta1 <- .5
  stores <- list(g,d);opts <- lapply(stores,function(p)nn$Optimizer(p,.0002,config))
  old_g <- lapply(nn$grad(forward()$generator_loss,g$parameters()),function(v)v$values())
  for(group in c(2L,1L)) {
   before <- lapply(stores,function(p)p$flatten())
   if(group==1L)expect_false(identical(old_g,lapply(nn$grad(forward()$generator_loss,g$parameters()),function(v)v$values())))
   out <- forward();loss <- if(group==2L)out$discriminator_loss else out$generator_loss
   opts[[group]]$step(nn$tensor_vector(nn$grad(loss,stores[[group]]$parameters())))
   expect_equal(vapply(seq_along(stores),function(k)!identical(stores[[k]]$flatten(),before[[k]]),logical(1)),seq_along(stores)==group)
  }
  expect_equal(vapply(opts,function(o)o$iterations(),numeric(1)),c(1,1))
 }
})

test_that("Both networks receive conditions and seven fields outlive stores", {
 g <- nn$ParameterStore(5101L);d <- nn$ParameterStore(5113L);x <- nn$full(c(2L,784L),.3);z <- nn$full(c(2L,16L),-.2)
 labels <- nn$from_vector_int64(c(0,1),c(2L));wrong <- nn$from_vector_int64(c(1,2),c(2L))
 out <- nn$ConditionalGan(g,d,x,z,labels);expect_equal(c(g$count(),d$count()),c(26736,25473))
 expect_false(identical(nn$ConditionalGanGenerate(g,z,wrong)$values(),out$fake$values()))
 expect_false(identical(nn$ConditionalGanDiscriminate(d,x,wrong)$values(),out$real_logit$values()))
 names <- c("fake","real_logit","fake_logit","real_probability","fake_probability","discriminator_loss","generator_loss")
 fields <- lapply(names,function(name)do.call("$",list(out,name)));rm(out,g,d);gc()
 expect_length(fields,7);expect_true(all(vapply(fields,function(a)all(is.finite(a$values())),logical(1))))
})

test_that("Invalid stores options and labels fail before parameter creation", {
 g <- nn$ParameterStore(1L);d <- nn$ParameterStore(2L);x <- nn$zeros(c(2L,3L));z <- nn$zeros(c(2L,2L))
 for(bad in list(nn$from_vector_int64(c(-1,0),c(2L)),nn$from_vector_int64(c(0,2),c(2L)),nn$zeros(c(1L),nn$DType$Int32),nn$zeros(c(2L))))expect_error(nn$ConditionalGan(g,d,x,z,bad,3L,2L,3L))
 expect_error(nn$DenseGan(g,g,x,z,nn$DenseGanObjective$Logistic,3L,3L));expect_error(nn$DenseGan(g,d,x,z,999L,3L,3L));expect_error(nn$MlpGanDiscriminate(d,x,3L,0L))
 expect_error(nn$LeastSquaresGanObjectives(nn$zeros(c(0L,1L)),nn$zeros(c(2L,1L))));expect_equal(g$size()+d$size(),0)
})
