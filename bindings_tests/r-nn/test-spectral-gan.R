library(testthat)
nn <- datamungerNN::nn

test_that("Cached kernels own live fields with a signed sigma and stopped columns", {
 w <- nn$from_vector_double(2,c(1L,1L))$requires_grad_()
 one <- nn$ones(c(1L,1L),nn$DType$Float64)
 out <- nn$cached_spectral_kernel(w,one,one);weight <- out$weight;sigma <- out$sigma;rm(out);gc()
 expect_equal(weight$item(),1);expect_equal(sigma$item(),2)
 expect_equal(nn$grad(nn$sum(weight),list(w))[[1]]$item(),0)
 negative <- nn$cached_spectral_kernel(nn$from_vector_double(-2,c(1L,1L)),one,one,.5)
 expect_equal(negative$sigma$item(),-2);expect_equal(negative$weight$item(),-4)
 pair <- nn$cached_spectral_power_iteration(nn$zeros(c(2L,3L)),nn$ones(c(3L,1L)))
 u <- pair$u;v <- pair$v;rm(pair);gc()
 expect_equal(u$shape(),c(3,1));expect_equal(v$shape(),c(2,1))
 expect_false(u$requires_grad());expect_false(v$requires_grad());expect_equal(u$values(),c(0,0,0))
 expect_error(nn$cached_spectral_power_iteration(nn$ones(c(2L,3L)),nn$ones(c(3L,1L)),0L))
 expect_error(nn$cached_spectral_kernel(w,one,one,0))
})

test_that("Host controls state then D then fresh G with isolated writes", {
 stores <- lapply(c(5801L,5807L,0L),nn$ParameterStore);g <- stores[[1]];d <- stores[[2]];s <- stores[[3]]
 x <- nn$zeros(c(2L,784L));z <- nn$zeros(c(3L,16L))
 forward <- function()nn$SpectralGan(g,d,s,x,z)
 out <- forward();expect_equal(vapply(stores,function(p)p$count(),numeric(1)),c(26416,25153,849))
 expect_equal(out$fake$shape(),c(3,784));expect_equal(out$real_score$shape(),c(2,1))
 sb <- s$flatten();forward();expect_identical(s$flatten(),sb)
 nn$advance_cached_spectral_state(d,s,16L)
 config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam;config$beta1 <- .5
 go <- nn$Optimizer(g,.0002,config);disc <- nn$Optimizer(d,.0002,config)
 gb <- g$flatten();db <- d$flatten();nn$advance_cached_spectral_state(d,s);sb <- s$flatten()
 expect_identical(g$flatten(),gb);expect_identical(d$flatten(),db)
 first <- forward();disc$step(nn$tensor_vector(nn$grad(first$discriminator_loss,d$parameters())))
 expect_identical(g$flatten(),gb);expect_identical(s$flatten(),sb);expect_false(identical(d$flatten(),db))
 db <- d$flatten();fresh <- forward();go$step(nn$tensor_vector(nn$grad(fresh$generator_loss,g$parameters())))
 expect_identical(d$flatten(),db);expect_identical(s$flatten(),sb);expect_false(identical(g$flatten(),gb))
 expect_equal(go$iterations(),1);expect_equal(disc$iterations(),1)
 empty <- nn$ParameterStore(0L);expect_error(nn$advance_cached_spectral_state(d,empty))
 expect_identical(d$flatten(),db);expect_equal(empty$size(),0)
})

test_that("Five model fields retain storage and gradients after parent collection", {
 t <- nn$DType$Float64;g <- nn$ParameterStore(3L,t);d <- nn$ParameterStore(4L,t);s <- nn$ParameterStore(0L,t)
 x <- nn$full(c(2L,3L),.3,t);z <- nn$full(c(3L,2L),-.2,t)$requires_grad_()
 nn$SpectralGan(g,d,s,x,z,3L,2L);nn$advance_cached_spectral_state(d,s,16L)
 out <- nn$SpectralGan(g,d,s,x,z,3L,2L)
 fields <- lapply(c("fake","real_score","fake_score","discriminator_loss","generator_loss"),function(k)do.call("$",list(out,k)))
 rm(out,g,d,s);gc();expect_true(all(vapply(fields,function(a)all(is.finite(a$values())),logical(1))))
 expect_true(any(abs(nn$grad(fields[[5]],list(z))[[1]]$values())>1e-12))
})
