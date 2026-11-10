library(testthat)
nn <- datamungerNN::nn

test_that("Captured BN and moving candidates own fields without implicit writes", {
 t <- nn$DType$Float64;stats <- nn$ParameterStore(0L,t);x <- nn$from_vector_double(c(1,3),c(2L,1L))$requires_grad_()
 out <- nn$capture_batch_norm(stats,"bn",x,TRUE);expect_equal(out$mean$values(),2);expect_equal(out$unbiased_variance$values(),2);expect_equal(stats$at("bn/moving_mean")$values(),0)
 value <- out$value;mean <- out$mean;variance <- out$unbiased_variance;loss <- nn$add(nn$sum(value),nn$sum(mean))
 candidate <- nn$batch_norm_moving_values(stats$at("bn/moving_mean"),stats$at("bn/moving_var"),mean,variance,.5);cm <- candidate$mean;cv <- candidate$variance
 rm(out,candidate,stats);gc();expect_equal(cm$item(),1);expect_equal(cv$item(),1.5);expect_false(cm$requires_grad());expect_false(cv$requires_grad());expect_equal(nn$grad(loss,list(x))[[1]]$values(),c(.5,.5))
 expect_error(nn$capture_batch_norm(nn$ParameterStore(0L),"bn",nn$ones(c(1L,2L)),TRUE))
})

test_that("Rejection prior preserves exact tested candidates and final casting", {
 a <- nn$GaussianRejectionPrior(1L,3L,6907L,.5);wide <- nn$GaussianRejectionPrior(1L,3L,6907L,.5,nn$DType$Float64)
 expect_equal(a$z$values(),c(-0.4454278349876404,-0.08479340374469757,0.38326919078826904),tolerance=0)
 expect_equal(a$proposals$values(),c(-0.9388149930815035,-0.44542782166946016,-1.2078747239081584,-0.08479340404496108,0.3832691843572766),tolerance=0)
 expect_equal(a$z$values(),nn$cast(wide$z,nn$DType$Float32)$values(),tolerance=0)
 z <- a$z;proposals <- a$proposals;rm(a,wide);gc();expect_false(z$requires_grad());expect_false(proposals$requires_grad())
 expect_error(nn$GaussianRejectionPrior(1L,1L,1L,1e-300));expect_error(nn$GaussianRejectionPrior(0L,3L,1L));expect_error(nn$GaussianRejectionPrior(1L,3L,1L,0))
})

test_that("BigGAN census and display switch preserve batch-path objectives", {
 stores <- lapply(c(6803L,6823L,0L,0L,0L),nn$ParameterStore)
 x <- nn$full(c(2L,8L,8L,1L),.3);z <- nn$GaussianRejectionPrior(2L,12L,6872L)$z;c <- nn$from_vector_int64(c(0,1),2L)
 forward <- function(batch=FALSE)nn$BigGan(stores[[1]],stores[[2]],stores[[3]],stores[[4]],stores[[5]],x,z,c,batch)
 forward();expect_equal(vapply(stores,function(p)p$count(),numeric(1)),c(789,241,241,77,24));expect_equal(vapply(stores,function(p)p$size(),numeric(1)),c(17,9,18,10,6))
 nn$advance_cached_spectral_state(stores[[1]],stores[[3]],16L);nn$advance_cached_spectral_state(stores[[2]],stores[[4]],16L)
 before <- lapply(stores,function(p)p$flatten());a <- forward(FALSE);b <- forward(TRUE);expect_identical(lapply(stores,function(p)p$flatten()),before)
 expect_identical(a$fake$values(),a$fake_inference$values());expect_identical(b$fake$values(),b$fake_training$values());expect_false(identical(a$fake$values(),b$fake$values()))
 expect_equal(a$g_loss$item(),b$g_loss$item(),tolerance=0);expect_equal(a$d_loss$item(),b$d_loss$item(),tolerance=0);expect_equal(a$moments$bn1_mean$shape(),4)
 inference <- nn$BigGanGenerate(stores[[1]],stores[[3]],stores[[5]],z,c,FALSE);expect_false(inference$moments$bn1_mean$defined())
})

test_that("Five explicit writes include a captured pre-Adam BN commit", {
 stores <- lapply(c(6803L,6823L,0L,0L,0L),nn$ParameterStore);g <- stores[[1]];d <- stores[[2]];gs <- stores[[3]];ds <- stores[[4]];moving <- stores[[5]]
 x <- nn$full(c(2L,8L,8L,1L),.3);z <- nn$GaussianRejectionPrior(2L,12L,6872L)$z;c <- nn$from_vector_int64(c(0,1),2L)
 forward <- function()nn$BigGan(g,d,gs,ds,moving,x,z,c,TRUE)
 forward();nn$advance_cached_spectral_state(g,gs,16L);nn$advance_cached_spectral_state(d,ds,16L)
 config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam;config$beta1 <- 0;config$beta2 <- .999
 go <- nn$Optimizer(g,.0002,config);disc <- nn$Optimizer(d,.0002,config);captured <- NULL
 g_write <- function(){out <- forward();captured <<- out$moments;go$step(nn$tensor_vector(nn$grad(out$g_loss,g$parameters())))}
 actions <- list(function()nn$advance_cached_spectral_state(d,ds),function()disc$step(nn$tensor_vector(nn$grad(forward()$d_loss,d$parameters()))),function()nn$advance_cached_spectral_state(g,gs),g_write,function()nn$BigGanCommitMoments(moving,captured))
 for(i in seq_along(actions)){owner <- c(4,2,3,1,5)[i];before <- lapply(stores,function(p)p$flatten());actions[[i]]();after <- lapply(stores,function(p)p$flatten());expect_false(identical(before[[owner]],after[[owner]]));expect_identical(before[-owner],after[-owner])}
 expect_equal(go$iterations(),1);expect_equal(disc$iterations(),1)
 before <- moving$flatten();captured$output_unbiased_variance <- nn$full(4L,NaN);expect_error(nn$BigGanCommitMoments(moving,captured));expect_identical(moving$flatten(),before)
})

test_that("Nested captures and image/projection fields retain graphs after parent collection", {
 t <- nn$DType$Float64;stores <- lapply(c(6803L,6823L,0L,0L,0L),function(seed)nn$ParameterStore(seed,t))
 x <- nn$full(c(2L,8L,8L,1L),.3,t);z <- nn$GaussianRejectionPrior(2L,12L,6872L,2,t)$z$requires_grad_();c <- nn$from_vector_int64(c(0,1),2L)
 nn$BigGan(stores[[1]],stores[[2]],stores[[3]],stores[[4]],stores[[5]],x,z,c)
 nn$advance_cached_spectral_state(stores[[1]],stores[[3]],16L);nn$advance_cached_spectral_state(stores[[2]],stores[[4]],16L)
 out <- nn$BigGan(stores[[1]],stores[[2]],stores[[3]],stores[[4]],stores[[5]],x,z,c,TRUE)
 generated <- nn$BigGanGenerate(stores[[1]],stores[[3]],stores[[5]],z,c,TRUE);discriminated <- nn$BigGanDiscriminate(stores[[2]],stores[[4]],x,c)
 capture <- out$moments;image <- generated$image;projection <- discriminated$projection;loss <- nn$add(out$g_loss,nn$sum(capture$bn1_mean))
 rm(out,generated,discriminated,stores);gc();expect_equal(image$shape(),c(2,8,8,1));expect_equal(projection$shape(),c(2,1));expect_equal(capture$bn1_mean$shape(),4);expect_true(any(abs(nn$grad(loss,list(z))[[1]]$values())>1e-10))
})

test_that("Orthogonal, resize and early input checks expose canonical primitives", {
 t <- nn$DType$Float64;p <- nn$ParameterStore(1L,t);p$set("layer/W",nn$ones(c(2L,2L),t));p$set("bias",nn$zeros(2L,t))
 penalty <- nn$ModifiedOrthogonalPenalty(p,.5);expect_equal(penalty$item(),4);partials <- nn$grad(penalty,p$parameters());expect_equal(partials[[1]]$values(),c(4,4,4,4));expect_equal(partials[[2]]$values(),c(0,0))
 expect_equal(nn$BigGanResize8(nn$ones(c(2L,28L,28L,1L)))$shape(),c(2,8,8,1))
 g <- nn$ParameterStore(1L);d <- nn$ParameterStore(2L);gs <- nn$ParameterStore(0L);ds <- nn$ParameterStore(0L);moving <- nn$ParameterStore(0L)
 x <- nn$zeros(c(2L,8L,8L,1L));z <- nn$zeros(c(2L,12L));c <- nn$from_vector_int64(c(0,1),2L)
 expect_error(nn$BigGan(g,g,gs,ds,moving,x,z,c));expect_error(nn$BigGan(g,d,gs,ds,moving,x,nn$zeros(c(3L,12L)),c));expect_error(nn$BigGan(g,d,gs,ds,moving,x,z,nn$from_vector_int64(c(0,10),2L)))
 expect_equal(g$size()+d$size()+gs$size()+ds$size()+moving$size(),0)
})
