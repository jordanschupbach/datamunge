library(testthat)
nn <- datamungerNN::nn

test_that("Fused affine BN captures are detached, owned and do not commit themselves", {
 t <- nn$DType$Float64;p <- nn$ParameterStore(0L,t);s <- nn$ParameterStore(0L,t)
 x <- nn$from_vector_double(c(1,3),c(2L,1L))$requires_grad_();out <- nn$capture_affine_batch_norm(p,s,"bn",x,TRUE)
 expect_equal(out$mean$item(),2);expect_equal(out$unbiased_variance$item(),2);expect_false(out$mean$requires_grad());expect_false(out$unbiased_variance$requires_grad());expect_equal(s$at("bn/moving_mean")$item(),0)
 next_values <- nn$batch_norm_moving_values(s$at("bn/moving_mean"),s$at("bn/moving_var"),out$mean,out$unbiased_variance,.9)
 expect_equal(next_values$mean$item(),.2);expect_equal(next_values$variance$item(),1.1)
 value <- out$value;mean <- out$mean;variance <- out$unbiased_variance;rm(out,p,s);gc()
 expect_equal(mean$item(),2);expect_equal(variance$item(),2);expect_length(nn$grad(nn$sum(nn$mul(value,value)),list(x))[[1]]$values(),2)
})

test_that("Saved-BN DCGAN census and signed-to-unit conversion are explicit", {
 stores <- lapply(c(5003L,5011L,0L,0L),nn$ParameterStore);x <- nn$full(c(2L,28L,28L,1L),.3);z <- nn$GaussianRejectionPrior(2L,16L,4862L)$z
 forward <- function()nn$Dcgan(stores[[1]],stores[[2]],stores[[3]],stores[[4]],x,z)
 out <- forward();expect_equal(vapply(stores,function(p)p$count(),numeric(1)),c(6873,989,24,16));expect_equal(vapply(stores,function(p)p$size(),numeric(1)),c(8,7,4,2))
 before <- lapply(stores,function(p)p$flatten());generated <- nn$DcganGenerate(stores[[1]],stores[[3]],z)
 expect_false(generated$stem_mean$defined());expect_identical(out$fake$values(),nn$reshape(nn$mul(nn$add(generated$image,1),.5),c(2L,784L))$values());expect_identical(lapply(stores,function(p)p$flatten()),before)
})

test_that("D has stopped batch G and G uses current saved D with available weight paths", {
 t <- nn$DType$Float64;stores <- lapply(c(5003L,5011L,0L,0L),function(seed)nn$ParameterStore(seed,t))
 x <- nn$full(c(2L,28L,28L,1L),.3,t);z <- nn$GaussianRejectionPrior(2L,16L,4862L,2,t)$z$requires_grad_()
 a <- nn$DcganDiscriminatorPhase(stores[[1]],stores[[2]],stores[[3]],stores[[4]],x,z);b <- nn$DcganGeneratorPhase(stores[[1]],stores[[2]],stores[[3]],stores[[4]],x,z)
 expect_false(a$fake$requires_grad());expect_true(b$fake$requires_grad());expect_false(a$down7_mean$requires_grad());expect_false(b$real_logit$defined());expect_false(b$down7_mean$defined())
 partials <- nn$grad(b$loss,c(list(z),nn$tensor_list(stores[[2]]$parameters())));expect_true(any(abs(partials[[1]]$values())>1e-10));expect_length(partials,8)
 direct <- nn$DcganDiscriminate(stores[[2]],stores[[4]],b$fake);expect_identical(b$fake_logit$values(),direct$logit$values());expect_equal(b$loss$item(),nn$mean(nn$softplus(nn$neg(direct$logit)))$item(),tolerance=0)
})

test_that("Four explicit owners commit captured BN before selected Adam writes", {
 stores <- lapply(c(5003L,5011L,0L,0L),nn$ParameterStore);g <- stores[[1]];d <- stores[[2]];gs <- stores[[3]];ds <- stores[[4]]
 x <- nn$full(c(2L,28L,28L,1L),.3);z <- nn$GaussianRejectionPrior(2L,16L,4862L)$z;nn$Dcgan(g,d,gs,ds,x,z)
 cfg <- nn$OptimizerConfig();cfg$kind <- nn$OptimizerKind$Adam;cfg$beta1 <- .5;go <- nn$Optimizer(g,.0002,cfg);disc <- nn$Optimizer(d,.0002,cfg)
 commit <- function(s,name,mean,variance){v <- nn$batch_norm_moving_values(s$at(paste0(name,"/moving_mean")),s$at(paste0(name,"/moving_var")),mean,variance,.9);s$set(paste0(name,"/moving_mean"),v$mean);s$set(paste0(name,"/moving_var"),v$variance)}
 a <- nn$DcganDiscriminatorPhase(g,d,gs,ds,x,z)
 actions <- list(function()commit(ds,"down7_bn",a$down7_mean,a$down7_variance),function()disc$step(nn$tensor_vector(nn$grad(a$loss,d$parameters()))))
 for(i in 1:2){owner <- c(4,2)[i];before <- lapply(stores,function(p)p$flatten());actions[[i]]();after <- lapply(stores,function(p)p$flatten());expect_false(identical(before[[owner]],after[[owner]]));expect_identical(before[-owner],after[-owner])}
 b <- nn$DcganGeneratorPhase(g,d,gs,ds,x,z);actions <- list(function(){commit(gs,"stem_bn",b$stem_mean,b$stem_variance);commit(gs,"up14_bn",b$up14_mean,b$up14_variance)},function()go$step(nn$tensor_vector(nn$grad(b$loss,g$parameters()))))
 for(i in 1:2){owner <- c(3,1)[i];before <- lapply(stores,function(p)p$flatten());actions[[i]]();after <- lapply(stores,function(p)p$flatten());expect_false(identical(before[[owner]],after[[owner]]));expect_identical(before[-owner],after[-owner])}
 expect_equal(go$iterations(),1);expect_equal(disc$iterations(),1)
})

test_that("Flat records retain lifetime and invalid phase inputs precede leaves", {
 t <- nn$DType$Float64;stores <- lapply(c(5003L,5011L,0L,0L),function(seed)nn$ParameterStore(seed,t));x <- nn$full(c(2L,28L,28L,1L),.3,t);z <- nn$GaussianRejectionPrior(2L,16L,4862L,2,t)$z$requires_grad_()
 out <- nn$DcganGeneratorPhase(stores[[1]],stores[[2]],stores[[3]],stores[[4]],x,z);fake <- out$fake;loss <- out$loss;mean <- out$stem_mean;rm(out,stores);gc();expect_equal(fake$shape(),c(2,28,28,1));expect_equal(mean$shape(),8);expect_false(mean$requires_grad());expect_true(any(abs(nn$grad(loss,list(z))[[1]]$values())>1e-10))
 stores <- lapply(c(5003L,5011L,0L,0L),nn$ParameterStore);x <- nn$zeros(c(2L,28L,28L,1L));z <- nn$ones(c(2L,16L))
 expect_error(nn$Dcgan(stores[[1]],stores[[1]],stores[[3]],stores[[4]],x,z));expect_error(nn$Dcgan(stores[[1]],stores[[2]],stores[[3]],stores[[4]],x,nn$ones(c(1L,16L))));expect_error(nn$DcganGenerate(stores[[1]],stores[[3]],nn$full(c(2L,16L),NaN)));expect_equal(sum(vapply(stores,function(p)p$size(),numeric(1))),0)
})
