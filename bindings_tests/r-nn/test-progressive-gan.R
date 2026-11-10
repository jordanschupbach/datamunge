library(testthat)
nn <- datamungerNN::nn

pg_inputs <- function(){t <- nn$DType$Float64;list(p=lapply(c(6401L,6407L,6413L,6421L,6427L,6449L,6451L,6469L),function(seed)nn$ParameterStore(seed,t)),x=nn$full(c(2L,8L,8L,1L),.3,t),z=nn$GaussianRejectionPrior(2L,16L,6504L,2,t)$z,a=nn$full(integer(),.5,t))}
pg_forward <- function(d,a=d$a)do.call(nn$ProgressiveGan,c(d$p,list(d$x,d$z,a)))

test_that("Pixel RMS is channel-wise division with live higher paths", {
 x <- nn$from_vector_double(c(3,4,0,0),c(2L,2L))$requires_grad_();out <- nn$pixel_rms_normalize(x)
 expect_equal(out$values(),c(.8485281371,1.1313708499,0,0),tolerance=1e-8)
 options <- nn$GradOptions();options$create_graph <- TRUE;gradient <- nn$grad(nn$sum(out),list(x),options)[[1]];expect_equal(nn$grad(nn$sum(nn$mul(gradient,gradient)),list(x))[[1]]$shape(),c(2,2))
 expect_error(nn$pixel_rms_normalize(x,0));expect_error(nn$pixel_rms_normalize(nn$full(c(2L,2L),NaN)))
})

test_that("Native activity is an R integer vector and both source fade forms work", {
 expect_identical(nn$ProgressiveActiveGroups(0L),c(0L,1L));expect_identical(nn$ProgressiveActiveGroups(1L),0:3);expect_identical(nn$ProgressiveActiveGroups(2L,FALSE),c(4L,6L,7L));expect_error(nn$ProgressiveActiveGroups(3L))
 d <- pg_inputs();out <- pg_forward(d);other <- pg_forward(d,nn$full(1L,.5,nn$DType$Float64));expect_identical(out$fake$values(),other$fake$values());expect_equal(vapply(d$p,function(p)p$count(),numeric(1)),c(1236,5,296,5,213,8,296,8));expect_equal(sum(vapply(d$p,function(p)p$size(),numeric(1))),24)
})

test_that("Real old/new generator images and D features have exact immutable endpoints", {
 d <- pg_inputs();pg_forward(d);before <- lapply(d$p,function(p)p$flatten());old <- do.call(nn$ProgressiveGanGenerate,c(d$p[1:4],list(d$z,nn$full(integer(),0,nn$DType$Float64))));high <- do.call(nn$ProgressiveGanGenerate,c(d$p[1:4],list(d$z,nn$full(integer(),1,nn$DType$Float64))))
 expect_identical(old$image$values(),nn$upsample_nearest(old$old4,2L)$values());expect_identical(high$image$values(),high$new8$values())
 low_d <- do.call(nn$ProgressiveGanDiscriminate,c(d$p[5:8],list(d$x,nn$full(integer(),0,nn$DType$Float64))));high_d <- do.call(nn$ProgressiveGanDiscriminate,c(d$p[5:8],list(d$x,nn$full(integer(),1,nn$DType$Float64))))
 expect_identical(low_d$mixed$values(),low_d$old_path$values());expect_identical(high_d$mixed$values(),high_d$new_path$values());expect_identical(lapply(d$p,function(p)p$flatten()),before)
})

test_that("Each phase captures all active gradients before any selected Adam write", {
 d <- pg_inputs();pg_forward(d);cfg <- nn$OptimizerConfig();cfg$kind <- nn$OptimizerKind$Adam;cfg$beta1 <- .5;optim <- lapply(d$p,function(p)nn$Optimizer(p,.0002,cfg));options <- nn$GradOptions();options$retain_graph <- TRUE;options$allow_unused <- TRUE
 for(generator in c(FALSE,TRUE)){groups <- nn$ProgressiveActiveGroups(1L,generator);out <- pg_forward(d);loss <- if(generator)out$g_loss else out$d_loss;captured <- lapply(groups,function(k)nn$grad(loss,d$p[[k+1L]]$parameters(),options));before <- lapply(d$p,function(p)p$flatten());for(i in seq_along(groups))optim[[groups[i]+1L]]$step(nn$tensor_vector(captured[[i]]));after <- lapply(d$p,function(p)p$flatten());expect_identical(before[-(groups+1L)],after[-(groups+1L)])}
 expect_equal(vapply(optim,function(o)o$iterations(),numeric(1)),rep(1,8))
})

test_that("Owned records retain graphs and source G loss has no half factor", {
 d <- pg_inputs();z <- d$z$requires_grad_();a <- d$a$requires_grad_();out <- pg_forward(d);features <- do.call(nn$ProgressiveGanDiscriminate,c(d$p[5:8],list(d$x,a)));fake <- out$fake;loss <- out$g_loss;mixed <- features$mixed;rm(out,features,d);gc();expect_equal(fake$shape(),c(2,8,8,1));expect_equal(mixed$shape(),c(2,4,4,4));partials <- nn$grad(loss,list(z,a));expect_true(any(abs(partials[[1]]$values())>1e-12));expect_gt(abs(partials[[2]]$item()),1e-12)
 d <- pg_inputs();expect_error(pg_forward(d,nn$full(2L,.5,nn$DType$Float64)));expect_error(pg_forward(d,nn$full(integer(),1.1,nn$DType$Float64)));expect_equal(sum(vapply(d$p,function(p)p$size(),numeric(1))),0)
 pg_forward(d);for(p in d$p[5:8])for(name in p$names())p$set(name,nn$zeros(p$at(name)$shape(),nn$DType$Float64));out <- pg_forward(d);expect_equal(out$d_loss$item(),.5);expect_equal(out$g_loss$item(),1)
})
