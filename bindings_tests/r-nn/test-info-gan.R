library(testthat)
nn <- datamungerNN::nn

test_that("Generated prior owns exact accepted words and a final continuous cast", {
 a <- nn$InfoPrior(2L,1L,1073741825L,2L)
 expect_equal(a$raw$values(),c(1791095845,4282876139,3093770124,491263,550290313,1298508491),tolerance=0)
 expect_equal(a$category$values(),c(717354020,491263))
 first <- nn$InfoPrior(4L,5382L);wide <- nn$InfoPrior(4L,5382L,10L,2L,nn$DType$Float64)
 expect_equal(first$category$values(),c(0,9,1,1));expect_equal(first$continuous$values(),nn$cast(wide$continuous,nn$DType$Float32)$values(),tolerance=0)
 fields <- list(first$category,first$continuous,first$raw);rm(first);gc()
 expect_equal(lapply(fields,function(a)a$shape()),list(4,c(4,2),c(4,3)))
 expect_false(any(vapply(fields,function(a)a$requires_grad(),logical(1))))
 expect_error(nn$InfoPrior(0L,1L));expect_error(nn$InfoPrior(2L,1L,0L));expect_error(nn$InfoPrior(2L,1L,10L,0L));expect_error(nn$InfoPrior(2L,1L,10L,2L,nn$DType$Int32))
})

test_that("Recognition combines categorical CE and unit Gaussian NLL with negative bound", {
 q <- nn$InfoRecognition();q$logits <- nn$zeros(c(2L,2L),nn$DType$Float64)$requires_grad_();q$mean <- nn$zeros(c(2L,1L),nn$DType$Float64)$requires_grad_()
 c <- nn$from_vector_int64(c(0,1),2L);u <- nn$from_vector_double(c(-1,1),c(2L,1L))$requires_grad_()
 loss <- nn$InfoGanRecognitionLoss(q,c,u)
 expect_equal(loss$categorical$item(),log(2));expect_equal(loss$continuous$item(),(1+log(2*pi))/2)
 expect_equal(loss$lower_bound$item(),2*log(2)-loss$total$item());expect_lt(loss$lower_bound$item(),0)
 gradients <- nn$grad(loss$total,list(q$logits,q$mean,u))
 expect_equal(gradients[[1]]$values(),c(-.25,.25,.25,-.25));expect_equal(gradients[[2]]$values(),c(.5,-.5));expect_equal(gradients[[3]]$values(),c(-.5,.5))
 q$probability <- nn$full(3L,NaN);expect_equal(nn$InfoGanRecognitionLoss(q,c,u)$total$item(),loss$total$item(),tolerance=0)
})

test_that("Default parameter census unequal pools and image-only Q match standalone paths", {
 g <- nn$ParameterStore(5303L);d <- nn$ParameterStore(5309L);q <- nn$ParameterStore(5317L);codes <- nn$InfoPrior(3L,5351L)
 image <- nn$full(c(2L,784L),.3);noise <- nn$full(c(3L,16L),-.2)
 out <- nn$InfoGan(g,d,q,image,noise,codes$category,codes$continuous)
 expect_equal(c(g$count(),d$count(),q$count()),c(26800,25153,25516));expect_equal(c(g$size(),d$size(),q$size()),c(4,4,6))
 expect_equal(out$fake$shape(),c(3,784));expect_equal(out$real_logit$shape(),c(2,1))
 expect_equal(nn$InfoGanGenerate(g,noise,codes$category,codes$continuous)$values(),out$fake$values(),tolerance=0)
 expect_equal(nn$MlpGanDiscriminate(d,image)$values(),out$real_logit$values(),tolerance=0)
 prediction <- nn$InfoGanRecognize(q,out$fake);expect_equal(prediction$probability$values(),out$q_probability$values(),tolerance=0);expect_equal(prediction$mean$values(),out$q_mean$values(),tolerance=0)
 q$set("continuous/W",nn$zeros(c(32L,2L)));q$set("continuous/b",nn$full(2L,4))
 expect_equal(nn$InfoGanRecognize(q,image)$mean$values(),rep(4,4))
})

test_that("Fifteen fields retain continuous graphs after parent and store collection", {
 t <- nn$DType$Float64;g <- nn$ParameterStore(5303L,t);d <- nn$ParameterStore(5309L,t);q <- nn$ParameterStore(5317L,t)
 x <- nn$full(c(2L,3L),.3,t);z <- nn$full(c(3L,2L),-.2,t)$requires_grad_();u <- nn$full(c(3L,1L),.1,t)$requires_grad_();c <- nn$from_vector_int64(c(0,1,0),3L)
 out <- nn$InfoGan(g,d,q,x,z,c,u,3L,2L,3L)
 names <- c("fake","real_logit","fake_logit","real_probability","fake_probability","discriminator_loss","generator_loss","source_generator_loss","recognition_loss","categorical_loss","continuous_loss","mi_lower_bound","q_logits","q_probability","q_mean")
 fields <- lapply(names,function(name)do.call("$",list(out,name)));names(fields) <- names;rm(out,g,d,q);gc()
 expect_length(fields,15);expect_true(all(vapply(fields,function(a)all(is.finite(a$values())),logical(1))))
 derivatives <- nn$grad(fields$generator_loss,list(z,u));expect_true(all(vapply(derivatives,function(a)any(abs(a$values())>1e-12),logical(1))))
})

test_that("Host-owned D then Q then fresh G write only selected stores", {
 t <- nn$DType$Float64;stores <- lapply(c(5303L,5309L,5317L),function(seed)nn$ParameterStore(seed,t))
 x <- nn$full(c(2L,3L),.3,t);z <- nn$full(c(2L,2L),-.2,t);u <- nn$full(c(2L,1L),.1,t);c <- nn$from_vector_int64(c(0,1),2L)
 forward <- function()nn$InfoGan(stores[[1]],stores[[2]],stores[[3]],x,z,c,u,3L,2L,3L)
 forward();config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam;config$beta1 <- .5
 opts <- lapply(stores,function(p)nn$Optimizer(p,.0002,config))
 q_before <- lapply(nn$grad(forward()$recognition_loss,stores[[3]]$parameters()),function(a)a$values())
 g_before <- lapply(nn$grad(forward()$generator_loss,stores[[1]]$parameters()),function(a)a$values())
 for(group in c(2L,3L,1L)) {
  before <- lapply(stores,function(p)p$flatten())
  if(group==3L)expect_identical(q_before,lapply(nn$grad(forward()$recognition_loss,stores[[3]]$parameters()),function(a)a$values()))
  if(group==1L)expect_false(identical(g_before,lapply(nn$grad(forward()$generator_loss,stores[[1]]$parameters()),function(a)a$values())))
  out <- forward();loss <- if(group==1L)out$generator_loss else if(group==2L)out$discriminator_loss else out$recognition_loss
  opts[[group]]$step(nn$tensor_vector(nn$grad(loss,stores[[group]]$parameters())))
  expect_equal(vapply(seq_along(stores),function(k)!identical(stores[[k]]$flatten(),before[[k]]),logical(1)),seq_along(stores)==group)
 }
 expect_equal(vapply(opts,function(o)o$iterations(),numeric(1)),c(1,1,1))
})

test_that("Malformed inputs aliasing options and computed overflow reject", {
 t <- nn$DType$Float64;g <- nn$ParameterStore(1L,t);d <- nn$ParameterStore(2L,t);q <- nn$ParameterStore(3L,t)
 x <- nn$zeros(c(2L,3L),t);z <- nn$zeros(c(3L,2L),t);u <- nn$zeros(c(3L,1L),t);c <- nn$zeros(3L,nn$DType$Int32)
 for(bad in list(nn$zeros(c(2L,1L),t),nn$full(c(3L,1L),2,t),nn$full(c(3L,1L),NaN,t),nn$zeros(c(3L,1L))))expect_error(nn$InfoGan(g,d,q,x,z,c,bad,3L,2L,3L))
 for(bad in list(nn$from_vector_int64(c(-1,0,1),3L),nn$full(3L,2,nn$DType$Int32),nn$zeros(3L)))expect_error(nn$InfoGan(g,d,q,x,z,bad,u,3L,2L,3L))
 expect_error(nn$InfoGan(g,g,q,x,z,c,u,3L,2L,3L));expect_error(nn$InfoGan(g,d,g,x,z,c,u,3L,2L,3L));expect_error(nn$InfoGan(g,d,d,x,z,c,u,3L,2L,3L));expect_error(nn$InfoGan(g,d,q,x,z,c,u,3L,0L,3L))
 expect_equal(g$size()+d$size()+q$size(),0)
 prediction <- nn$InfoRecognition();prediction$logits <- nn$zeros(c(2L,2L));prediction$mean <- nn$full(c(2L,1L),1e30)
 expect_error(nn$InfoGanRecognitionLoss(prediction,nn$zeros(2L,nn$DType$Int32),nn$zeros(c(2L,1L))))
})
