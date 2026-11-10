library(testthat)
nn<-datamungerNN::nn
times<-function()nn$from_vector_int64(c(0,31),c(2L))
state<-function(dtype=nn$DType$Float32)nn$cast(nn$reshape(nn$sub(nn$div(nn$arange(0,128),128),.5),c(2L,8L,8L,1L)),dtype)
keep<-function(){g<-nn$GradOptions();g$retain_graph<-TRUE;g$create_graph<-TRUE;g}
test_that("cosine tables and live repeated-index coefficient selection own their values",{
 s<-nn$CosineDiffusionSchedule();expect_equal(s$alpha_bar$shape(),c(32));expect_equal(tail(s$alpha_bar$values(),1),2.369970271136327e-6);expect_equal(s$posterior_variance$values()[1],0)
 table<-nn$from_vector_double(c(.1,.2,.3,.4),c(4L))$requires_grad_();out<-nn$DiffusionExtract(table,nn$from_vector_int64(c(3,0,3),c(3L)),2L,nn$DType$Float64);expect_equal(out$values(),c(.4,.1,.4))
 g<-nn$grad(nn$sum(nn$square(out)),list(table),keep())[[1]];expect_equal(g$values(),c(.2,0,0,1.6));expect_equal(nn$grad(nn$sum(g),list(table))[[1]]$values(),c(2,0,0,4));values<-s$beta;rm(s);gc();expect_equal(length(values$values()),32)
})
test_that("source Float32 time features and noising coefficients survive Double promotion",{
 f<-nn$DiffusionTimeEmbedding(times());d<-nn$DiffusionTimeEmbedding(times(),31,nn$DType$Float64);expect_true(nn$equal_bits(nn$cast(f,nn$DType$Float64),d));expect_false(d$requires_grad());expect_equal(d$values()[1:8],c(0,0,0,0,1,1,1,1));expect_gt(abs(d$values()[9]),1e-8)
 s<-nn$CosineDiffusionSchedule();t<-nn$from_vector_int64(c(1,17),c(2L));x<-nn$full(c(2L,3L),.25,nn$DType$Float64);e<-nn$full(c(2L,3L),-.1,nn$DType$Float64);y<-nn$DiffusionNoised(x,e,t,s)
 signal<-nn$cast(nn$DiffusionExtract(s$sqrt_alpha_bar,t,2L),nn$DType$Float64);sigma<-nn$cast(nn$DiffusionExtract(s$sqrt_one_minus_alpha_bar,t,2L),nn$DType$Float64);expect_true(nn$equal_bits(y,nn$add(nn$mul(signal,x),nn$mul(sigma,e))))
 full<-nn$add(nn$mul(nn$DiffusionExtract(s$sqrt_alpha_bar,t,2L,nn$DType$Float64),x),nn$mul(nn$DiffusionExtract(s$sqrt_one_minus_alpha_bar,t,2L,nn$DType$Float64),e));expect_gt(nn$max(nn$abs(nn$sub(full,y)))$item(),1e-10)
})
test_that("last reverse step ignores innovations and reports unclipped estimates",{
 s<-nn$CosineDiffusionSchedule();x<-nn$from_vector_double(c(-.25,.25,2,-2),c(2L,2L));e<-nn$zeros(c(2L,2L),nn$DType$Float64)
 a<-nn$DdpmReverse(x,times(),e,nn$ones(c(2L,2L),nn$DType$Float64),s);b<-nn$DdpmReverse(x,times(),e,nn$full(c(2L,2L),123,nn$DType$Float64),s)
 expect_equal(a$variance$values()[1],0);expect_true(nn$equal_bits(nn$slice_axis(a$value,0L,0L,1L),nn$slice_axis(b$value,0L,0L,1L)));expect_gt(a$x0_raw$values()[3],100);expect_equal(a$x0_clipped$values()[3:4],c(1,-1));expect_gt(nn$max(nn$abs(nn$sub(a$value,b$value)))$item(),1)
})
test_that("epsilon model census explicit Adam and owning hidden graphs remain native",{
 p<-nn$ParameterStore(11003,nn$DType$Float64);x<-state(nn$DType$Float64)$requires_grad_();o<-nn$DdpmEpsilon(p,x,times());expect_equal(p$count(),5504);expect_equal(p$size(),6);expect_equal(o$hidden1$shape(),c(2,32))
 before<-p$flatten();config<-nn$OptimizerConfig();config$kind<-nn$OptimizerKind$Adam;optimizer<-nn$Optimizer(p,.003,config);gradients<-nn$grad(nn$mean(nn$square(nn$sub(o$epsilon,.2))),p$parameters());optimizer$step(nn$tensor_vector(gradients));expect_equal(optimizer$iterations(),1);expect_false(identical(before,p$flatten()))
 o<-nn$DdpmEpsilon(p,x,times());fields<-list(o$epsilon,o$hidden1,o$hidden2);embedding<-o$time_embedding;rm(o,p,optimizer);gc()
 for(value in fields){first<-nn$grad(nn$sum(nn$square(value)),list(x),keep())[[1]];expect_true(nn$grad(nn$sum(first),list(x),keep())[[1]]$defined())};expect_false(embedding$requires_grad());expect_equal(embedding$shape(),c(2,8))
})
test_that("pure noising and reverse graphs survive parent collection",{
 s<-nn$CosineDiffusionSchedule();x<-nn$full(c(2L,3L),.2,nn$DType$Float64)$requires_grad_();e<-nn$full(c(2L,3L),.1,nn$DType$Float64)$requires_grad_();t<-nn$from_vector_int64(c(4,17),c(2L));v<-nn$DiffusionNoised(x,e,t,s);o<-nn$DdpmReverse(v,t,e,nn$ones(c(2L,3L),nn$DType$Float64),s);fields<-list(v,o$x0_raw,o$mean,o$value);rm(o,s);gc()
 for(value in fields){first<-nn$grad(nn$sum(nn$square(value)),list(x),keep())[[1]];expect_gt(nn$max(nn$abs(first))$item(),1e-5);expect_true(nn$grad(nn$sum(first),list(x),keep())[[1]]$defined())}
})
test_that("strides warm immutability and RNG state are preserved",{
 p<-nn$ParameterStore(11003);q<-nn$ParameterStore(11003);x<-nn$transpose(state(),c(0L,2L,1L,3L));a<-nn$DdpmEpsilon(p,x,times());nn$DdpmEpsilon(q,x,times());before<-p$flatten();b<-nn$DdpmEpsilon(p,x$contiguous(),times());expect_true(nn$equal_bits(a$epsilon,b$epsilon));expect_equal(p$flatten(),before);expect_true(nn$equal_bits(p$get("extra",c(7L),nn$init_normal(.1)),q$get("extra",c(7L),nn$init_normal(.1))))
})
test_that("invalid indices geometry options and unsafe posterior tables reject",{
 for(count in c(-1L,0L,1L))expect_error(nn$CosineDiffusionSchedule(count));for(offset in c(-1,NaN,1e300))expect_error(nn$CosineDiffusionSchedule(32L,offset))
 for(t in list(nn$zeros(c(2L)),nn$from_vector_int64(c(-1),c(1L)),nn$from_vector_int64(c(32),c(1L))))expect_error(nn$DiffusionExtract(nn$CosineDiffusionSchedule()$beta,t))
 p<-nn$ParameterStore(11003);for(x in list(nn$zeros(c(2L,64L)),nn$zeros(c(0L,8L,8L,1L)),nn$full(c(2L,8L,8L,1L),NaN),state(nn$DType$Float64)))expect_error(nn$DdpmEpsilon(p,x,times()));expect_equal(p$size(),0)
 s<-nn$CosineDiffusionSchedule();s$posterior_variance<-nn$full(c(32L),-1,nn$DType$Float64);expect_error(nn$DdpmReverse(nn$zeros(c(2L,3L)),times(),nn$zeros(c(2L,3L)),nn$zeros(c(2L,3L)),s));expect_error(nn$DdpmReverse(nn$zeros(c(2L,3L)),times(),nn$zeros(c(2L,3L)),nn$zeros(c(2L,3L)),nn$DiffusionScheduleOutput()));expect_error(nn$DiffusionTimeEmbedding(times(),0))
})
