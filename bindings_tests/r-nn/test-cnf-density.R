library(testthat)
nn<-datamungerNN::nn
keep<-function(){g<-nn$GradOptions();g$retain_graph<-TRUE;g$create_graph<-TRUE;g}
points<-function()nn$from_vector_double(c(.4,-.7,1.2,.3,-1,2),c(3L,2L))
test_that("field trace both deterministic probes and owning higher graphs match",{
 p<-nn$ParameterStore(10003,nn$DType$Float64);x<-points()$requires_grad_();f<-nn$CnfField(p,x,.5)
 a<-nn$grad(nn$sum(nn$slice_axis(f$velocity,1L,0L,1L)),list(x),keep())[[1]];b<-nn$grad(nn$sum(nn$slice_axis(f$velocity,1L,1L,2L)),list(x),keep())[[1]]
 exact<-nn$add(nn$select(a,1L,0L),nn$select(b,1L,1L));plus<-nn$sum(nn$add(a,b),c(1L))
 minus<-nn$add(nn$sub(nn$sub(nn$select(a,1L,0L),nn$select(a,1L,1L)),nn$select(b,1L,0L)),nn$select(b,1L,1L))
 expect_lt(nn$max(nn$abs(nn$sub(exact,f$exact_trace)))$item(),1e-16);expect_lt(nn$max(nn$abs(nn$sub(plus,f$hutchinson_plus)))$item(),1e-16);expect_lt(nn$max(nn$abs(nn$sub(minus,f$hutchinson_minus)))$item(),1e-16)
 expect_gt(nn$max(nn$abs(nn$sub(plus,exact)))$item(),1e-4)
 values<-list(f$exact_trace,f$hutchinson_plus,f$hutchinson_minus,nn$mul(nn$add(f$hutchinson_plus,f$hutchinson_minus),.5))
 for(mode in 0:3)expect_true(nn$equal_bits(nn$CnfField(p,x,.5,mode)$divergence,values[[mode+1]]))
 trace<-f$exact_trace;velocity<-f$velocity;rm(f,p);gc()
 for(value in list(trace,velocity)){first<-nn$grad(nn$sum(value),list(x),keep())[[1]];expect_true(nn$grad(nn$sum(nn$square(first)),list(x),keep())[[1]]$defined())}
})
test_that("field time rounds to structural Float32 before Double tensor arithmetic",{
 p<-nn$ParameterStore(10003,nn$DType$Float64);x<-nn$zeros(c(1L,2L),nn$DType$Float64);nn$CnfField(p,x,0);expect_equal(p$count(),50);expect_equal(p$size(),4)
 wi<-rep(0,24);wi[17]<-1;wo<-rep(0,16);wo[1]<-1;p$set("field/hidden/W",nn$from_vector_double(wi,c(3L,8L)));p$set("field/output/W",nn$from_vector_double(wo,c(8L,2L)))
 p$set("field/hidden/b",nn$zeros(c(8L),nn$DType$Float64));p$set("field/output/b",nn$zeros(c(2L),nn$DType$Float64))
 time<-.1+1e-10;out<-nn$CnfField(p,x,time);rounded<-.10000000149011612
 expect_lt(abs(out$velocity$values()[1]-.3*tanh(rounded)),1e-16);expect_gt(abs(out$velocity$values()[1]-.3*tanh(time)),1e-12);expect_equal(out$divergence$item(),0)
})
test_that("density reverse path and quarter fields retain owning higher graphs",{
 p<-nn$ParameterStore(10003,nn$DType$Float64);x<-points()$requires_grad_();out<-nn$CnfDensity(p,x)
 expect_equal(p$count(),52);expect_equal(p$size(),5);expect_equal(out$stage_divergence2$shape(),c(3));expect_true(nn$equal_bits(out$stage3,out$unscaled))
 rec<-nn$CnfInverse(p,out$latent);expect_lt(nn$max(nn$abs(nn$sub(rec$value,x)))$item(),1e-7);expect_lt(nn$max(nn$abs(nn$add(rec$logdet,out$logdet)))$item(),1e-7)
 quarter<-out$stage1;cumulative<-out$stage_logdet2;endpoint<-out$stage_velocity3;inverse<-rec$value;rm(out,rec,p);gc()
 for(value in list(quarter,cumulative,endpoint,inverse)){first<-nn$grad(nn$sum(value),list(x),keep())[[1]];expect_true(nn$grad(nn$sum(nn$square(first)),list(x),keep())[[1]]$defined())}
})
test_that("host-owned Adam and reverse solver parameter gradients remain explicit",{
 p<-nn$ParameterStore(10003);x<-nn$from_vector(c(.4,-.7,1.2,.3,-1,2),c(3L,2L));nn$CnfDensity(p,x)
 config<-nn$OptimizerConfig();config$kind<-nn$OptimizerKind$Adam;optimizer<-nn$Optimizer(p,.003,config);before<-p$flatten();out<-nn$CnfDensity(p,x)
 gradients<-nn$grad(nn$NormalFlowDensity(out$latent,out$logdet)$loss,p$parameters());optimizer$step(nn$tensor_vector(gradients));expect_equal(optimizer$iterations(),1);expect_false(identical(p$flatten(),before))
 inv<-nn$CnfInverse(p,x);objective<-nn$add(nn$mean(nn$sum(nn$square(inv$value),c(1L)),c(0L)),nn$mul(nn$mean(inv$logdet,c(0L)),.3));expect_equal(length(nn$grad(objective,p$parameters())),5)
})
test_that("step counts reverse checkpoints and cumulative trace fields retain source semantics",{
 p<-nn$ParameterStore(10003,nn$DType$Float64);x<-points()
 for(inverse in c(FALSE,TRUE))for(steps in c(4L,8L,12L,16L)){
  out<-nn$CnfIntegrate(p,x,inverse,steps);expect_true(nn$equal_bits(out$latent,out$stage3));expect_true(nn$equal_bits(out$logdet,out$stage_logdet3));expect_equal(out$stage_velocity1$shape(),c(3,2));expect_gt(nn$max(nn$abs(nn$sub(out$stage_logdet3,out$stage_logdet0)))$item(),1e-5)
 }
 expect_equal(p$count(),50);expect_equal(p$size(),4)
})
test_that("invalid geometry time modes steps and unsafe scales reject",{
 p<-nn$ParameterStore(10003);for(bad in list(nn$zeros(c(2L)),nn$zeros(c(1L,3L)),nn$zeros(c(0L,2L)),nn$full(c(1L,2L),NaN)))expect_error(nn$CnfDensity(p,bad))
 x<-nn$zeros(c(1L,2L));for(steps in c(0L,2L,3L,5L,6L,10L)){expect_error(nn$CnfDensity(p,x,steps));expect_error(nn$CnfInverse(p,x,steps))}
 for(mode in c(-1L,4L)){expect_error(nn$CnfField(p,x,.5,mode));expect_error(nn$CnfIntegrate(p,x,FALSE,8L,mode))}
 expect_error(nn$CnfField(p,x,NaN));expect_equal(p$size(),0);nn$CnfDensity(p,x);p$set("log_scale",nn$full(c(2L),1000));expect_error(nn$CnfDensity(p,x));expect_error(nn$CnfInverse(p,x))
})
