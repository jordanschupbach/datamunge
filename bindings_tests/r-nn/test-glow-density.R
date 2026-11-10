library(testthat)
nn<-datamungerNN::nn
test_that("ActNorm moments detach and parameter installation stays explicit",{
 x<-nn$from_vector(c(1,2,3,6),c(2L,2L))$requires_grad_();fit<-nn$FlowActNormCalibration(x)
 expect_equal(fit$mean$values(),c(2,4));expect_equal(fit$variance$values(),c(1,4));expect_equal(fit$bias$values(),c(-2,-4))
 expect_equal(fit$mean$dtype(),nn$DType$Float64);expect_equal(fit$log_scale$dtype(),nn$DType$Float32)
 for(field in list(fit$mean,fit$variance,fit$bias,fit$log_scale))expect_false(field$requires_grad())
 p<-nn$ParameterStore(9503);expect_true(nn$equal_bits(nn$FlowActNorm(p,"norm",x)$value,x))
 bias<-fit$bias;scale<-fit$log_scale;rm(fit);gc();p$set("norm/bias",bias);p$set("norm/log_scale",scale);before<-p$flatten()
 o<-nn$FlowActNorm(p,"norm",x);rec<-nn$FlowActNorm(p,"norm",o$value,TRUE)
 expect_lt(nn$max(nn$abs(nn$sub(rec$value,x)))$item(),1e-6);expect_equal(nn$add(rec$logdet,o$logdet)$item(),0)
 nn$FlowActNorm(p,"norm",nn$full(c(2L,2L),100));expect_identical(p$flatten(),before)
})
test_that("negative mixing determinants use log absolute values and retain higher paths",{
 p<-nn$ParameterStore(1,nn$DType$Float64);p$set("mix/W",nn$from_vector_double(c(1,2,3,4),c(2L,2L)))
 x<-nn$from_vector_double(c(.2,-.7,1.1,.4,-.3,2),c(3L,2L))$requires_grad_()
 o<-nn$FlowMix2D(p,"mix",x);rec<-nn$FlowMix2D(p,"mix",o$value,.35,TRUE)
 expect_equal(o$logdet$item(),log(2),tolerance=1e-14);expect_equal(nn$add(o$logdet,rec$logdet)$item(),0)
 expect_lt(nn$max(nn$abs(nn$sub(rec$value,x)))$item(),1e-14)
 graph<-nn$GradOptions();graph$retain_graph<-TRUE;graph$create_graph<-TRUE
 loss<-nn$sub(nn$sum(nn$square(o$value)),o$logdet);gradient<-nn$grad(loss,list(p$at("mix/W")),graph)[[1]]
 expect_true(nn$grad(nn$sum(gradient),list(x),graph)[[1]]$defined())
 value<-o$value;rm(o,rec,p);gc();expect_true(nn$grad(nn$sum(value),list(x))[[1]]$defined())
})
test_that("flat Glow stage fields retain higher graphs and immutable inference",{
 p<-nn$ParameterStore(9503,nn$DType$Float64)
 x<-nn$from_vector_double(c(.4,-.7,1.2,.3,-1,2),c(3L,2L))$requires_grad_()
 path<-nn$GlowDensity(p,x);expect_equal(p$count(),296);expect_equal(p$size(),28);expect_equal(path$logdet$shape(),c(3))
 before<-p$flatten();rec<-nn$GlowInverse(p,path$latent)
 expect_lt(nn$max(nn$abs(nn$sub(rec$value,x)))$item(),1e-14)
 expect_lt(nn$max(nn$abs(nn$add(rec$logdet,path$logdet)))$item(),1e-14)
 density<-nn$NormalFlowDensity(path$latent,path$logdet,TRUE)
 norm<-path$actnorm_output1;mix<-path$mixed2;scale<-path$coupling_scale2;shift<-path$coupling_shift3;loss<-density$loss
 rm(path,density,rec);gc();graph<-nn$GradOptions();graph$retain_graph<-TRUE;graph$create_graph<-TRUE
 first<-nn$grad(loss,list(x),graph)[[1]];expect_true(nn$grad(nn$sum(first),list(x),graph)[[1]]$defined())
 expect_identical(p$flatten(),before);rm(p);gc()
 for(field in list(norm,mix,scale,shift))expect_true(nn$grad(nn$sum(field),list(x),graph)[[1]]$defined())
})
test_that("invalid calibration names matrices and scales reject before initial leaves",{
 for(epsilon in c(0,-1,Inf,NaN))expect_error(nn$FlowActNormCalibration(nn$zeros(c(2L,2L)),epsilon))
 p<-nn$ParameterStore(9503)
 for(bad in list(nn$zeros(c(2L)),nn$zeros(c(1L,3L)),nn$zeros(c(0L,2L)),nn$full(c(1L,2L),NaN))){
  expect_error(nn$GlowDensity(p,bad));expect_error(nn$GlowInverse(p,bad));expect_equal(p$size(),0)
 }
 expect_error(nn$FlowActNorm(p,"",nn$zeros(c(2L,2L))))
 expect_error(nn$FlowMix2D(p,"mix",nn$zeros(c(2L,2L)),Inf));expect_equal(p$size(),0)
 nn$GlowDensity(p,nn$zeros(c(1L,2L)));p$set("mix0/W",nn$zeros(c(2L,2L)))
 expect_error(nn$GlowDensity(p,nn$zeros(c(1L,2L))));expect_error(nn$GlowInverse(p,nn$zeros(c(1L,2L))))
 p$set("mix0/W",nn$from_vector(c(1,0,0,1),c(2L,2L)));p$set("actnorm0/log_scale",nn$full(c(2L),1000))
 expect_error(nn$GlowDensity(p,nn$zeros(c(1L,2L))));expect_error(nn$GlowInverse(p,nn$zeros(c(1L,2L))))
})
