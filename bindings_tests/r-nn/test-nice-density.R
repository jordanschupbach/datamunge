library(testthat)
nn<-datamungerNN::nn
test_that("explicit banana coordinates detach and analytic density stays live",{
 base<-nn$from_vector_double(c(0,0,2,1,-1,3),c(3L,2L))$requires_grad_()
 points<-nn$BananaPoints(base)
 expect_equal(points$values(),c(0,-.5,2,2.5,-1,3));expect_false(points$requires_grad())
 logp<-nn$BananaLogProbability(base);expect_equal(logp$dtype(),nn$DType$Float64)
 expect_equal(logp$values()[1],-log(2*pi)-.125,tolerance=1e-14)
 expect_equal(nn$grad(nn$sum(logp),list(base))[[1]]$values()[2],-.5,tolerance=1e-14)
 expect_equal(nn$BananaPoints(base,-.5,nn$DType$Float64)$values(),c(0,.5,2,-.5,-1,3))
})
test_that("NICE flat fields retain graphs through parent and store collection",{
 p<-nn$ParameterStore(9301,nn$DType$Float64)
 x<-nn$from_vector_double(c(.4,-.7,1.2,.3,-1,2),c(3L,2L))$requires_grad_()
 path<-nn$NiceDensity(p,x);expect_equal(p$count(),198);expect_equal(p$size(),17)
 before<-p$flatten();rec<-nn$NiceInverse(p,path$latent)
 expect_lt(nn$max(nn$abs(nn$sub(rec$value,x)))$item(),1e-14)
 expect_equal(nn$add(rec$logdet,path$logdet)$item(),0)
 density<-nn$NormalFlowDensity(path$latent,path$logdet,TRUE)
 stage<-path$coupling2;loss<-density$loss;rm(path,density,rec);gc()
 graph<-nn$GradOptions();graph$retain_graph<-TRUE;graph$create_graph<-TRUE
 g<-nn$grad(loss,list(x),graph)[[1]];expect_equal(g$shape(),c(3,2))
 expect_true(nn$grad(nn$sum(g),list(x),graph)[[1]]$defined());expect_identical(p$flatten(),before)
 rm(p);gc();expect_true(nn$grad(nn$sum(stage),list(x))[[1]]$defined())
})
test_that("both coupling orders invert and Gaussian constant rounding is explicit",{
 p<-nn$ParameterStore(9301);x<-nn$from_vector(c(.2,.7,-1,2),c(2L,2L))
 for(block in 0:3){
  y<-nn$NiceCoupling(p,x,block);cond<-as.integer(block%%2)
  expect_true(nn$equal_bits(nn$slice_axis(y,1L,cond,cond+1L),nn$slice_axis(x,1L,cond,cond+1L)))
  rec<-nn$NiceCoupling(p,y,block,TRUE);expect_lt(nn$max(nn$abs(nn$sub(rec,x)))$item(),3e-7)
 }
 z<-nn$zeros(c(1L,2L),nn$DType$Float64)
 full<-nn$StandardNormalLogProbability(z)$item();rounded<-nn$StandardNormalLogProbability(z,TRUE)$item()
 expect_equal(full,-log(2*pi),tolerance=1e-14);expect_false(full==rounded)
 expect_equal(rounded,-1.8378770351409912,tolerance=1e-15)
})
test_that("invalid matrices and scales reject without initial leaf creation",{
 p<-nn$ParameterStore(9301)
 for(bad in list(nn$zeros(c(2L)),nn$zeros(c(1L,3L)),nn$zeros(c(0L,2L)),nn$full(c(1L,2L),NaN))){
  expect_error(nn$NiceDensity(p,bad));expect_error(nn$NiceInverse(p,bad));expect_equal(p$size(),0)
 }
 expect_error(nn$NiceCoupling(p,nn$zeros(c(1L,2L)),4L))
 expect_error(nn$BananaPoints(nn$zeros(c(1L,2L)),Inf))
 expect_error(nn$NormalFlowDensity(nn$zeros(c(2L,2L)),nn$zeros(c(1L))))
 nn$NiceDensity(p,nn$zeros(c(1L,2L)))
 for(scale in c(1000,-1000,NaN)){
  p$set("log_scale",nn$full(c(2L),scale))
  expect_error(nn$NiceDensity(p,nn$zeros(c(1L,2L))));expect_error(nn$NiceInverse(p,nn$zeros(c(1L,2L))))
 }
})
