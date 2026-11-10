library(testthat)
nn<-datamungerNN::nn
test_that("affine blocks keep raw conditioning coordinates and invert their own logdets",{
 p<-nn$ParameterStore(9403,nn$DType$Float64);x<-nn$from_vector_double(c(.4,-.7,1.2,.3,-1,2),c(3L,2L))
 for(block in 0:3){
  o<-nn$RealNvpCoupling(p,x,block);cond<-as.integer(block%%2)
  expect_true(nn$equal_bits(nn$slice_axis(o$value,1L,cond,cond+1L),nn$slice_axis(x,1L,cond,cond+1L)))
  expect_equal(o$scale$shape(),c(3,1));expect_equal(o$logdet$shape(),c(3))
  expect_lte(nn$max(nn$abs(o$scale))$item(),.8)
  expect_true(nn$equal_bits(nn$reshape(o$scale,c(3L)),o$logdet))
  rec<-nn$RealNvpCoupling(p,o$value,block,TRUE)
  expect_lt(nn$max(nn$abs(nn$sub(rec$value,x)))$item(),1e-14)
  expect_lt(nn$max(nn$abs(nn$add(rec$logdet,o$logdet)))$item(),1e-14)
 }
 expect_equal(p$count(),264);expect_equal(p$size(),16)
})
test_that("flat affine stage fields retain higher graphs through collection",{
 p<-nn$ParameterStore(9403,nn$DType$Float64)
 x<-nn$from_vector_double(c(.4,-.7,1.2,.3,-1,2),c(3L,2L))$requires_grad_()
 path<-nn$RealNvpDensity(p,x);expect_equal(p$count(),266);expect_equal(p$size(),17);expect_equal(path$logdet$shape(),c(3))
 expect_gt(nn$sub(nn$max(path$logdet),nn$min(path$logdet))$item(),1e-5)
 before<-p$flatten();rec<-nn$RealNvpInverse(p,path$latent)
 expect_lt(nn$max(nn$abs(nn$sub(rec$value,x)))$item(),1e-14)
 expect_lt(nn$max(nn$abs(nn$add(rec$logdet,path$logdet)))$item(),1e-14)
 density<-nn$NormalFlowDensity(path$latent,path$logdet,TRUE)
 stage<-path$coupling2;scale<-path$coupling_scale1;shift<-path$coupling_shift3;ld<-path$coupling_logdet0;loss<-density$loss
 rm(path,density,rec);gc();graph<-nn$GradOptions();graph$retain_graph<-TRUE;graph$create_graph<-TRUE
 first<-nn$grad(loss,list(x),graph)[[1]];expect_true(nn$grad(nn$sum(first),list(x),graph)[[1]]$defined())
 expect_identical(p$flatten(),before);rm(p);gc()
 for(field in list(stage,scale,shift,ld))expect_true(nn$grad(nn$sum(field),list(x),graph)[[1]]$defined())
})
test_that("global diagonal scale accepts three coordinates and scalar or per-row determinants",{
 p<-nn$ParameterStore(1,nn$DType$Float64)
 p$set("log_scale",nn$from_vector_double(c(log(2),log(.5),.2),c(3L)))
 x<-nn$ones(c(2L,3L),nn$DType$Float64)$requires_grad_();incoming<-nn$full(c(2L),.3,nn$DType$Float64)$requires_grad_()
 o<-nn$FlowDiagonalScale(p,x,incoming);before<-p$flatten()
 expect_equal(o$logdet$shape(),c(2));expect_equal(o$logdet$values()[1],.5,tolerance=1e-14)
 expect_equal(o$value$values()[1],2,tolerance=1e-14)
 rec<-nn$FlowDiagonalScale(p,o$value,o$logdet,TRUE)
 expect_lt(nn$max(nn$abs(nn$sub(rec$value,x)))$item(),1e-14)
 expect_lt(nn$max(nn$abs(nn$sub(rec$logdet,incoming)))$item(),1e-14)
 scalar<-nn$FlowDiagonalScale(p,x,nn$full(integer(),.3,nn$DType$Float64));expect_length(scalar$logdet$shape(),0)
 derivative<-nn$grad(nn$sum(o$logdet),list(incoming))[[1]]
 expect_equal(derivative$values(),c(1,1));expect_identical(p$flatten(),before)
})
test_that("invalid matrices determinants and scales reject without initial leaves",{
 p<-nn$ParameterStore(9403)
 for(bad in list(nn$zeros(c(2L)),nn$zeros(c(1L,3L)),nn$zeros(c(0L,2L)),nn$full(c(1L,2L),NaN))){
  expect_error(nn$RealNvpDensity(p,bad));expect_error(nn$RealNvpInverse(p,bad));expect_equal(p$size(),0)
 }
 expect_error(nn$RealNvpCoupling(p,nn$zeros(c(1L,2L)),4L))
 for(ld in list(nn$zeros(c(1L)),nn$zeros(c(2L,1L)),nn$full(integer(),NaN))){
  expect_error(nn$FlowDiagonalScale(p,nn$zeros(c(2L,2L)),ld));expect_equal(p$size(),0)
 }
 nn$RealNvpDensity(p,nn$zeros(c(1L,2L)))
 for(scale in c(1000,-1000,NaN)){
  p$set("log_scale",nn$full(c(2L),scale));expect_error(nn$RealNvpDensity(p,nn$zeros(c(1L,2L))))
  expect_error(nn$RealNvpInverse(p,nn$zeros(c(1L,2L))))
 }
})
