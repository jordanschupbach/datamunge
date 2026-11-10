library(testthat)
nn<-datamungerNN::nn
test_that("generic bin geometry bounds and owning metadata retain higher paths",{
 x<-nn$from_vector_double(c(-4,-.7,.4,4),c(4L,1L));heads<-nn$full(c(4L,5L),.15,nn$DType$Float64)$requires_grad_()
 out<-nn$RationalQuadraticSpline(x,heads,FALSE,2.5,.03,.002)
 expect_equal(out$widths$shape(),c(4,2));expect_equal(out$derivatives$shape(),c(4,3))
 rec<-nn$RationalQuadraticSpline(out$value,heads,TRUE,2.5,.03,.002)
 expect_lt(nn$max(nn$abs(nn$sub(rec$value,x)))$item(),1e-14)
 expect_equal(out$value$values()[1],-4);expect_equal(out$value$values()[4],4)
 expect_gt(nn$min(out$widths)$item(),0);expect_gt(nn$min(out$derivatives)$item(),0)
 width<-nn$slice_axis(out$widths,1L,0L,1L);rm(out,rec);gc();graph<-nn$GradOptions();graph$retain_graph<-TRUE;graph$create_graph<-TRUE
 first<-nn$grad(nn$sum(width),list(heads),graph)[[1]];expect_true(nn$grad(nn$sum(nn$square(first)),list(heads),graph)[[1]]$defined())
})
test_that("rounded source offset and actual scalar derivatives remain explicit",{
 x<-nn$from_vector_double(c(-5,-.7,.3,5),c(4L,1L))$requires_grad_()
 heads<-nn$from_vector_double(rep(c(0,0,0,0,.2,-.3,.1,.4,-.2,.3,.6),4),c(4L,11L))$requires_grad_()
 out<-nn$RationalQuadraticSpline(x,heads);graph<-nn$GradOptions();graph$retain_graph<-TRUE;graph$create_graph<-TRUE
 j<-nn$grad(nn$sum(out$value),list(x),graph)[[1]];expect_gt(nn$min(j)$item(),0)
 expect_lt(nn$max(nn$abs(nn$sub(nn$reshape(nn$log(j),c(4L)),out$logdet)))$item(),1e-14)
 expect_true(nn$grad(nn$sum(j),list(heads),graph)[[1]]$defined())
 zero<-nn$zeros(c(2L,11L),nn$DType$Float64);points<-nn$full(c(2L,1L),.25,nn$DType$Float64)
 precise<-nn$RationalQuadraticSpline(points,zero);source<-nn$RationalQuadraticSpline(points,zero,FALSE,4,.01,.001,TRUE)
 expect_gt(abs(precise$derivatives$values()[2]-source$derivatives$values()[2]),1e-10)
})
test_that("compact878-value spline inverse and flat fields retain higher graphs after collection",{
 p<-nn$ParameterStore(9803,nn$DType$Float64);x<-nn$from_vector_double(c(.4,-.7,1.2,.3,-1,2),c(3L,2L))$requires_grad_()
 path<-nn$SplineDensity(p,x);expect_equal(p$count(),878);expect_equal(p$size(),17)
 expect_equal(path$coupling_widths1$shape(),c(3,4));expect_equal(path$coupling_derivatives2$shape(),c(3,5))
 rec<-nn$SplineInverse(p,path$latent);before<-p$flatten()
 expect_lt(nn$max(nn$abs(nn$sub(rec$value,x)))$item(),1e-13)
 expect_lt(nn$max(nn$abs(nn$add(rec$logdet,path$logdet)))$item(),1e-13)
 loss<-nn$NormalFlowDensity(path$latent,path$logdet)$loss
 width<-nn$slice_axis(path$coupling_widths3,1L,0L,1L);height<-nn$slice_axis(path$coupling_heights1,1L,0L,1L)
 derivative<-nn$slice_axis(path$coupling_derivatives2,1L,1L,2L);stage<-path$coupling0
 rm(path,rec);gc();graph<-nn$GradOptions();graph$retain_graph<-TRUE;graph$create_graph<-TRUE
 first<-nn$grad(loss,list(x),graph)[[1]];expect_true(nn$grad(nn$sum(first),list(x),graph)[[1]]$defined())
 expect_identical(p$flatten(),before);rm(p);gc()
 for(field in list(width,height,derivative,stage))expect_true(nn$grad(nn$sum(field),list(x),graph)[[1]]$defined())
})
test_that("host-owned Adam updates all source spline leaves explicitly",{
 p<-nn$ParameterStore(9803);x<-nn$from_vector(c(.4,-.7,1.2,.3,-1,2),c(3L,2L));nn$SplineDensity(p,x)
 config<-nn$OptimizerConfig();config$kind<-nn$OptimizerKind$Adam;optimizer<-nn$Optimizer(p,.003,config)
 before<-p$flatten();path<-nn$SplineDensity(p,x);gradients<-nn$grad(nn$NormalFlowDensity(path$latent,path$logdet)$loss,p$parameters())
 optimizer$step(nn$tensor_vector(gradients));expect_equal(optimizer$iterations(),1);expect_false(identical(p$flatten(),before));expect_equal(p$count(),878)
 expect_true(nn$SplineInverse(p,nn$zeros(c(2L,2L)))$value$defined())
})
test_that("invalid pure spline geometry floors and global scales reject",{
 x<-nn$zeros(c(2L,1L));heads<-nn$zeros(c(2L,11L))
 for(bad in list(nn$zeros(c(2L)),nn$zeros(c(2L,2L)),nn$zeros(c(0L,1L)),nn$full(c(2L,1L),NaN)))expect_error(nn$RationalQuadraticSpline(bad,heads))
 expect_error(nn$RationalQuadraticSpline(x,nn$zeros(c(2L,4L))))
 expect_error(nn$RationalQuadraticSpline(x,heads,FALSE,0));expect_error(nn$RationalQuadraticSpline(x,heads,FALSE,4,.25))
 expect_error(nn$RationalQuadraticSpline(x,heads,FALSE,4,.01,1))
 p<-nn$ParameterStore(9803);expect_error(nn$SplineDensity(p,nn$zeros(c(1L,3L))))
 expect_error(nn$SplineCoupling(p,nn$zeros(c(1L,2L)),4L));expect_equal(p$size(),0)
 nn$SplineDensity(p,nn$zeros(c(1L,2L)));p$set("log_scale",nn$full(c(2L),1000))
 expect_error(nn$SplineDensity(p,nn$zeros(c(1L,2L))));expect_error(nn$SplineInverse(p,nn$zeros(c(1L,2L))))
})
