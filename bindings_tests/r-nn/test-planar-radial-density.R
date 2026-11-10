library(testthat)
nn<-datamungerNN::nn
keep<-function(){g<-nn$GradOptions();g$retain_graph<-TRUE;g$create_graph<-TRUE;g}
scalar<-function(v)nn$full(integer(),v,nn$DType$Float64)
test_that("both density families retain owning stages inverse and higher paths",{
 for(radial in c(FALSE,TRUE)){
  p<-nn$ParameterStore(9903,nn$DType$Float64);x<-nn$from_vector_double(c(.4,-.7,1.2,.3,-1,2),c(3L,2L))$requires_grad_()
  path<-nn$PlanarRadialDensity(p,x,radial);expect_equal(p$count(),if(radial)18 else 22);expect_equal(p$size(),if(radial)13 else 21)
  expect_equal(path$flow_factor1$shape(),c(3,1));expect_equal(path$flow_coordinate2$shape(),c(3,1))
  inv<-nn$PlanarRadialInverse(p,path$latent,radial)
  expect_lt(nn$max(nn$abs(nn$sub(inv$value,x)))$item(),2e-13);expect_lt(nn$max(nn$abs(nn$add(inv$logdet,path$logdet)))$item(),2e-13)
  factor<-path$flow_factor1;coordinate<-path$flow_coordinate2;stage<-path$flow3;inverse<-inv$value;rm(path,inv,p);gc()
  for(field in list(factor,coordinate,stage,inverse)){
   first<-nn$grad(nn$sum(field),list(x),keep())[[1]];expect_true(nn$grad(nn$sum(nn$square(first)),list(x),keep())[[1]]$defined())
  }
 }
})
test_that("explicit host Adam and both inverse parameter paths work",{
 for(radial in c(FALSE,TRUE)){
  p<-nn$ParameterStore(9903);x<-nn$from_vector(c(.4,-.7,1.2,.3,-1,2),c(3L,2L));nn$PlanarRadialDensity(p,x,radial)
  config<-nn$OptimizerConfig();config$kind<-nn$OptimizerKind$Adam;optimizer<-nn$Optimizer(p,.003,config)
  before<-p$flatten();path<-nn$PlanarRadialDensity(p,x,radial);gradients<-nn$grad(nn$NormalFlowDensity(path$latent,path$logdet)$loss,p$parameters());optimizer$step(nn$tensor_vector(gradients))
  expect_equal(optimizer$iterations(),1);expect_false(identical(p$flatten(),before))
  inverse<-nn$PlanarRadialInverse(p,x,radial);objective<-nn$add(nn$mean(nn$sum(nn$square(inverse$value),c(1L)),c(0L)),nn$mul(nn$mean(inverse$logdet,c(0L)),.3))
  partials<-nn$grad(objective,p$parameters());expect_equal(length(partials),p$size());expect_true(all(vapply(partials,function(g)g$defined(),logical(1))))
 }
})
test_that("pure planar actual determinant and stopped higher correction stay explicit",{
 x<-nn$from_vector_double(c(.3,-.2),c(1L,2L));bias<-scalar(.1)$requires_grad_();raw<-scalar(-1)
 inv<-nn$PlanarFlow(x,scalar(0),scalar(0),raw,scalar(0),bias,TRUE)
 first<-nn$grad(nn$sum(nn$slice_axis(inv$value,1L,0L,1L)),list(bias),keep())[[1]];second<-nn$grad(first,list(bias),keep())[[1]]
 p<-log1p(exp(-1))-.9;t<-inv$coordinate$item();a<-tanh(t);sech<-1-a*a;low<-.4-(abs(p)+1);high<-.4+(abs(p)+1)
 for(i in 1:32){mid<-.5*(low+high);if((mid+p*tanh(mid))-.4<0)low<-mid else high<-mid}
 root<-.5*(low+high);den<-1+p*(1-tanh(root)^2)
 expect_lt(abs(first$item()+p*sech/den),2e-13);expect_lt(abs(second$item()-2*p*a*sech/(den*den)),2e-12)
 points<-nn$from_vector_double(c(.4,-.7,1.2,.3),c(2L,2L))$requires_grad_()
 out<-nn$PlanarFlow(points,scalar(.6),scalar(.3),scalar(-6),scalar(.7),scalar(.1))
 j0<-nn$grad(nn$sum(nn$slice_axis(out$value,1L,0L,1L)),list(points),keep())[[1]]$values();j1<-nn$grad(nn$sum(nn$slice_axis(out$value,1L,1L,2L)),list(points),keep())[[1]]$values()
 for(r in 0:1)expect_lt(abs(log(j0[2*r+1]*j1[2*r+2]-j0[2*r+2]*j1[2*r+1])-out$logdet$values()[r+1]),2e-12)
})
test_that("radial contraction includes exact center and both stable inverse branches",{
 center<-nn$from_vector_double(c(.3,-.2),c(2L));x<-nn$from_vector_double(c(.3,-.2,.31,-.19,10,-10),c(3L,2L))$requires_grad_()
 alpha<-scalar(.4);beta<-scalar(log(expm1(.12)));out<-nn$RadialFlow(x,center,alpha,beta);rec<-nn$RadialFlow(out$value,center,alpha,beta,TRUE)
 expect_lt(nn$max(nn$abs(nn$sub(rec$value,x)))$item(),2e-13)
 row<-nn$grad(nn$sum(nn$slice_axis(out$value,1L,0L,1L)),list(x),keep())[[1]]$values();expect_lt(abs(row[1]-out$factor$values()[1]),1e-15);expect_equal(row[2],0)
 radius<-nn$grad(nn$sum(out$coordinate),list(x),keep())[[1]]$values();expect_equal(radius[1:2],c(0,0))
})
test_that("invalid elementary geometry frequency factors and global scales reject",{
 p<-nn$ParameterStore(9903)
 for(bad in list(nn$zeros(c(2L)),nn$zeros(c(1L,3L)),nn$zeros(c(0L,2L)),nn$full(c(1L,2L),NaN)))expect_error(nn$PlanarRadialDensity(p,bad))
 expect_error(nn$ElementaryFlow(p,nn$zeros(c(1L,2L)),4L));expect_equal(p$size(),0)
 x<-nn$zeros(c(1L,2L));zero<-nn$full(integer(),0)
 expect_error(nn$PlanarFlow(x,nn$zeros(c(1L)),zero,zero,zero,zero));expect_error(nn$PlanarFlow(x,zero,nn$full(integer(),1000),zero,zero,zero))
 expect_error(nn$RadialFlow(x,nn$zeros(c(2L)),zero,nn$full(integer(),-1000)))
 nn$PlanarRadialDensity(p,x);p$set("log_scale",nn$full(c(2L),1000));expect_error(nn$PlanarRadialInverse(p,x))
})
