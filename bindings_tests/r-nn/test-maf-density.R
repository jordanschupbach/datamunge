library(testthat)
nn<-datamungerNN::nn
test_that("strict masks and actual conditioner Jacobians exclude own and future inputs",{
 graph<-nn$GradOptions();graph$retain_graph<-TRUE;graph$create_graph<-TRUE
 for(block in 0:3){
  masks<-nn$MafMasks(block,nn$DType$Float64);degrees<-if(block%%2)c(2,1) else c(1,2)
  expect_equal(masks$input_degrees$values(),degrees);expect_equal(masks$output_degrees$values(),rep(degrees,2))
  expect_equal(masks$hidden_degrees$values(),c(rep(0,8),rep(1,8)))
  im<-masks$input_mask;om<-masks$output_mask;rm(masks);gc()
  expect_equal(sum(im$values()==0)+sum(om$values()==0),40);expect_false(im$requires_grad());expect_equal(im$dtype(),nn$DType$Float64)
  p<-nn$ParameterStore(9603,nn$DType$Float64)
  x<-nn$from_vector_double(c(.2,-.7,1.1,.4,-.3,2),c(3L,2L))$requires_grad_();out<-nn$MafConditioner(p,x,block)
  expect_equal(p$count(),116);expect_equal(p$size(),4)
  for(k in 0:3){head<-nn$slice_axis(if(k<2)out$mean else out$scale,1L,k%%2,k%%2+1)
   g<-nn$grad(nn$sum(head),list(x),graph)[[1]]$values()
   for(row in 0:2)for(j in 0:1)if(degrees[j+1]>=degrees[k%%2+1])expect_equal(g[row*2+j+1],0)
  }
 }
})
test_that("parallel density and sequential inverse retain owning higher graphs",{
 p<-nn$ParameterStore(9603,nn$DType$Float64)
 x<-nn$from_vector_double(c(.4,-.7,1.2,.3,-1,2),c(3L,2L))$requires_grad_()
 path<-nn$MafDensity(p,x);expect_equal(p$count(),466);expect_equal(p$size(),17)
 expect_equal(path$logdet$shape(),c(3));expect_equal(path$coupling_scale2$shape(),c(3,2))
 before<-p$flatten();rec<-nn$MafInverse(p,path$latent)
 expect_lt(nn$max(nn$abs(nn$sub(rec$value,x)))$item(),1e-14)
 expect_lt(nn$max(nn$abs(nn$add(rec$logdet,path$logdet)))$item(),1e-14)
 density<-nn$NormalFlowDensity(path$latent,path$logdet,TRUE);cond<-nn$MafConditioner(p,x,1L)
 input2<-path$conditioner_input2;scale<-path$coupling_scale3;shift<-path$coupling_shift1;mean<-cond$mean;loss<-density$loss
 rm(path,cond,rec,density);gc();graph<-nn$GradOptions();graph$retain_graph<-TRUE;graph$create_graph<-TRUE
 first<-nn$grad(loss,list(x),graph)[[1]];expect_true(nn$grad(nn$sum(first),list(x),graph)[[1]]$defined())
 expect_identical(p$flatten(),before);rm(p);gc()
 for(field in list(input2,scale,shift,mean))expect_true(nn$grad(nn$sum(field),list(x),graph)[[1]]$defined())
})
test_that("host Adam preserves all160 structurally forbidden stored coordinates",{
 p<-nn$ParameterStore(9603);x<-nn$from_vector(c(.4,-.7,1.2,.3,-1,2),c(3L,2L));nn$MafDensity(p,x)
 config<-nn$OptimizerConfig();config$kind<-nn$OptimizerKind$Adam;optimizer<-nn$Optimizer(p,.003,config)
 names<-p$names();before<-lapply(names,function(name)p$at(name)$values());names(before)<-names
 path<-nn$MafDensity(p,x);loss<-nn$NormalFlowDensity(path$latent,path$logdet)$loss
 gradients<-nn$grad(loss,p$parameters());optimizer$step(nn$tensor_vector(gradients));checked<-0
 for(block in 0:3){masks<-nn$MafMasks(block)
  for(suffix in c("hidden/W","heads/W")){
   mask<-if(suffix=="hidden/W")masks$input_mask else masks$output_mask
   name<-paste0("autoregressive",block,"/",suffix);g<-gradients[[match(name,names)]]$values();after<-p$at(name)$values();indices<-which(mask$values()==0)
   expect_equal(g[indices],rep(0,length(indices)));expect_identical(after[indices],before[[name]][indices]);checked<-checked+length(indices)
  }
 }
 expect_equal(checked,160);expect_equal(optimizer$iterations(),1)
})
test_that("invalid masks inputs and unrepresentable global scales reject",{
 p<-nn$ParameterStore(9603)
 for(block in c(-1L,4L)){expect_error(nn$MafMasks(block));expect_error(nn$MafConditioner(p,nn$zeros(c(2L,2L)),block))}
 expect_error(nn$MafMasks(0L,nn$DType$Int32))
 for(bad in list(nn$zeros(c(2L)),nn$zeros(c(1L,3L)),nn$zeros(c(0L,2L)),nn$full(c(1L,2L),NaN))){
  expect_error(nn$MafDensity(p,bad));expect_error(nn$MafInverse(p,bad));expect_equal(p$size(),0)
 }
 nn$MafDensity(p,nn$zeros(c(1L,2L)));p$set("log_scale",nn$full(c(2L),1000))
 expect_error(nn$MafDensity(p,nn$zeros(c(1L,2L))));expect_error(nn$MafInverse(p,nn$zeros(c(1L,2L))))
})
