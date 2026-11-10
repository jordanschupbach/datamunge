library(testthat)
nn<-datamungerNN::nn
test_that("degree inventories retain strict source paths and owning fields",{
 for(reverse in c(FALSE,TRUE)){
  o<-nn$MadeMasks(reverse,nn$DType$Float64)
  fields<-list(o$input_degrees,o$hidden1_degrees,o$hidden2_degrees,o$input_mask,o$hidden_mask,o$output_mask)
  rm(o);gc()
  expect_equal(fields[[1]]$values(),if(reverse)64:1 else 1:64)
  expect_equal(fields[[1]]$dtype(),nn$DType$Int32)
  expect_equal(sum(vapply(fields[4:6],function(x)sum(x$values()==0),numeric(1))),1144)
  paths<-nn$matmul(nn$matmul(fields[[4]],fields[[5]]),fields[[6]])$values()
  degree<-fields[[1]]$values();blocked<-outer(degree,degree,">=")
  expect_true(all(matrix(paths,64,64,byrow=TRUE)[blocked]==0))
 }
 a<-nn$from_vector_int64(c(0,1,2),c(3L));b<-nn$from_vector_int64(c(0,2),c(2L))
 expect_equal(nn$AutoregressiveDegreeMask(a,b,TRUE)$values(),c(0,1,0,1,0,0))
 expect_error(nn$AutoregressiveDegreeMask(nn$ones(c(3L)),b))
})
test_that("model census order guards and field graphs survive store collection",{
 for(reverse in c(FALSE,TRUE)){
  p<-nn$ParameterStore(8803,nn$DType$Float64)
  x<-nn$full(c(1L,8L,8L,1L),.25,nn$DType$Float64)$requires_grad_()
  o<-nn$MadeDensity(p,x,reverse);expect_equal(c(p$count(),p$size()),c(2400,6))
  before<-p$flatten();nn$MadeDensity(p,x,!reverse);expect_identical(before,p$flatten())
  h1<-o$hidden1;h2<-o$hidden2;logits<-o$logits;rm(o,p);gc()
  site<-if(reverse)47L else 16L
  g<-nn$grad(nn$sum(nn$slice_axis(nn$reshape(logits,c(64L)),0L,site,site+1L)),list(x))[[1]]
  expect_true(all(if(reverse)g$values()[1:(site+1L)]==0 else g$values()[(site+1L):64]==0))
  expect_equal(h1$shape(),c(1,16));expect_equal(h2$shape(),c(1,16))
 }
 p<-nn$ParameterStore(1)
 expect_error(nn$MadeDensity(p,nn$zeros(c(1L,4L,16L,1L))))
 expect_error(nn$MaskedDensityDense(p,"bad",nn$zeros(c(2L,3L)),nn$full(c(3L,4L),.5)))
 expect_equal(p$size(),0)
})
test_that("masked dense stops its structural mask and retains higher input paths",{
 p<-nn$ParameterStore(1,nn$DType$Float64)
 x<-nn$ones(c(2L,3L),nn$DType$Float64)$requires_grad_()
 mask<-nn$ones(c(3L,4L),nn$DType$Float64)$requires_grad_()
 y<-nn$MaskedDensityDense(p,"toy",x,mask)
 graph<-nn$GradOptions();graph$create_graph<-TRUE;graph$allow_unused<-TRUE
 g<-nn$grad(nn$sum(nn$square(y)),list(x,mask),graph)
 expect_false(g[[2]]$defined());rm(p,y);gc()
 expect_gt(nn$max(nn$abs(nn$grad(nn$sum(g[[1]]),list(x))[[1]]))$item(),0)
})
test_that("joint mixture is normalized and its derivative is component responsibility",{
 p<-nn$from_vector_double(c(.4,.1,.1,.4,.1,.4,.4,.1),c(4L,2L))
 nll<-nn$neg(nn$log(p))$requires_grad_();o<-nn$EqualDensityMixture(nll)
 score<-o$per_image_nll;mu<-o$mean_nll;weight<-o$responsibilities;rm(o);gc()
 expect_equal(score$dtype(),nn$DType$Float64)
 expect_equal(score$values(),rep(log(4),4));expect_equal(nn$sum(nn$exp(nn$neg(score)))$item(),1)
 expect_lt(mu$item(),nn$mean(nll)$item())
 graph<-nn$GradOptions();graph$create_graph<-TRUE;g<-nn$grad(nn$sum(score),list(nll),graph)[[1]]
 expect_equal(g$values(),weight$values())
 h<-nn$grad(nn$sum(nn$slice_axis(g,1L,0L,1L)),list(nll))[[1]]
 expect_equal(h$values(),rep(c(-.16,.16),4))
 o<-nn$EqualDensityMixture(nn$from_vector_double(c(1000,1002,-1002,-1000),c(2L,2L)))
 expect_equal(o$mean_nll$item(),-1-log1p(exp(-2))+log(2))
 expect_error(nn$EqualDensityMixture(nn$zeros(c(2L))))
 expect_error(nn$EqualDensityMixture(nn$zeros(c(2L,2L),nn$DType$Int64)))
})
