library(testthat)
nn<-datamungerNN::nn
labels<-function(n)nn$zeros(c(n),nn$DType$Int64)
test_that("native dual masks and paired gate retain live higher paths",{
 expect_equal(nn$sum(nn$GatedPixelMask(1L,TRUE,TRUE))$item(),24)
 expect_equal(nn$sum(nn$GatedPixelMask(4L,FALSE,FALSE))$item(),64)
 expect_equal(nn$GatedPixelMask(2L,FALSE,TRUE,3L)$shape(),c(1,3,2,3))
 x<-nn$from_vector_double(c(-1,0,1,2,0,1,-1,2),c(1L,1L,1L,8L))$requires_grad_()
 y<-nn$GatedPixelGate(x)
 expect_equal(y$values(),tanh(c(-1,0,1,2))*plogis(c(0,1,-1,2)))
 graph<-nn$GradOptions();graph$create_graph<-TRUE
 g<-nn$grad(nn$sum(y),list(x),graph)[[1]]
 expect_gt(nn$max(nn$abs(nn$grad(nn$sum(nn$square(g)),list(x))[[1]]))$item(),.01)
 expect_error(nn$GatedPixelMask(0L,TRUE,TRUE))
 expect_error(nn$GatedPixelGate(nn$zeros(c(1L,2L,3L,4L))))
})
test_that("census causal input graph and seven fields survive parent and store",{
 for(conditional in c(FALSE,TRUE)){
  p<-nn$ParameterStore(8501,nn$DType$Float64)
  x<-nn$full(c(1L,5L,7L,1L),.25,nn$DType$Float64)$requires_grad_()
  o<-nn$GatedPixelCnn(p,x,labels(1L),conditional)
  expect_equal(c(p$count(),p$size()),if(conditional)c(1553,29) else c(1073,23))
  fields<-list(o$vertical0,o$horizontal0,o$vertical1,o$horizontal1,o$vertical,o$horizontal,o$logits)
  rm(o,p);gc()
  loss<-nn$sum(nn$slice_axis(nn$reshape(fields[[7]],c(35L)),0L,17L,18L))
  g<-nn$grad(loss,list(x))[[1]]
  expect_true(all(g$values()[18:35]==0))
  expect_true(all(vapply(fields[1:6],function(x)isTRUE(all.equal(as.numeric(x$shape()),c(1,5,7,4))),logical(1))))
 }
 p<-nn$ParameterStore(8609)
 o<-nn$GatedPixelConvolutional(p,nn$zeros(c(2L,4L,6L,3L)),nn$Tensor(),FALSE,3L,30L)
 expect_equal(c(p$count(),p$size()),c(1410,23))
 expect_equal(o$logits$shape(),c(2,4,6,30))
})
test_that("labels affect only conditional mode and invalid inputs create no leaves",{
 x<-nn$full(c(2L,3L,5L,1L),.25)
 for(conditional in c(FALSE,TRUE)){
  p<-nn$ParameterStore(8501)
  a<-nn$GatedPixelCnn(p,x,labels(2L),conditional)$logits
  before<-p$flatten()
  b<-nn$GatedPixelCnn(p,x,nn$ones(c(2L),nn$DType$Int64),conditional)$logits
  if(conditional)expect_gt(nn$max(nn$abs(nn$sub(a,b)))$item(),1e-6) else expect_true(nn$equal_bits(a,b))
  expect_identical(before,p$flatten())
 }
 p<-nn$ParameterStore(1)
 expect_error(nn$GatedPixelCnn(p,x,nn$full(c(2L),10,nn$DType$Int64),FALSE))
 expect_error(nn$GatedPixelCnn(p,x,nn$zeros(c(2L)),TRUE))
 expect_error(nn$GatedPixelConvolutional(p,x,nn$Tensor(),TRUE))
 expect_equal(p$size(),0)
})
