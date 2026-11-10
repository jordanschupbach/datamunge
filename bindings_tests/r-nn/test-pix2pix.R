library(testthat)
nn <- datamungerNN::nn
px_inputs <- function(){t<-nn$DType$Float64;list(g=nn$ParameterStore(6101L,t),d=nn$ParameterStore(6113L,t),x=nn$full(c(2L,28L,28L,1L),.2,t),c=nn$ones(c(2L,28L,28L,1L),t))}

test_that("Native quantized Sobel uses inclusive integer threshold and zero borders",{
 x<-nn$ones(c(1L,3L,3L,1L),nn$DType$Float64)$requires_grad_();edge<-nn$quantized_sobel_edges(x)
 expect_equal(edge$shape(),c(1,3,3,1));expect_false(edge$requires_grad());expect_equal(nn$sum(edge)$item(),8)
 impulse<-nn$from_vector_double(c(0,0,0,0,64/255,0,0,0,0),c(1L,3L,3L,1L));expect_equal(nn$sum(nn$quantized_sobel_edges(impulse,128L))$item(),8);expect_equal(nn$sum(nn$quantized_sobel_edges(impulse,129L))$item(),0)
 expect_error(nn$quantized_sobel_edges(nn$full(c(1L,2L,2L,1L),NaN)));expect_error(nn$quantized_sobel_edges(x,-1L))
})

test_that("Both real skip joins and conditional source paths preserve their contracts",{
 a<-px_inputs();o<-nn$Pix2Pix(a$g,a$d,a$x,a$c);gen<-nn$Pix2PixGenerate(a$g,a$c)
 expect_equal(c(a$g$count(),a$g$size(),a$d$count(),a$d$size()),c(1390,10,725,6));expect_equal(o$real_logit$shape(),c(2,7,7,1))
 expect_identical(nn$slice_axis(gen$joined14,3L,0L,4L)$values(),gen$up14$values());expect_identical(nn$slice_axis(gen$joined14,3L,4L,8L)$values(),gen$down14$values());expect_identical(nn$slice_axis(gen$joined28,3L,2L,3L)$values(),a$c$values())
 before<-list(a$g$flatten(),a$d$flatten());expect_identical(nn$Pix2Pix(a$g,a$d,nn$ones(a$x$shape(),a$x$dtype()),a$c)$fake$values(),o$fake$values());blank<-nn$zeros(a$c$shape(),a$c$dtype());expect_false(identical(nn$Pix2PixGenerate(a$g,blank)$image$values(),o$fake$values()));expect_false(identical(nn$Pix2PixDiscriminate(a$d,blank,a$x)$values(),o$real_logit$values()));expect_identical(list(a$g$flatten(),a$d$flatten()),before)
})

test_that("Explicit separate Adam writes use a fresh G forward through updated D",{
 a<-px_inputs();o<-nn$Pix2Pix(a$g,a$d,a$x,a$c);cfg<-nn$OptimizerConfig();cfg$kind<-nn$OptimizerKind$Adam;cfg$beta1<-.5;od<-nn$Optimizer(a$d,.0002,cfg);og<-nn$Optimizer(a$g,.0002,cfg)
 before<-a$g$flatten();od$step(nn$tensor_vector(nn$grad(o$discriminator_loss,a$d$parameters())));expect_identical(a$g$flatten(),before);after_d<-a$d$flatten();fresh<-nn$Pix2Pix(a$g,a$d,a$x,a$c);og$step(nn$tensor_vector(nn$grad(fresh$generator_loss,a$g$parameters())));expect_identical(a$d$flatten(),after_d);expect_false(identical(a$g$flatten(),before));expect_equal(c(od$iterations(),og$iterations()),c(1,1))
})

test_that("Owned fields retain higher source paths and patch objectives use the source scales",{
 a<-px_inputs();c<-a$c$requires_grad_();o<-nn$Pix2Pix(a$g,a$d,a$x,c);loss<-o$generator_loss;join<-nn$Pix2PixGenerate(a$g,c)$joined28;rm(a,o);gc();options<-nn$GradOptions();options$create_graph<-TRUE;first<-nn$grad(loss,list(c),options)[[1]];second<-nn$grad(nn$sum(nn$square(first)),list(c))[[1]];expect_equal(second$shape(),c$shape());expect_gt(nn$max(nn$abs(second))$item(),0);expect_identical(nn$slice_axis(join,3L,2L,3L)$values(),c$values())
 a<-px_inputs();expect_error(nn$Pix2Pix(a$g,a$d,a$x,a$c,-1));expect_error(nn$Pix2Pix(a$g,a$g,a$x,a$c));expect_equal(a$g$size()+a$d$size(),0)
 nn$Pix2Pix(a$g,a$d,a$x,a$c);for(name in a$d$names())a$d$set(name,nn$zeros(a$d$at(name)$shape(),a$d$dtype()));o<-nn$Pix2Pix(a$g,a$d,a$x,a$c);expect_equal(o$discriminator_loss$item(),2*log(2));expect_equal(o$generator_loss$item(),o$g_adversarial$item()+100*o$l1$item());expect_identical(nn$Pix2Pix(a$g,a$d,a$x,a$c,0)$generator_loss$values(),o$g_adversarial$values())
})
