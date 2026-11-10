library(testthat)
nn<-datamungerNN::nn
test_that("integer RGB area means round half up and own detached fields",{
 raw<-nn$from_vector_int64(c(0,0,0,1,1,1,0,0,0,1,1,1),c(1L,2L,2L,3L))
 o<-nn$RgbAreaPixels(raw,2L);codes<-o$codes;image<-o$images;rm(o,raw);gc()
 expect_equal(codes$values(),c(1,1,1));expect_equal(codes$dtype(),nn$DType$Int32)
 expect_equal(image$shape(),c(1,1,1,3));expect_equal(image$values(),rep(1/127.5-1,3),tolerance=1e-7)
 expect_false(image$requires_grad())
 expect_error(nn$RgbAreaPixels(nn$zeros(c(1L,4L,4L,3L))))
 expect_error(nn$RgbAreaPixels(nn$full(c(1L,4L,4L,3L),256,nn$DType$Int64)))
 expect_error(nn$RgbAreaPixels(nn$zeros(c(1L,4L,4L,3L),nn$DType$Int64),3L))
})
test_that("all256 logistic bins include normalized infinite endpoint tails",{
 image<-nn$from_vector_double(rep((0:255)/127.5-1,each=3),c(256L,3L))
 o<-nn$DiscretizedLogisticRGB(image,nn$zeros(c(256L,10L),nn$DType$Float64),1L)
 mass<-o$channel_log_mass;weight<-o$mixture_log_probability;loss<-o$mean_channel_nll;rm(o);gc()
 expect_equal(nn$sum(nn$exp(mass),c(0L))$values(),rep(1,3),tolerance=2e-5)
 expect_equal(weight$values(),rep(0,256));expect_true(is.finite(loss$item()))
 expect_error(nn$DiscretizedLogisticRGB(image,nn$zeros(c(256L,20L),nn$DType$Float64)))
 expect_error(nn$DiscretizedLogisticRGB(nn$full(c(1L,3L),1.001),nn$zeros(c(1L,30L))))
})
test_that("likelihood owns image parameter higher paths and clips raw scales",{
 values<-rep(.1,10);values[5]<- -8
 parameters<-nn$from_vector_double(values,c(1L,10L))$requires_grad_()
 image<-nn$from_vector_double(c(-.6,.2,.8),c(1L,3L))$requires_grad_()
 o<-nn$DiscretizedLogisticRGB(image,parameters,1L);means<-o$means;loss<-o$mean_channel_nll;rm(o);gc()
 expect_equal(means$values(),c(.1,.1+tanh(.1)*(-.6),.1+tanh(.1)*(-.4)))
 graph<-nn$GradOptions();graph$create_graph<-TRUE;g<-nn$grad(loss,list(parameters,image),graph)
 expect_equal(g[[1]]$values()[5],0);expect_true(g[[2]]$defined())
 second<-nn$grad(nn$sum(g[[1]]),list(parameters))[[1]];expect_true(all(is.finite(second$values())))
})
test_that("compact model fields outlive store and exclude whole current pixels",{
 p<-nn$ParameterStore(8609,nn$DType$Float64)
 x<-nn$full(c(1L,2L,3L,3L),.25,nn$DType$Float64)$requires_grad_();o<-nn$PixelCnnPP(p,x)
 expect_equal(c(p$count(),p$size()),c(1410,23))
 head<-o$parameters;loss<-o$loss;score<-o$per_image_nll;vertical<-o$vertical
 expect_equal(score$item()/18,loss$item());rm(o,p);gc()
 expect_equal(vertical$shape(),c(1,2,3,4))
 g<-nn$grad(nn$sum(nn$slice_axis(nn$reshape(head,c(1L,6L,30L)),1L,2L,3L)),list(x))[[1]]
 expect_true(all(g$values()[7:18]==0))
 fresh<-nn$ParameterStore(1);expect_error(nn$PixelCnnPP(fresh,nn$full(c(1L,2L,3L,3L),1.1)));expect_equal(fresh$size(),0)
})
