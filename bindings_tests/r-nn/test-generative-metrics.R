library(testthat)
nn<-datamungerNN::nn

test_that("unequal Gaussian feature pools match closed form and own detached fields",{
 a<-nn$from_vector_double(c(0,2),c(2L,1L))$requires_grad_()
 b<-nn$from_vector(c(1,3,5),c(3L,1L))$requires_grad_()
 o<-nn$GaussianFrechet(a,b)
 expect_equal(o$fid$item(),4+(sqrt(2)-2)^2)
 expect_equal(o$fid$item(),nn$GaussianFrechet(b,a)$fid$item())
 fields<-list(o$fid,o$raw_fid,o$mean_term,o$trace_a,o$trace_b,o$cross_trace,o$cross_singular_values)
 rm(o,a,b);gc()
 expect_true(all(vapply(fields,function(x)!x$requires_grad(),logical(1))))
 expect_equal(fields[[3]]$item(),4)
 expect_equal(fields[[7]]$shape(),2)
})
test_that("probability controls and unequal ordered partitions retain population spread",{
 p<-nn$from_vector_double(c(1,0,1,0,0,1,0,1,1,0),c(5L,2L))$requires_grad_()
 expect_equal(nn$InceptionScore(nn$full(c(5L,4L),.25))$item(),1)
 expect_equal(nn$InceptionScore(nn$from_vector_double(c(1,0,0,1),c(2L,2L)))$item(),2)
 o<-nn$InceptionScoreSplits(p,3L)
 score<-o$scores;mu<-o$mean;spread<-o$population_std
 rm(o,p);gc()
 expect_equal(score$values(),c(1,2,2))
 expect_equal(mu$item(),5/3)
 expect_equal(spread$item(),sqrt(2/9))
 expect_false(score$requires_grad())
})
test_that("invalid feature widths probability rows and split counts reject",{
 expect_error(nn$GaussianFrechet(nn$zeros(c(1L,2L)),nn$zeros(c(2L,2L))))
 expect_error(nn$GaussianFrechet(nn$zeros(c(2L,2L)),nn$zeros(c(2L,3L))))
 expect_error(nn$InceptionScore(nn$from_vector(c(.5,.8),c(1L,2L))))
 expect_error(nn$InceptionScore(nn$from_vector(c(-.1,1.1),c(1L,2L))))
 expect_error(nn$InceptionScoreSplits(nn$full(c(2L,3L),1/3),3L))
})
