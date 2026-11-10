library(testthat)
nn<-datamungerNN::nn
test_that("explicit triples transform all three families with component labels",{
 u<-nn$from_vector_double(c(0,0,0,2,1,-10),c(2L,3L))$requires_grad_()
 banana<-nn$FlowStudyPoints(u,0L);expect_equal(banana$points$values(),c(0,-.5,2,2.5));expect_equal(banana$components$values(),c(-1,-1));expect_equal(banana$points$dtype(),nn$DType$Float32)
 ring<-nn$FlowStudyPoints(u,1L,nn$DType$Float64);expect_equal(ring$components$values(),c(4,0));expect_equal(ring$points$values(),c(-2,0,2.3,.15),tolerance=3e-16)
 checker<-nn$FlowStudyPoints(u,2L,nn$DType$Float64);expect_equal(checker$points$values(),c(-1.5,.5,-1.0227501319481793,-1.1586552539314572));expect_false(banana$points$requires_grad());expect_false(checker$points$requires_grad())
})
test_that("erfc tail and rounded boundary points remain unclipped",{
 u<-nn$from_vector_double(c(-10,0,.5,10,10,10),c(2L,3L));out<-nn$FlowStudyPoints(u,2L,nn$DType$Float64);v<-out$points$values()
 expect_lt(abs(v[1]/7.619853024160593e-24-1),3e-14);expect_equal(v[3:4],c(2,2));expect_equal(out$components$values(),c(5,7));expect_equal(nn$FlowStudyLogProbability(nn$slice_axis(out$points,0L,1L,2L),2L)$item(),-Inf)
})
test_that("reference laws have half-open checker support and stable mixture tails",{
 x<-nn$from_vector_double(c(-2,-2,-1,-2,0,0,2,0,-2,2),c(5L,2L))$requires_grad_();truth<-nn$FlowStudyLogProbability(x,2L)
 expect_equal(truth$values(),c(-log(8),-Inf,-log(8),-Inf,-Inf));expect_false(truth$requires_grad());expect_equal(truth$dtype(),nn$DType$Float64)
 banana<-nn$from_vector_double(c(0,-.5,2,2.5),c(2L,2L));expect_lt(nn$max(nn$abs(nn$sub(nn$FlowStudyLogProbability(banana,0L),nn$BananaLogProbability(banana))))$item(),1e-15)
 expect_true(is.finite(nn$FlowStudyLogProbability(nn$from_vector_double(c(100,100),c(1L,2L)),1L)$item()))
})
test_that("quality diagnostics use inclusive thresholds and own count storage",{
 samples<-nn$from_vector_double(c(2,0,-2,0,2,.45,2,.45000000000000007,0,0),c(5L,2L));out<-nn$FlowStudySampleDiagnostics(samples);counts<-out$mode_counts
 expect_equal(counts$values(),c(2,0,0,0,1,0,0,0));expect_equal(out$modes_covered,2L);expect_equal(out$fraction_near_modes,.6);expect_equal(out$checkerboard_support_fraction,.4);expect_equal(out$banana_curvature_moment,.54)
 points<-nn$FlowStudyPoints(nn$zeros(c(1L,3L)),0L)$points;rm(out,samples);gc();expect_equal(counts$values(),c(2,0,0,0,1,0,0,0));expect_equal(points$values(),c(0,-.5))
})
test_that("strided floating inputs remain immutable",{
 for(dtype in c(nn$DType$Float32,nn$DType$Float64)){
  u<-nn$cast(nn$from_vector_double(c(0,0,0,2,1,-1),c(2L,3L)),dtype);before<-u$clone();strided<-nn$transpose(nn$transpose(u,c(1L,0L))$contiguous(),c(1L,0L))
  for(kind in 0:2){a<-nn$FlowStudyPoints(u,kind,dtype);b<-nn$FlowStudyPoints(strided,kind,dtype);expect_true(nn$equal_bits(a$points,b$points));expect_true(nn$equal_bits(a$components,b$components))}
  expect_true(nn$equal_bits(u,before))
 }
})
test_that("invalid geometry selectors and unsafe math reject",{
 for(bad in list(nn$zeros(c(3L)),nn$zeros(c(0L,3L)),nn$zeros(c(1L,2L)),nn$full(c(1L,3L),NaN)))expect_error(nn$FlowStudyPoints(bad,0L))
 for(kind in c(-1L,3L)){expect_error(nn$FlowStudyPoints(nn$zeros(c(1L,3L)),kind));expect_error(nn$FlowStudyLogProbability(nn$zeros(c(1L,2L)),kind))}
 for(bad in list(nn$zeros(c(2L)),nn$zeros(c(0L,2L)),nn$zeros(c(1L,3L)),nn$full(c(1L,2L),Inf))){expect_error(nn$FlowStudySampleDiagnostics(bad));expect_error(nn$FlowStudyLogProbability(bad,2L))}
 expect_error(nn$FlowStudyPoints(nn$zeros(c(1L,3L)),0L,nn$DType$Int32));expect_error(nn$FlowStudySampleDiagnostics(nn$full(c(1L,2L),1e200,nn$DType$Float64)));expect_error(nn$FlowStudyLogProbability(nn$full(c(1L,2L),1e200,nn$DType$Float64),1L))
})
