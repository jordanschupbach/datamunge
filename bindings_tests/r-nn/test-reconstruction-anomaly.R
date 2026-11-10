library(testthat)
nn <- datamungerNN::nn

test_that("Residual and ink scores flatten widen detach and own values", {
  x <- nn$from_vector(c(0,1,2,3),c(2L,2L))$requires_grad_();y <- nn$zeros(c(2L,1L,2L),nn$DType$Float64)$requires_grad_()
  residual <- nn$ReconstructionScores(x,y);ink <- nn$InkScores(x,.5)
  expect_equal(residual$dtype(),nn$DType$Float64);expect_equal(residual$values(),c(.5,6.5),tolerance=0)
  expect_equal(ink$values(),c(0,2),tolerance=0);expect_false(residual$requires_grad());expect_false(ink$requires_grad())
  expect_true(x$requires_grad());expect_true(y$requires_grad());nn$no_grad(function()x$fill_(9));rm(x,y);gc()
  expect_equal(residual$values(),c(.5,6.5),tolerance=0)
  expect_equal(nn$InkScores(nn$from_vector_double(c(1e16,1,-1e16),c(1L,3L)),.25)$item(),.25,tolerance=0)
})
test_that("Nominal nearest-rank calibration has no interpolation or query refit", {
  scores <- nn$select(nn$arange(0,40)$reshape(c(20L,2L)),1L,0L)$requires_grad_()
  expect_equal(nn$NominalThreshold(scores),36,tolerance=0);expect_equal(nn$NominalThreshold(scores,.5),18,tolerance=0)
  expect_equal(nn$NominalThreshold(scores,.5+.Machine$double.eps),20,tolerance=0)
  expect_equal(nn$NominalThreshold(scores,1),38,tolerance=0);expect_true(scores$requires_grad())
  threshold <- nn$NominalThreshold(nn$arange(0,20));expect_equal(threshold,18,tolerance=0)
  result <- nn$RankAnomalies(nn$from_vector(c(18,19),2L),nn$from_vector(c(18,1000),2L),threshold)
  expect_equal(result$normal_fpr,.5,tolerance=0);expect_equal(result$anomaly_tpr,.5,tolerance=0)
})
test_that("Tied rankings strict thresholds and prevalence remain explicit", {
  normal <- nn$from_vector_double(c(1,2),2L)$requires_grad_();anomaly <- nn$from_vector(c(2,3),2L)
  result <- nn$RankAnomalies(normal,anomaly,2);expect_equal(result$auroc,.875,tolerance=0)
  expect_equal(result$average_precision,5/6,tolerance=1e-15);expect_equal(result$normal_fpr,0,tolerance=0);expect_equal(result$anomaly_tpr,.5,tolerance=0)
  lower <- nn$RankAnomalies(normal,anomaly,2-.Machine$double.eps);expect_equal(lower$normal_fpr,.5,tolerance=0);expect_equal(lower$anomaly_tpr,1,tolerance=0)
  tied <- nn$RankAnomalies(nn$ones(3L),nn$ones(2L,nn$DType$Float64),1)
  expect_equal(c(tied$auroc,tied$average_precision,tied$normal_fpr,tied$anomaly_tpr),c(.5,.4,0,0),tolerance=0)
  expect_true(normal$requires_grad());rm(normal,anomaly);gc();expect_equal(result$auroc,.875,tolerance=0)
})
test_that("Paired stress transforms retain full pixels shapes flags and ownership", {
  original <- nn$transpose(nn$full(c(2L,28L,28L,1L),.5),c(0L,2L,1L,3L))$requires_grad_()
  outputs <- list()
  for(kind in 0:2) {
    out <- nn$AnomalyCorruption(original,kind);expect_true(out$is_contiguous());expect_false(out$requires_grad());expect_equal(out$shape(),c(2,28,28,1))
    expected <- rep(.5,1568)
    if(kind==0)for(b in 0:1)for(y in 0:7)for(x in 0:7)expected[b*784+y*28+x+1]<-1
    if(kind==1)expected<-nn$full(c(2L,28L,28L,1L),.175)$values()
    if(kind==2)for(b in 0:1)for(y in 0:27)for(x in 0:5)expected[b*784+y*28+x+1]<-0
    expect_equal(out$values(),expected,tolerance=0);outputs[[kind+1]]<-out
  }
  expect_true(original$requires_grad());expect_equal(original$values(),rep(.5,1568),tolerance=0)
  nn$no_grad(function()original$fill_(9));rm(original);gc();expect_equal(outputs[[3]]$values()[11],.5,tolerance=0)
})
test_that("Malformed score geometry calibration and stress inputs fail", {
  for(bad in list(nn$zeros(0L),nn$ones(c(2L,2L)),nn$ones(2L,nn$DType$Int32),nn$full(2L,NaN))) {
    expect_error(nn$NominalThreshold(bad));expect_error(nn$RankAnomalies(bad,nn$ones(2L),0))
  }
  for(q in c(0,-1,1.00001,Inf,NaN))expect_error(nn$NominalThreshold(nn$ones(2L),q))
  expect_error(nn$RankAnomalies(nn$ones(2L),nn$ones(2L),Inf));expect_error(nn$ReconstructionScores(nn$ones(c(2L,3L)),nn$ones(c(3L,2L))))
  expect_error(nn$InkScores(nn$ones(2L),NaN));expect_error(nn$AnomalyCorruption(nn$zeros(c(0L,28L,28L,1L)),0L))
  expect_error(nn$AnomalyCorruption(nn$ones(c(1L,28L,28L,1L),nn$DType$Float64),0L));expect_error(nn$AnomalyCorruption(nn$ones(c(1L,28L,28L,1L)),3L))
})
