library(testthat)
nn <- datamungerNN::nn

test_that("Selected patch values retain signed zeros and own lifetime", {
  x <- nn$from_vector_double(c(-0.,.2,.3,.4,-0.,.6,.7,.8),c(2L,2L,2L));p <- nn$full(c(2L,2L,2L),.9,nn$DType$Float64);h <- nn$from_vector_double(c(0,1,1,0),c(2L,2L))
  visible <- nn$VisiblePatchValues(x,h,-0.);v <- visible$values();expect_true(all(is.infinite(1/v[c(1,3,5)])));expect_true(all(1/v[c(1,3,5)]<0))
  o <- nn$CompositePatchValues(x,p,h);expect_equal(o$values()[-1],c(.2,.9,.9,.9,.9,.7,.8),tolerance=0);expect_lt(1/o$values()[1],0)
  o$fill_(17);expect_equal(x$values()[-1],c(.2,.3,.4,0,.6,.7,.8),tolerance=0)
  kept <- nn$CompositePatchValues(x,p,h);rm(x,p,h);gc();expect_equal(kept$shape(),c(2,2,2));expect_equal(kept$values()[2],.2)
})

test_that("Patch derivatives reach only selected branches", {
  t <- nn$DType$Float64;x <- nn$full(c(2L,2L,3L),.2,t)$requires_grad_();p <- nn$full(c(2L,2L,3L),.7,t)$requires_grad_();h <- nn$from_vector_double(c(0,1,1,0),c(2L,2L))
  g <- nn$grad(nn$sum(nn$CompositePatchValues(x,p,h)),nn$tensor_vector(list(x,p)))
  expect_equal(g[[1]]$values(),c(1,1,1,0,0,0,0,0,0,1,1,1),tolerance=0);expect_equal(g[[2]]$values(),c(0,0,0,1,1,1,1,1,1,0,0,0),tolerance=0)
  v <- nn$grad(nn$sum(nn$VisiblePatchValues(x,h,9)),nn$tensor_vector(list(x)))[[1]];expect_equal(v$values(),g[[1]]$values(),tolerance=0)
})

test_that("Generic strides all-visible and all-hidden masks are accepted", {
  t <- nn$DType$Float64;x <- nn$transpose(nn$full(c(4L,3L,2L),-3,t),c(2L,1L,0L));p <- nn$full(c(2L,3L,4L),7,t);h <- nn$transpose(nn$ones(c(3L,2L),t))
  expect_false(x$is_contiguous());expect_false(h$is_contiguous());expect_equal(nn$CompositePatchValues(x,p,h)$values(),p$values(),tolerance=0)
  expect_equal(nn$CompositePatchValues(x,p,nn$zeros(c(2L,3L),t))$values(),x$values(),tolerance=0)
  expect_equal(nn$VisiblePatchValues(x,h,-7)$values(),rep(-7,24),tolerance=0)
  expect_equal(nn$VisiblePatchValues(nn$ones(c(1L,1L,1L)),nn$ones(c(1L,1L)),1e-100)$item(),0)
})

test_that("Invalid masks fill and even unselected nonfinite values fail", {
  x <- nn$zeros(c(2L,3L,4L))
  for(h in list(nn$zeros(c(1L,3L)),nn$full(c(2L,3L),.5),nn$zeros(c(2L,3L),nn$DType$Float64),nn$zeros(c(2L,3L),nn$DType$Bool),nn$ones(c(2L,3L))$requires_grad_())) {
    expect_error(nn$VisiblePatchValues(x,h));expect_error(nn$CompositePatchValues(x,x,h))
  }
  for(fill in c(Inf,NaN,1e100))expect_error(nn$VisiblePatchValues(x,nn$zeros(c(2L,3L)),fill))
  expect_error(nn$CompositePatchValues(x,nn$full(c(2L,3L,4L),NaN),nn$zeros(c(2L,3L))))
  expect_error(nn$VisiblePatchValues(nn$zeros(c(2L,3L)),nn$zeros(c(2L,3L))))
})
