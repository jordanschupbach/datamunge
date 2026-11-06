library(testthat)
library(datamunger)

test_that("dense and CSR expressions agree with native direct QR", {
  for (constructor in list(Mat, SpMat)) {
    x <- constructor(matrix(c(1.1, 2.2, 3.3, 4.4, 5.5, 6.6), ncol = 2, byrow = TRUE))
    y <- constructor(matrix(c(1.2, 3.3, 4.4), ncol = 1))
    beta <- solve(t(x) %*% x) %*% t(x) %*% y
    expect_equal(dim(beta), c(2L, 1L))
    expect_equal(as.numeric(as.matrix(beta)), c(7/33, 17/33), tolerance = 1e-10)
    expect_equal(as.matrix(solve(x, y)), as.matrix(beta), tolerance = 1e-10)
    expect_identical(class(eval_mat(beta)), "Mat")
    expect_match(paste(capture.output(print(beta)), collapse = "\n"), "0.2121212", fixed = TRUE)
  }
})

test_that("temporary operands survive collection and snapshots remain valid", {
  expression <- solve(Mat(diag(c(2, 4)))) %*% SpMat(matrix(c(6, 8), ncol = 1))
  gc()
  snapshot <- eval_mat(expression)
  rm(expression)
  gc()
  expect_equal(as.matrix(snapshot), matrix(c(3, 2), ncol = 1))
})

test_that("factorisation is deferred and errors propagate through R", {
  expression <- solve(Mat(matrix(0))) %*% Mat(matrix(1))
  expect_equal(dim(expression), c(1L, 1L))
  expect_error(eval_mat(expression), "singular")
  expect_error(Mat(matrix(c(1, 2), nrow = 1)) %*% Mat(matrix(c(1, 2), nrow = 1)), "dimensions")
  expect_error(solve(Mat(matrix(c(1, 2), nrow = 1))), "square")
})

test_that("CSR zeros and fused multiple-right-hand-side products are preserved", {
  a <- SpMat(diag(c(2, 4)))
  expect_s3_class(t(a) %*% a, "SpMat")
  expect_match(paste(capture.output(print(a)), collapse = "\n"), "nnz=2", fixed = TRUE)
  b <- Mat(matrix(1:6, nrow = 2, byrow = TRUE))
  c <- Mat(matrix(c(1, 0, 1), ncol = 1))
  expect_equal(as.matrix(solve(a) %*% b %*% c), matrix(c(2, 2.5), ncol = 1))
})
