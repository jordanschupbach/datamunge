test_that("tensor values survive collection of temporary SWIG objects", {
  x <- NnTensor(c(1, 2, 3, 4), as.integer(c(2, 2)), "float64")
  result <- x$transpose()$add(x)
  rm(x)
  gc()
  expect_equal(as.numeric(result$values()), c(2, 5, 5, 8))
  expect_error(result$reshape(as.integer(3)))
})
