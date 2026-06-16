test_that("hello() is callable", {
  expect_true(exists("hello", where = asNamespace("octrar")))
  expect_silent(octrar::hello())
})

test_that("STL templates are usable", {
  expect_true(exists("DVector", where = asNamespace("octrar")))
  expect_true(exists("DPair", where = asNamespace("octrar")))
})
