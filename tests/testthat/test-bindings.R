test_that("hello() is callable", {
  expect_true(exists("hello", where = asNamespace("datamunger")))
  expect_silent(datamunger::hello())
})

test_that("STL templates are usable", {
  expect_true(exists("DVector", where = asNamespace("datamunger")))
  expect_true(exists("DPair", where = asNamespace("datamunger")))
})
