test_that("hierarchical fits subtract ancestors and stay local", {
  bases <- BSplineBasisVector()
  bases$push_back(BSplineBasis_open_uniform(1L, 2L))
  model <- HierarchicalBSpline(c(0), c(1))
  model$fit_cell(0L, bases, list(c(0), c(0.5), c(1)), c(1, 2, 3))
  child <- model$refine(0L, 1L)
  # Raw observations equal parent + 2; the patch must fit only the residual.
  model$fit_cell(child, bases, list(c(0.5), c(0.75), c(1)), c(4, 4.5, 5))
  rm(bases)
  gc()
  expect_equal(model$evaluate(c(0.25)), 1.5, tolerance = 1e-10)
  expect_equal(model$evaluate(c(0.75)), 4.5, tolerance = 1e-10)
  expect_equal(model$cell_min(child), c(0.5))
  expect_equal(model$cell_max(child), c(1))
  expect_error(model$fit_cell(child, BSplineBasisVector(), list(c(0.75)), c(0)),
               "patch|spline")
  expect_error(model$evaluate(c(1.5)), "outside")
})

test_that("penalized hierarchy preserves constants and rejects negative penalties", {
  bases <- BSplineBasisVector()
  bases$push_back(BSplineBasis_open_uniform(3L, 6L))
  model <- HierarchicalBSpline(c(0), c(1))
  model$fit_cell_penalized(0L, bases, list(c(0), c(1)), c(2, 2), c(100), 2L)
  child <- model$refine(0L, 1L)
  model$fit_cell_penalized(child, bases, list(c(0.5), c(1)), c(3, 3), c(100), 2L)
  expect_equal(model$evaluate(c(0.25)), 2, tolerance = 1e-9)
  expect_equal(model$evaluate(c(0.75)), 3, tolerance = 1e-9)
  other <- model$refine(0L, 0L)
  expect_error(model$fit_cell_penalized(other, bases, list(c(0), c(0.5)),
                                      c(2, 2), c(-1), 2L), "lambda")
})

test_that("tensor and truncated hierarchical spline objects can be fitted", {
  basis <- BSplineBasis_open_uniform(1L, 2L)
  tensor <- TensorBSpline(basis, c(0, 0))
  model <- TruncatedHierarchicalBSpline(tensor)
  model$fit(list(c(0), c(0.5), c(1)), c(1, 2, 3))
  rm(tensor, basis)
  gc()
  expect_equal(model$evaluate(c(0.25)), 1.5, tolerance = 1e-10)
})
