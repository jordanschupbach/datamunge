test_that("BSR fits by basis count or knots and RGBA renders translucent", {
  x <- seql(0, 1, 50)
  y <- x^3 - x
  fit <- BSR(x, y, 8)
  expect_equal(fit$basis_size(), 8)
  expect_equal(fit$predict(0.5), 0.125 - 0.5, tolerance = 1e-8)
  expect_equal(BSplineRegression(x, y, seql(0, 1, 5))$basis_size(), 7)

  plot <- RPlot_create()
  plot$points(c(0, 1), c(0, 1), "p", RGBA(1, 2, 3, 0.25), 4)$title("T")$xlab("X")
  expect_equal(plot$title_text(), "T")
  expect_match(plot$to_svg(), 'fill-opacity="0.25"', fixed = TRUE)
  expect_identical(environmentName(environment(rnorm)), "stats") # not masked by datamunger
})

test_that("PSR fits with a fixed lambda or one chosen by GCV", {
  x <- seql(0, 1, 60)
  fit <- PSR(x, 1 - 2 * x, 30, 1e4) # the 2nd-order penalty leaves lines alone
  expect_equal(fit$basis_size(), 30)
  expect_equal(fit$predict(0.25), 0.5, tolerance = 1e-8)

  set.seed(5)
  xs <- runif(200)
  ys <- sin(6 * xs) + rnorm(200, sd = 0.1)
  auto <- PSplineRegression(xs, ys, 40)
  expect_lte(auto$gcv(), PSR(xs, ys, 40, 1e3)$gcv())
  expect_gt(auto$effective_df(), 2)
  expect_equal(PSR(xs, ys, seql(0, 1, 11))$basis_size(), 13)
})
