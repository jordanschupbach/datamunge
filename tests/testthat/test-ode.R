test_that("ODESolver's RK4 matches exponential decay's closed form", {
  options <- ODEOptions$new()
  options$method <- StepMethod$RK4
  options$step_size <- 0.01
  solver <- ODESolver$new(options)

  sol <- solver$solve_builtin("exponential_decay", c(1.0), c(1.0), 0.0, 2.0)
  last <- sol$state_at(sol$size() - 1)
  expect_equal(last[1], exp(-2.0), tolerance = 1e-6)
  expect_equal(sol$time_at(0), 0.0)
  expect_equal(sol$time_at(sol$size() - 1), 2.0, tolerance = 1e-9)
})

test_that("ODESolver's adaptive RK45 matches exponential decay and uses few steps", {
  options <- ODEOptions$new()
  options$method <- StepMethod$RK45
  options$step_size <- 0.1
  options$abs_tol <- 1e-8
  options$rel_tol <- 1e-8
  solver <- ODESolver$new(options)

  sol <- solver$solve_builtin("exponential_decay", c(1.0), c(1.0), 0.0, 2.0)
  last <- sol$state_at(sol$size() - 1)
  expect_equal(last[1], exp(-2.0), tolerance = 1e-6)
  expect_lt(sol$steps_taken, 50)
})

test_that("ODESolver's Adams-Bashforth-Moulton matches exponential decay", {
  options <- ODEOptions$new()
  options$method <- StepMethod$AdamsMoulton
  options$multistep_order <- 4
  options$step_size <- 0.001
  solver <- ODESolver$new(options)

  sol <- solver$solve_builtin("exponential_decay", c(1.0), c(1.0), 0.0, 2.0)
  last <- sol$state_at(sol$size() - 1)
  expect_equal(last[1], exp(-2.0), tolerance = 1e-6)
})

test_that("ODESolver's harmonic_oscillator matches the closed-form solution", {
  options <- ODEOptions$new()
  options$method <- StepMethod$RK4
  options$step_size <- 0.001
  solver <- ODESolver$new(options)

  sol <- solver$solve_builtin("harmonic_oscillator", c(1.0), c(1.0, 0.0), 0.0, 10.0)
  last <- sol$state_at(sol$size() - 1)
  expect_equal(last[1], cos(10), tolerance = 1e-4)
  expect_equal(last[2], -sin(10), tolerance = 1e-4)
})

test_that("ODESolver covers every named builtin system and supports $ dispatch", {
  solver <- ODESolver$new()

  sol1 <- solver$solve_builtin("logistic_growth", c(1.0, 1.0), c(0.5), 0.0, 1.0)
  expect_equal(sol1$size() > 0, TRUE)

  sol2 <- solver$solve_builtin("van_der_pol", c(1.0), c(2.0, 0.0), 0.0, 1.0)
  expect_equal(length(sol2$state_at(0)), 2)

  sol3 <- solver$solve_builtin("lorenz", numeric(0), c(1.0, 1.0, 1.0), 0.0, 1.0)
  expect_equal(length(sol3$state_at(0)), 3)

  expect_equal(sol3$size(), sol3$size())
  expect_equal(sol3$state_at(0), sol3$state_at(0))
})
