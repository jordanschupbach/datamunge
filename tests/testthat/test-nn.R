# datamunge::nn binding surface: NnTensor and NnModel.

test_that("NnTensor creates, computes and differentiates", {
  a <- NnTensor(c(1, 2, 3, 4, 5, 6), as.integer(c(2, 3)), "float32")
  expect_equal(as.numeric(a$shape()), c(2, 3))
  expect_equal(a$dtype(), "float32")
  expect_equal(a$sum()$item(), 21)
  expect_equal(as.numeric(a$matmul(a, FALSE, TRUE)$values()), c(14, 32, 32, 77))

  x <- NnTensor(c(1, 2, 3), as.integer(3), "float64")
  x$set_requires_grad(TRUE)
  x$square()$sum()$backward()
  expect_equal(as.numeric(x$grad()$values()), c(2, 4, 6))
})

test_that("NnModel trains a linear model and round-trips weights", {
  x <- NnTensor_uniform(as.integer(c(64, 2)), 3, -1, 1)
  y <- x$matmul(NnTensor(c(2, -1), as.integer(c(2, 1)), "float32"))$add_scalar(0.5)
  model <- NnModel(11)
  model$add_dense(1L)
  model$set_optimizer("adam", 0.05)
  first <- model$train_step(x, y, "mse")
  for (i in 1:300) last <- model$train_step(x, y, "mse")
  expect_lt(last, 1e-3)
  expect_lt(last, first)
  expect_equal(model$parameter_count(), 3)

  path <- tempfile(fileext = ".npz")
  model$save(path)
  copy <- NnModel()
  copy$add_dense(1L)
  copy$load(path)
  expect_equal(as.numeric(copy$predict(x)$values()), as.numeric(model$predict(x)$values()))
})
