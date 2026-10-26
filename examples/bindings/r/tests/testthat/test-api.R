test_that("the catalogue and native fixtures work", {
  expect_length(nn_example_names(), 165L)
  expect_setequal(nn_example_names(), as.character(datamungerNN:::native_nn_example_names()))
  expect_identical(nn_example_run("convolution_im2col"), 0L)
  expect_error(nn_example_run("missing"), "unknown NN example")
})
