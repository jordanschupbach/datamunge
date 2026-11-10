library(testthat)
nn <- datamungerNN::nn
fg_bytes <- c(10,30,10,1,99,18,5,67,111,110,115,116,42,18,10,5,118,97,108,117,101,18,9,66,7,8,1,45,0,0,0,64)
fg_vector <- function(x) {
  as.integer(x)
}

test_that("byte and file models own exact scalar constants and metadata", {
  g <- nn$FrozenGraph(fg_vector(fg_bytes))
  expect_equal(g$node_names(),"c"); expect_equal(g$operation("c:0"),"Const")
  expect_equal(g$constant("c")$dim(),0); expect_equal(g$constant("c")$item(),2)
  file <- tempfile(fileext=".pb"); on.exit(unlink(file))
  writeBin(as.raw(fg_bytes),file)
  loaded <- nn$FrozenGraph_load(file); unlink(file); gc()
  expect_equal(loaded$constant("c")$item(),2)
})
test_that("named detached graph results own storage after model and feeds expire", {
  g <- nn$FrozenGraph(fg_vector(fg_bytes)); feed <- nn$full(integer(),7)$requires_grad_()
  out <- nn$named_list(g$evaluate(nn$named_tensors(list(c=feed)),c("c:0","c")))
  expect_equal(names(out),c("c:0","c"))
  first <- out[[1]]; second <- out[[2]]; constant <- g$constant("c")
  rm(out,g,feed); gc()
  expect_equal(first$item(),7); expect_equal(second$item(),7); expect_equal(constant$item(),2)
  expect_false(first$requires_grad()); expect_false(constant$requires_grad())
})
test_that("bad graph bytes bounds ports and unknown endpoints reject", {
  expect_error(nn$FrozenGraph(fg_vector(c(10,255))))
  expect_error(nn$FrozenGraph(fg_vector(fg_bytes),2))
  g <- nn$FrozenGraph(fg_vector(fg_bytes))
  expect_error(g$operation("c:1")); expect_error(g$constant("unknown"))
  expect_equal(length(nn$named_list(g$evaluate(nn$named_tensors(list()),character()))),0)
})
