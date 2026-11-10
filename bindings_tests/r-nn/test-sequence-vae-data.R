library(testthat)
nn <- datamungerNN::nn

test_that("Native corpus preparation and selected pools own every tensor", {
  cache <- tempfile("sentence-corpus-");on.exit(unlink(cache,recursive=TRUE))
  root <- file.path(nn$prepare_example_data("sequence-vae-assets",cache),"sequence_vae")
  pools <- nn$LoadSequenceVaeData(root);expect_length(pools,3)
  fit <- pools[[1]];expect_equal(fit$tokens$shape(),c(288,7))
  arrays <- nn$SequenceDataArrays(pools);expect_length(arrays,15);expect_false("config" %in% names(arrays))
  rm(pools);gc()
  fields <- lapply(c("tokens","inputs","lengths","factors","source"),function(name)do.call("$",list(fit,name)))
  rm(fit);unlink(cache,recursive=TRUE);gc()
  expect_equal(fields[[1]]$shape(),c(288,7));expect_equal(fields[[5]]$shape(),288)
  expect_true(nn$SequenceGrammarValid(fields[[1]],0L))
  expect_equal(arrays$fit_tokens$values(),fields[[1]]$values())
})

test_that("Vocabulary copies and original prefix grammar semantics are explicit", {
  words <- nn$SequenceVocabulary();expect_length(words,19);expect_equal(words[1:3],c("[pad]","[bos]","[eos]"))
  words[1] <- "changed";expect_equal(nn$SequenceVocabulary()[1],"[pad]")
  tokens <- nn$from_vector_int64(c(3,5,8,11,13,2,999),c(1L,7L))
  expect_equal(nn$SequenceText(tokens,0L),"a red circle moves left [eos]")
  expect_true(nn$SequenceGrammarValid(tokens,0L))
  expect_false(nn$SequenceGrammarValid(nn$zeros(c(1L,7L),nn$DType$Int32),0L))
  expect_error(nn$SequenceText(tokens,1L))
})

test_that("Loader/export errors use the actual native grammar checks", {
  cache <- tempfile("sentence-invalid-");on.exit(unlink(cache,recursive=TRUE))
  root <- file.path(nn$prepare_example_data("sequence-vae-assets",cache),"sequence_vae")
  writeLines("wrong",file.path(root,"vocabulary.txt"))
  expect_error(nn$LoadSequenceVaeData(root));expect_error(nn$SequenceDataArrays(list()))
  expect_error(nn$LoadSequenceVaeData(file.path(root,"absent")))
})
