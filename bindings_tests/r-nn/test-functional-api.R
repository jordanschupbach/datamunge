testthat::test_that("R composes a functional model and owns its tensors", {
  p<-nn$ParameterStore(9L);x<-nn$from_vector(c(.1,.8,.7,.2),c(2L,2L))
  code<-nn$dense(p,"encoder",x,1L,nn$Activation$Relu)
  y<-nn$dense(p,"decoder",code,2L,nn$Activation$Sigmoid)
  loss<-nn$mean(nn$square(nn$sub(y,x)));gradients<-nn$grad(loss,p$parameters())
  testthat::expect_equal(p$names(),c("encoder/W","encoder/b","decoder/W","decoder/b"))
  testthat::expect_length(gradients,4L);testthat::expect_equal(p$count(),7)
  gc()
  testthat::expect_true(all(vapply(gradients,function(g)all(is.finite(g$values())),logical(1))))
  config<-nn$OptimizerConfig();config$kind<-nn$OptimizerKind$Adam
  optimizer<-nn$Optimizer(p,.003,config);before<-p$flatten()
  optimizer$step(nn$tensor_vector(gradients))
  testthat::expect_false(identical(p$flatten(),before))
  file<-tempfile(fileext=".npz");p$save(file);arrays<-nn$load_npz(file);gc()
  testthat::expect_equal(names(arrays),p$names())
  testthat::expect_equal(arrays[[1L]]$shape(),c(2,1))
  copy<-tempfile(fileext=".npz");nn$save_npz(copy,arrays)
  testthat::expect_identical(readBin(file,"raw",file.info(file)$size),readBin(copy,"raw",file.info(copy)$size))
})
testthat::test_that("R grad mode and fixed-size convolution options work", {
  before<-nn$is_grad_enabled()
  testthat::expect_error(nn$no_grad(function(){stopifnot(!nn$is_grad_enabled());stop("fixture")}),"fixture")
  testthat::expect_identical(nn$is_grad_enabled(),before)
  options<-nn$Conv2DOptions();options$strides<-c(2,1);options$dilations<-c(1,2)
  output<-nn$conv2d(nn$ones(c(1L,5L,6L,1L)),nn$ones(c(2L,2L,1L,3L)),options)
  testthat::expect_equal(output$shape(),c(1,3,6,3))
})
testthat::test_that("native R example code uses operations, not compiled entry points", {
  root<-normalizePath(testthat::test_path("../.."))
  old<-setwd(root);on.exit(setwd(old))
  environment<-new.env()
  source("examples/r/nn/2_autoencoders/2.1_deterministic_autoencoders/undercomplete_autoencoder.R",local=environment)
  environment$undercomplete_checks()
  source("examples/r/nn/1_convolutional_neural_networks/1.1_fundamentals/convolution_im2col.R",local=environment)
  testthat::expect_identical(environment$main(character()),0L)
  source("examples/r/nn/1_convolutional_neural_networks/1.1_fundamentals/cnn_mnist.R",local=environment)
  p<-nn$ParameterStore(17L);logits<-environment$cnn_mnist_logits(p,nn$zeros(c(2L,784L)))
  testthat::expect_equal(logits$shape(),c(2,10));testthat::expect_equal(p$count(),105866)
})

testthat::test_that("temporary fluent tensors and integer scalar operands are safe", {
  leaf <- nn$zeros(3L)$requires_grad_()
  gc()
  testthat::expect_true(leaf$defined())
  testthat::expect_equal(nn$grad(nn$sum(nn$square(leaf)),list(leaf))[[1L]]$values(),c(0,0,0))
  testthat::expect_equal(nn$mul(nn$ones(3L),2L)$values(),c(2,2,2))
  testthat::expect_equal(nn$div(nn$full(3L,6L),2L)$values(),c(3,3,3))
  original <- nn$zeros(3L)
  alias <- original$fill_(4L)
  testthat::expect_true(alias$is_same(original))
  rm(original);gc()
  testthat::expect_equal(alias$values(),c(4,4,4))
})
testthat::test_that("confusion matrices accumulate every batch", {
  confusion <- nn$ConfusionMatrix(3L)
  confusion$add(nn$from_vector_int64(c(0L,1L,2L)),nn$from_vector_int64(c(0L,2L,2L)))
  confusion$add(1L,1L)
  testthat::expect_equal(confusion$to_tensor()$values(),c(1,0,0,0,1,1,0,0,1))
  testthat::expect_equal(confusion$accuracy(),.75)
})

testthat::test_that("clipping pairs expose gradients and norm across collection", {
  result <- nn$clip_by_global_norm(nn$tensor_vector(list(nn$from_vector(c(3,4)))),2)
  gradients <- nn$tensor_list(result$first)
  gc()
  testthat::expect_equal(result$second,5)
  testthat::expect_equal(gradients[[1L]]$values(),c(1.2,1.6),tolerance=1e-6)
})

testthat::test_that("schedule namespace uses completed update indices", {
  testthat::expect_equal(nn$schedule$step_decay(.1,.5,10L,9L),.1)
  testthat::expect_equal(nn$schedule$step_decay(.1,.5,10L,10L),.05)
  testthat::expect_equal(nn$schedule$warmup(.1,10L,0L),0)
  testthat::expect_error(nn$schedule$cosine(.1,0,0L,1L),"duration")
})

testthat::test_that("monitor decisions own their values and weighted means reset", {
  monitor <- nn$EarlyStopping(2L,.1)
  decision <- monitor$observe(1,0L)
  rm(monitor); gc()
  testthat::expect_true(decision$improved)
  testthat::expect_false(decision$stop)
  metric <- nn$WeightedMean()
  metric$add(2,4L); metric$add(10,1L)
  testthat::expect_equal(metric$count(),5L)
  testthat::expect_equal(metric$value(),3.6)
  metric$reset()
  testthat::expect_equal(metric$count(),0L)
})

testthat::test_that("the standard 64-bit stream and shuffle have independent state", {
  first <- nn$Mt19937_64(113L); second <- nn$Mt19937_64(113L)
  testthat::expect_identical(first$normal64(),second$normal64())
  rows <- first$shuffle(0L:63L)
  testthat::expect_identical(rows,second$shuffle(0L:63L))
  testthat::expect_equal(sort(rows),0L:63L)
  testthat::expect_identical(first$modulo(11L),second$modulo(11L))
  testthat::expect_error(first$modulo(0L),"positive")
})

testthat::test_that("bulk normals preserve odd-count scalar caches", {
  for (constructor in list(nn$Mt19937,nn$Mt19937_64)) {
    for (precision in c(32L,64L)) {
      bulk <- constructor(113L);scalar <- constructor(113L)
      draws <- if(precision==32L) bulk$normal32_values else bulk$normal64_values
      one <- if(precision==32L) scalar$normal32 else scalar$normal64
      for (entry in list(c(3,1.25,.75),c(0,2,3),c(5,-.2,.4),c(2,0,1))) {
        expected <- vapply(seq_len(entry[1L]),function(i) one(entry[2L],entry[3L]),numeric(1))
        testthat::expect_identical(draws(as.integer(entry[1L]),entry[2L],entry[3L]),expected)
        testthat::expect_identical(bulk$uniform64(),scalar$uniform64())
      }
      testthat::expect_error(draws(-1L),"nonnegative")
    }
  }
})

testthat::test_that("attention masks isolate all keys in plain and fused modes", {
  for (implementation in c(nn$AttentionImpl$Plain,nn$AttentionImpl$Fused)) {
    q <- nn$ones(c(1L,2L,3L),nn$DType$Float64)$requires_grad_()
    k <- nn$ones(c(1L,4L,3L),nn$DType$Float64)$requires_grad_()
    v <- nn$ones(c(1L,4L,2L),nn$DType$Float64)$requires_grad_()
    options <- nn$AttentionOptions();options$mask <- nn$zeros(c(2L,4L),nn$DType$Bool);options$impl <- implementation
    attended <- nn$scaled_dot_product_attention(q,k,v,options)
    testthat::expect_equal(attended$output$values(),rep(0,4L))
    gradients <- nn$grad(nn$sum(attended$output),list(q,k,v))
    testthat::expect_true(all(vapply(gradients,function(g) all(g$values()==0),logical(1))))
  }
})

testthat::test_that("cache tensor pairs keep their storage across reset", {
  cache <- nn$KvCache()
  result <- nn$no_grad(function() cache$append(nn$ones(c(1L,1L,2L,3L)),nn$full(c(1L,1L,2L,3L),7)))
  keys <- result$first;values <- result$second
  cache$reset();rm(cache,result);gc()
  testthat::expect_equal(keys$values(),rep(1,6L))
  testthat::expect_equal(values$values(),rep(7,6L))
})

testthat::test_that("attention rejects a zero key width before scaling", {
  testthat::expect_error(nn$scaled_dot_product_attention(nn$zeros(c(1L,1L,0L)),
    nn$zeros(c(1L,1L,0L)),nn$ones(c(1L,1L,1L)),nn$Tensor()),"key width")
})
