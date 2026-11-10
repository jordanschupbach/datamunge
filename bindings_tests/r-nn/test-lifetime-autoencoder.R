testthat::test_that("lifetime training depends on companions while deployment is row-wise", {
    nn <- datamungerNN::nn
    p <- nn$ParameterStore(1931L);sample <- nn$ones(c(1L,3L))
    nn$CompetitiveAutoencoder(p,sample,TRUE,FALSE,3L,2L)
    p$set("encoder/W",nn$full(c(3L,2L),.2));p$set("encoder/b",nn$full(2L,.1))
    low <- nn$CompetitiveAutoencoder(p,nn$concat(nn$tensor_vector(list(sample,nn$zeros(c(7L,3L)))),0L),TRUE,FALSE,3L,2L)
    high <- nn$CompetitiveAutoencoder(p,nn$concat(nn$tensor_vector(list(sample,nn$full(c(7L,3L),2))),0L),TRUE,FALSE,3L,2L)
    testthat::expect_equal(nn$take_rows(low$mask,0L)$values(),c(1,1))
    testthat::expect_equal(nn$take_rows(high$mask,0L)$values(),c(0,0))
    testthat::expect_equal(nn$take_rows(low$code,0L)$values(),nn$take_rows(high$code,0L)$values())
    testthat::expect_equal(nn$take_rows(low$reconstruction,0L)$values(),nn$take_rows(high$reconstruction,0L)$values())
    testthat::expect_false(identical(nn$take_rows(low$training_reconstruction,0L)$values(),nn$take_rows(high$training_reconstruction,0L)$values()))
    testthat::expect_equal(sum(low$code$values()>0),16)
    testthat::expect_equal(sum(low$mask$values()),2)
    testthat::expect_false(low$mask$requires_grad())
    reconstruction <- low$training_reconstruction;code <- low$code;expected <- code$values()
    rm(low,high);invisible(gc())
    testthat::expect_equal(code$values(),expected)
    gradients <- nn$grad(nn$sum(reconstruction),p$parameters())
    testthat::expect_length(gradients,4L)
    testthat::expect_true(all(vapply(gradients,function(g)any(g$values()!=0),logical(1))))
})
testthat::test_that("zero winners count as selected slots and partial batches use ceiling", {
    nn <- datamungerNN::nn
    for(batch in c(1L,7L,8L,9L,32L)) {
        scores <- nn$zeros(c(batch,2L))$requires_grad_();mask <- nn$CompetitiveMask(scores,TRUE)
        keep <- ceiling(batch/8)
        testthat::expect_equal(sum(mask$values()),2*keep)
        testthat::expect_equal(mask$values(),rep(as.numeric(seq_len(batch)<=keep),each=2L))
        testthat::expect_true(all(nn$mul(scores,mask)$values()==0))
        testthat::expect_equal(nn$grad(nn$sum(nn$mul(scores,mask)),list(scores))[[1L]]$values(),mask$values())
    }
})
