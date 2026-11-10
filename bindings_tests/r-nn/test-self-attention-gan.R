library(testthat)
nn <- datamungerNN::nn

test_that("Unscaled spectral attention owns fields and starts with an identity gate", {
 p <- nn$ParameterStore(4L);s <- nn$ParameterStore(0L);x <- nn$full(c(2L,3L,8L),.2)$requires_grad_()
 a <- nn$SpectralSpatialAttention(p,s,x);expect_identical(a$input$values(),a$output$values())
 expect_equal(a$probability$shape(),c(2,3,3));before <- s$flatten();nn$SpectralSpatialAttention(p,s,x);expect_identical(s$flatten(),before)
 nn$advance_cached_spectral_state(p,s,16L);p$set("attention/gamma",nn$scalar(.3));a <- nn$SpectralSpatialAttention(p,s,x)
 y <- a$output;probability <- a$probability;rm(a,p,s);gc()
 expect_true(all(is.finite(probability$values())));expect_true(any(abs(nn$grad(nn$sum(y),list(x))[[1]]$values())>1e-8))
 p <- nn$ParameterStore(1L);s <- nn$ParameterStore(0L);expect_equal(nn$cached_spectral_dense(p,s,"x",nn$ones(c(2L,3L)),4L)$shape(),c(2,4));expect_equal(p$count(),16);expect_equal(s$count(),7)
})

test_that("Patch GAN census, unequal pools and immutable forward retain the source", {
 stores <- lapply(c(5903L,5923L,0L,0L),nn$ParameterStore);x <- nn$zeros(c(2L,784L));z <- nn$zeros(c(3L,16L))
 forward <- function()nn$SelfAttentionGan(stores[[1]],stores[[2]],stores[[3]],stores[[4]],x,z)
 a <- forward();expect_equal(vapply(stores,function(p)p$count(),numeric(1)),c(2730,650,245,110));expect_equal(vapply(stores,function(p)p$size(),numeric(1)),c(13,14,12,12))
 expect_equal(a$fake$shape(),c(3,784));expect_equal(a$real_score$shape(),c(2,1));expect_identical(a$g_attention_input$values(),a$g_attention_output$values());expect_identical(a$real_attention_input$values(),a$real_attention_output$values())
 before <- lapply(stores,function(p)p$flatten());forward();expect_identical(lapply(stores,function(p)p$flatten()),before)
})

test_that("Host D-state D G-state fresh-G phases own only their selected stores", {
 stores <- lapply(c(5903L,5923L,0L,0L),nn$ParameterStore);g <- stores[[1]];d <- stores[[2]];gs <- stores[[3]];ds <- stores[[4]]
 x <- nn$full(c(2L,784L),.3);z <- nn$full(c(3L,16L),-.2)
 forward <- function()nn$SelfAttentionGan(g,d,gs,ds,x,z)
 forward();nn$advance_cached_spectral_state(g,gs,16L);nn$advance_cached_spectral_state(d,ds,16L)
 config <- nn$OptimizerConfig();config$kind <- nn$OptimizerKind$Adam;config$beta1 <- .5
 go <- nn$Optimizer(g,.0002,config);disc <- nn$Optimizer(d,.0002,config)
 actions <- list(function()nn$advance_cached_spectral_state(d,ds),function()disc$step(nn$tensor_vector(nn$grad(forward()$discriminator_loss,d$parameters()))),function()nn$advance_cached_spectral_state(g,gs),function()go$step(nn$tensor_vector(nn$grad(forward()$generator_loss,g$parameters()))))
 for(i in seq_along(actions)){owner <- c(4,2,3,1)[i];before <- lapply(stores,function(p)p$flatten());actions[[i]]();after <- lapply(stores,function(p)p$flatten());expect_false(identical(before[[owner]],after[[owner]]));expect_identical(before[-owner],after[-owner])}
 expect_equal(go$iterations(),1);expect_equal(disc$iterations(),1)
})

test_that("All record kinds retain storage after parent collection and inputs validate early", {
 t <- nn$DType$Float64;g <- nn$ParameterStore(5903L,t);d <- nn$ParameterStore(5923L,t);gs <- nn$ParameterStore(0L,t);ds <- nn$ParameterStore(0L,t)
 z <- nn$full(c(2L,16L),.2,t)$requires_grad_();x <- nn$full(c(2L,784L),.3,t)
 a <- nn$SelfAttentionGanGenerate(g,gs,z);b <- nn$SelfAttentionGanDiscriminate(d,ds,x)
 nn$advance_cached_spectral_state(g,gs,16L);nn$advance_cached_spectral_state(d,ds,16L);full <- nn$SelfAttentionGan(g,d,gs,ds,x,z)
 pixels <- a$pixels;score <- b$score;loss <- full$generator_loss;rm(a,b,full,g,d,gs,ds);gc()
 expect_equal(pixels$shape(),c(2,784));expect_equal(score$shape(),c(2,1));expect_true(any(abs(nn$grad(loss,list(z))[[1]]$values())>1e-10))
 g <- nn$ParameterStore(1L);d <- nn$ParameterStore(2L);gs <- nn$ParameterStore(0L);ds <- nn$ParameterStore(0L)
 x <- nn$zeros(c(2L,784L));z <- nn$zeros(c(3L,16L))
 expect_error(nn$SelfAttentionGan(g,g,gs,ds,x,z));expect_error(nn$SelfAttentionGan(g,d,gs,gs,x,z));expect_error(nn$SelfAttentionGan(g,d,gs,ds,x,nn$zeros(c(3L,15L))))
 expect_error(nn$SpectralSpatialAttention(g,gs,nn$ones(c(2L,3L,8L)),0L));expect_error(nn$cached_spectral_dense(g,gs,"",nn$ones(c(2L,3L)),4L));expect_equal(g$size()+d$size()+gs$size()+ds$size(),0)
})
