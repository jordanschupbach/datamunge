library(testthat)
nn <- datamungerNN::nn
sig_inputs <- function() {
  t <- nn$DType$Float64
  list(nn$ParameterStore(7307L,t),nn$ParameterStore(7309L,t),
       nn$full(c(2L,5L,7L,3L),.3,t),nn$full(c(3L,5L,7L,3L),-.2,t),
       nn$SingleImageNoise(3L,5L,7L,FALSE,7411L,t),
       nn$full(c(2L,5L,7L,3L),.1,t),nn$SingleImageNoise(2L,5L,7L,TRUE,7411L,t),
       nn$full(1L,.2,t))
}
sig_diff <- function(a,b) nn$max(nn$abs(nn$sub(a,b)))$item()

test_that("one Gaussian stream tiles only coarse RGB and RMS amplitude is detached", {
  for(shared in c(FALSE,TRUE)) {
    z <- nn$SingleImageNoise(2L,3L,5L,shared,7411L)
    expected <- nn$cast(nn$GaussianNoise(c(3L,5L,if(shared)1L else 3L),2L,7411L),nn$DType$Float32)
    if(shared) expected <- nn$tile(expected,c(1L,1L,1L,3L))
    expect_true(nn$equal_bits(z,expected)); expect_false(z$requires_grad())
  }
  x <- nn$full(c(1L,3L,5L,3L),-2,nn$DType$Float64)$requires_grad_()
  p <- nn$ones(x$shape(),x$dtype())$requires_grad_()
  a <- nn$SingleImageAmplitude(x,p)
  expect_equal(a$shape(),1); expect_equal(a$item(),.3)
  expect_false(a$requires_grad()); expect_equal(nn$SingleImageAmplitude(x,p,0)$item(),0)
  expect_error(nn$SingleImageNoise(0L,2L,3L,TRUE,1L))
  expect_error(nn$SingleImageAmplitude(x,p,-1))
})
test_that("source census supports unequal rectangular pools and exact objective scales", {
  q <- sig_inputs(); o <- do.call(nn$SingleImageGan,q)
  expect_equal(q[[1]]$count(),371); expect_equal(q[[2]]$count(),297)
  expect_equal(q[[1]]$size(),6); expect_equal(q[[2]]$size(),6)
  expect_equal(o$fake$shape(),c(3,5,7,3)); expect_equal(o$real_score$shape(),c(2,5,7,1))
  expect_equal(o$g_loss$item(),o$adversarial_loss$item()+10*o$reconstruction_loss$item())
  expect_equal(o$d_loss$item(),.5*(nn$mean(nn$square(nn$sub(o$real_score,1)))$item()+nn$mean(nn$square(o$fake_score))$item()))
  for(store in q[1:2]) for(name in store$names()) store$set(name,nn$zeros(store$at(name)$shape(),q[[3]]$dtype()))
  high <- nn$full(q[[4]]$shape(),4,q[[3]]$dtype())
  expect_true(nn$equal_bits(nn$SingleImageGenerate(q[[1]],high,q[[5]],q[[8]])$image,high))
})
test_that("D then fresh G retains update owners and independent clocks", {
  q <- sig_inputs(); o <- do.call(nn$SingleImageGan,q)
  keep <- nn$GradOptions(); keep$retain_graph <- TRUE
  stale <- nn$grad(o$g_loss,q[[1]]$parameters(),keep); dd <- nn$grad(o$d_loss,q[[2]]$parameters())
  cfg <- nn$OptimizerConfig(); cfg$kind <- nn$OptimizerKind$Adam; cfg$beta1 <- .5
  go <- nn$Optimizer(q[[1]],.0005,cfg); do_ <- nn$Optimizer(q[[2]],.0005,cfg)
  before <- q[[1]]$flatten(); do_$step(nn$tensor_vector(dd)); expect_identical(q[[1]]$flatten(),before)
  after <- q[[2]]$flatten(); fresh <- nn$grad(do.call(nn$SingleImageGan,q)$g_loss,q[[1]]$parameters())
  expect_gt(max(vapply(seq_along(fresh),function(i)sig_diff(stale[[i]],fresh[[i]]),numeric(1))),1e-9)
  go$step(nn$tensor_vector(fresh)); expect_identical(q[[2]]$flatten(),after)
  expect_equal(go$iterations(),1); expect_equal(do_$iterations(),1)
  expect_equal(nn$Optimizer(q[[1]],.0005,cfg)$iterations(),0)
})
test_that("owning fields retain higher paths after parent and stores are collected", {
  t <- nn$DType$Float64
  x <- nn$full(c(1L,3L,5L,3L),.2,t)$requires_grad_()
  z <- nn$SingleImageNoise(1L,3L,5L,FALSE,7411L,t)$requires_grad_()
  a <- nn$full(integer(),.3,t)$requires_grad_()
  g <- nn$ParameterStore(7307L,t); d <- nn$ParameterStore(7309L,t)
  o <- nn$SingleImageGan(g,d,x,x,z,x,z,a)
  image <- o$fake; loss <- o$g_loss; score <- o$fake_score
  rm(o,g,d); gc()
  graph <- nn$GradOptions(); graph$create_graph <- TRUE
  first <- nn$grad(nn$add(loss,nn$mean(nn$square(image))),list(x,z,a),graph)
  expect_true(all(vapply(first,function(v)nn$sum(nn$abs(v))$item()>0,logical(1))))
  expect_gt(nn$max(nn$abs(nn$grad(nn$sum(nn$square(first[[1]])),list(z))[[1]]))$item(),1e-10)
  expect_equal(score$shape(),c(1,3,5,1))
})
test_that("invalid aliases geometry and amplitude fail before parameter creation", {
  q <- sig_inputs()
  expect_error(do.call(nn$SingleImageGan,c(q[1],q[1],q[3:8])))
  expect_error(nn$SingleImageGenerate(q[[1]],q[[4]],q[[5]],nn$ones(c(1L,1L),q[[3]]$dtype())))
  expect_error(nn$SingleImageDiscriminate(q[[2]],nn$full(q[[3]]$shape(),NaN,q[[3]]$dtype())))
  expect_equal(q[[1]]$size(),0); expect_equal(q[[2]]$size(),0)
})
