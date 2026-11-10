library(testthat)
nn <- datamungerNN::nn

test_that("Midpoint uniforms preserve raw words and one final cast", {
 wide <- nn$MidpointUniform(3L,1L,6601L,nn$DType$Float64)
 expect_equal(wide$values(),c(.14499446202535182,.4169559587026015,.9257562280399725),tolerance=0)
 expect_identical(nn$MidpointUniform(3L,1L,6601L)$values(),nn$cast(wide,nn$DType$Float32)$values());expect_false(wide$requires_grad())
 expect_error(nn$MidpointUniform(0L));expect_error(nn$MidpointUniform(2L,0L));expect_error(nn$MidpointUniform(2L,1L,1L,nn$DType$Int64))
})

test_that("Wasserstein Jacobians retain critic paths and stop external penalty paths", {
 t <- nn$DType$Float64;g <- nn$ParameterStore(5501L,t);c <- nn$ParameterStore(5503L,t)
 x <- nn$from_vector_double(c(.2,.7,-.3,.8),c(2L,2L))$requires_grad_();z <- nn$GaussianRejectionPrior(2L,2L,5601L,2,t)$z$requires_grad_();a <- nn$MidpointUniform(2L,1L,6601L,t)$requires_grad_()
 out <- nn$WassersteinGan(g,c,x,z,a);expect_equal(g$count(),82);expect_equal(c$count(),65);expect_equal(g$size(),4);expect_equal(c$size(),4)
 expect_equal(out$input_gradient$values(),out$symbolic_input_gradient$values(),tolerance=1e-14);expect_false(out$interpolation$requires_grad());expect_true(out$symbolic_input_gradient$requires_grad())
 options <- nn$GradOptions();options$allow_unused <- TRUE;stopped <- nn$grad(out$penalty,list(x,z,a,g$at("hidden/W")),options);expect_true(all(vapply(stopped,function(v)!v$defined(),logical(1))))
 out <- nn$WassersteinGan(g,c,x,z,a);options$create_graph <- TRUE;partials <- nn$grad(out$penalty,c$parameters(),options);expect_true(partials[[1]]$requires_grad());expect_false(partials[[4]]$defined());expect_equal(nn$grad(nn$sum(nn$mul(partials[[1]],partials[[1]])),list(c$at("hidden/W")))[[1]]$shape(),c(2,16))
})

test_that("No-grad evaluation, weight zero and owned fields are explicit", {
 t <- nn$DType$Float64;g <- nn$ParameterStore(5501L,t);c <- nn$ParameterStore(5503L,t);x <- nn$ones(c(2L,2L),t);z <- nn$ones(c(2L,2L),t)$requires_grad_();a <- nn$full(c(2L,1L),.5,t)
 out <- nn$no_grad(function()nn$WassersteinGan(g,c,x,z,a,10,FALSE));expect_false(out$fake$requires_grad());expect_false(out$symbolic_input_gradient$requires_grad());expect_false(out$penalty$requires_grad())
 base <- nn$WassersteinGan(g,c,x,z,a,0);gp <- nn$WassersteinGan(g,c,x,z,a,10);expect_equal(base$critic_loss$item(),-base$critic_gap$item(),tolerance=0);expect_equal(base$generator_loss$item(),gp$generator_loss$item(),tolerance=0);expect_equal(base$penalty$item(),gp$penalty$item(),tolerance=0)
 loss <- gp$generator_loss;fake <- gp$fake;rm(base,gp,out,g,c);gc();expect_equal(fake$shape(),c(2,2));expect_true(any(abs(nn$grad(loss,list(z))[[1]]$values())>1e-10))
})

test_that("Both native variants keep three critic writes then one fresh G explicit", {
 for(gp in c(FALSE,TRUE)){
  t <- nn$DType$Float64;g <- nn$ParameterStore(5501L,t);c <- nn$ParameterStore(5503L,t);x <- nn$ones(c(2L,2L),t);nn$WassersteinGan(g,c,x,x,nn$full(c(2L,1L),.5,t));if(!gp)nn$ClipWassersteinCritic(c)
  cfg <- nn$OptimizerConfig();cfg$kind <- if(gp)nn$OptimizerKind$Adam else nn$OptimizerKind$RMSProp;cfg$beta1 <- 0;cfg$beta2 <- .9;cfg$rho <- .9
  rate <- if(gp).0001 else .00005;go <- nn$Optimizer(g,rate,cfg);critic <- nn$Optimizer(c,rate,cfg)
  for(step in 0:2){before <- g$flatten();z <- nn$GaussianRejectionPrior(2L,2L,5601L+step,2,t)$z;a <- nn$MidpointUniform(2L,1L,6601L+step,t);out <- nn$WassersteinGan(g,c,x,z,a,if(gp)10 else 0);critic$step(nn$tensor_vector(nn$grad(out$critic_loss,c$parameters())));if(!gp){nn$ClipWassersteinCritic(c);expect_lte(max(abs(c$flatten())),.01)};expect_identical(g$flatten(),before)}
  before <- c$flatten();z <- nn$GaussianRejectionPrior(2L,2L,7601L,2,t)$z;a <- nn$MidpointUniform(2L,1L,8601L,t);out <- nn$WassersteinGan(g,c,x,z,a,if(gp)10 else 0);go$step(nn$tensor_vector(nn$grad(out$generator_loss,g$parameters())));expect_identical(c$flatten(),before);expect_equal(critic$iterations(),3);expect_equal(go$iterations(),1)
 }
})

test_that("Wasserstein input failures precede leaves and empty clipping is rejected", {
 t <- nn$DType$Float64;g <- nn$ParameterStore(5501L,t);c <- nn$ParameterStore(5503L,t);x <- nn$ones(c(2L,2L),t);a <- nn$full(c(2L,1L),.5,t)
 expect_error(nn$WassersteinGan(g,g,x,x,a));expect_error(nn$WassersteinGan(g,c,x,x,nn$full(c(2L,1L),2,t)));expect_error(nn$WassersteinGan(g,c,x,x,a,-1));expect_error(nn$WassersteinGenerate(g,nn$full(c(2L,2L),NaN,t)));expect_equal(g$size()+c$size(),0);expect_error(nn$ClipWassersteinCritic(c))
})
