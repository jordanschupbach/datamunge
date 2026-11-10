library(testthat)
nn<-datamungerNN::nn
star_inputs<-function(){t<-nn$DType$Float64;list(g=nn$ParameterStore(6301L,t),d=nn$ParameterStore(6311L,t),x=nn$full(c(2L,28L,28L,1L),.3,t),o=nn$zeros(2L,nn$DType$Int64),t=nn$ones(2L,nn$DType$Int64))}
star_forward<-function(q)nn$StarGan(q$g,q$d,q$x,q$o,q$t)
star_diff<-function(a,b)nn$max(nn$abs(nn$sub(a,b)))$item()

test_that("Three native domains and disjoint observed-order selection preserve originals and unused IDs",{
 x<-nn$from_vector_double(c(0,.25,.5,.75),c(1L,2L,2L,1L))$requires_grad_();y<-nn$StarDomainImages(x,nn$full(1L,2,nn$DType$Int64));expect_equal(y$values(),c(.5,0,.75,.25));expect_equal(nn$grad(nn$sum(y),list(x))[[1]]$values(),rep(1,4));expect_error(nn$StarDomainImages(x,nn$full(1L,3,nn$DType$Int64)))
 x<-nn$full(c(40L,28L,28L,1L),.25,nn$DType$Float64)$requires_grad_();labels<-nn$from_vector_int64(rep(0:9,each=4),40L);ids<-nn$arange(100,140,1,nn$DType$Int64);p<-nn$SplitStarDomains(x,labels,ids,1);expect_equal(p$domain$values(),rep(0:2,10));expect_equal(p$unused$values(),seq(103,139,4));expect_identical(p$image$values(),nn$StarDomainImages(p$original,p$domain)$values());expect_equal(p$source$dtype(),nn$DType$Int64);expect_equal(nn$sum(nn$grad(nn$sum(p$original),list(x))[[1]])$item(),30*784);expect_error(nn$SplitStarDomains(x,labels,ids,2));expect_error(nn$SplitStarDomains(x,labels,nn$zeros(40L,nn$DType$Int64),1))
})

test_that("Explicit categorical draws reject biased words and own detached audit fields",{
 a<-nn$UniformCategorical(5L,0L,1073741825L);raw<-a$raw;labels<-a$labels;rm(a);gc();expect_equal(raw$values(),c(2357136044,2546248239,3071714933,2588848963,2340255427));expect_equal(labels$values(),raw$values()%%1073741825);expect_identical(nn$UniformCategorical(5L,0L,1073741825L)$raw$values(),raw$values());expect_false(raw$requires_grad());expect_equal(nn$max(nn$UniformCategorical(8L,6338L,1L)$labels)$item(),0);expect_error(nn$UniformCategorical(0L,1L))
})

test_that("One conditioned U-Net and image-only critic keep true skips and pure inference",{
 q<-star_inputs();o<-star_forward(q);expect_equal(q$g$count(),1609);expect_equal(q$g$size(),10);expect_equal(q$d$count(),688);expect_equal(q$d$size(),8);before<-list(q$g$flatten(),q$d$flatten());g<-nn$StarGanGenerate(q$g,q$x,q$t);d<-nn$StarGanDiscriminate(q$d,q$x);expect_equal(g$input$shape(),c(2,28,28,4));expect_equal(g$joined14$shape(),c(2,14,14,8));expect_equal(g$joined28$shape(),c(2,28,28,6));expect_identical(g$image$values(),o$fake$values());expect_identical(d$source$values(),o$real_source$values());expect_identical(nn$StarGan(q$g,q$d,q$x,q$t,q$t)$fake$values(),o$fake$values());expect_gt(star_diff(nn$StarGanGenerate(q$g,q$x,q$o)$image,g$image),1e-5);expect_identical(list(q$g$flatten(),q$d$flatten()),before)
})

test_that("Stable three-way CE and exact full-G/half-D scales retain live higher paths",{
 x<-nn$from_vector_double(c(1000,-1000,0),c(1L,3L))$requires_grad_();loss<-nn$StarDomainLoss(x,nn$ones(1L,nn$DType$Int64));expect_equal(loss$item(),2000);expect_equal(nn$grad(loss,list(x))[[1]]$values(),c(1,-1,0));x<-nn$zeros(c(1L,3L),nn$DType$Float64)$requires_grad_();graph<-nn$GradOptions();graph$create_graph<-TRUE;first<-nn$grad(nn$StarDomainLoss(x,nn$zeros(1L,nn$DType$Int64)),list(x),graph)[[1]];expect_gt(nn$max(nn$abs(nn$grad(nn$sum(nn$square(first)),list(x))[[1]]))$item(),0)
 q<-star_inputs();star_forward(q);for(name in q$d$names())q$d$set(name,nn$zeros(q$d$at(name)$shape(),q$d$dtype()));o<-star_forward(q);expect_equal(o$source_loss_g$item(),1);expect_equal(o$source_loss_d$item(),.5);expect_equal(o$discriminator_loss$item(),.5+log(3));expect_equal(o$generator_loss$item(),1+log(3)+10*o$reconstruction_loss$item())
})

test_that("Both G uses enter reconstruction and D then fresh G preserves write ownership",{
 q<-star_inputs();o<-star_forward(q);full<-nn$grad(o$reconstruction_loss,q$g$parameters());stopped<-nn$StarGanGenerate(q$g,o$fake$detach(),q$o)$image;wrong<-nn$grad(nn$mean(nn$abs(nn$sub(stopped,q$x))),q$g$parameters());expect_gt(max(vapply(seq_along(full),function(i)star_diff(full[[i]],wrong[[i]]),numeric(1))),1e-10)
 o<-star_forward(q);keep<-nn$GradOptions();keep$retain_graph<-TRUE;stale<-nn$grad(o$generator_loss,q$g$parameters(),keep);dd<-nn$grad(o$discriminator_loss,q$d$parameters());cfg<-nn$OptimizerConfig();cfg$kind<-nn$OptimizerKind$Adam;cfg$beta1<-.5;go<-nn$Optimizer(q$g,.0002,cfg);do_<-nn$Optimizer(q$d,.0002,cfg);before<-q$g$flatten();do_$step(nn$tensor_vector(dd));expect_identical(q$g$flatten(),before);after<-q$d$flatten();gg<-nn$grad(star_forward(q)$generator_loss,q$g$parameters());expect_gt(max(vapply(seq_along(gg),function(i)star_diff(stale[[i]],gg[[i]]),numeric(1))),1e-10);go$step(nn$tensor_vector(gg));expect_identical(q$d$flatten(),after);expect_equal(go$iterations(),1);expect_equal(do_$iterations(),1)
})

test_that("Owned fields survive parents and invalid domains reject before any leaves",{
 q<-star_inputs();x<-q$x$requires_grad_();o<-star_forward(q);image<-o$reconstruction;loss<-o$generator_loss;prob<-o$fake_domain_probability;rm(o,q);gc();graph<-nn$GradOptions();graph$create_graph<-TRUE;first<-nn$grad(loss,list(x),graph)[[1]];expect_gt(nn$max(nn$abs(nn$grad(nn$sum(nn$square(first)),list(x))[[1]]))$item(),0);expect_equal(image$shape(),x$shape());expect_equal(prob$shape(),c(2,3));q<-star_inputs();expect_error(nn$StarGan(q$g,q$g,q$x,q$o,q$t));expect_error(nn$StarGan(q$g,q$d,q$x,nn$full(2L,-1,nn$DType$Int64),q$t));expect_error(nn$StarGan(q$g,q$d,q$x,q$o,nn$full(2L,3,nn$DType$Int64)));expect_error(nn$StarGan(q$g,q$d,q$x,q$o,nn$ones(2L)));expect_equal(q$g$size(),0);expect_equal(q$d$size(),0)
})
