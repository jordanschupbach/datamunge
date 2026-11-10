library(testthat)
nn <- datamungerNN::nn
cg_inputs <- function(){t<-nn$DType$Float64;list(p=lapply(c(6203L,6211L,6217L,6221L),function(s)nn$ParameterStore(s,t)),a=nn$full(c(2L,28L,28L,1L),.2,t),b=nn$full(c(3L,28L,28L,1L),.8,t))}
cg_forward <- function(d,cycle=10,identity=5)do.call(nn$CycleGan,c(d$p,list(d$a,d$b,cycle,identity)))

test_that("Native rotation/inversion and disjoint source domains preserve their data contracts",{
 x<-nn$from_vector_double(c(0,.25,.5,.75),c(1L,2L,2L,1L))$requires_grad_();y<-nn$RotateInvertGrayscale(x);expect_equal(y$values(),c(.5,1,.25,.75));expect_identical(nn$RotateInvertGrayscale(y,TRUE)$values(),x$values());expect_equal(nn$grad(nn$sum(y),list(x))[[1]]$values(),rep(-1,4))
 labels<-nn$from_vector_int64(rep(0:9,each=2),20L);ids<-nn$arange(1000,1020,1,nn$DType$Int64);image<-nn$full(c(20L,28L,28L,1L),.3,nn$DType$Float64);d<-nn$SplitCycleDomains(image,labels,ids);expect_equal(d$ids_a$values(),seq(1000,1018,2));expect_equal(d$ids_b$values(),seq(1001,1019,2));expect_identical(d$b$values(),nn$RotateInvertGrayscale(d$original_b)$values());expect_error(nn$SplitCycleDomains(image,labels,nn$zeros(20L,nn$DType$Int64)));expect_error(nn$RotateInvertGrayscale(nn$ones(c(2L,3L))))
})

test_that("Unpaired unequal rows have source-only translations and immutable four-store ownership",{
 d<-cg_inputs();o<-cg_forward(d);expect_equal(vapply(d$p,function(p)p$count(),numeric(1)),c(1390,1390,661,661));expect_equal(sum(vapply(d$p,function(p)p$size(),numeric(1))),32);expect_equal(o$fake_a$shape(),d$b$shape());expect_equal(o$fake_b$shape(),d$a$shape());expect_equal(o$cycle_a$shape(),d$a$shape());expect_equal(o$cycle_b$shape(),d$b$shape());before<-lapply(d$p,function(p)p$flatten());expect_identical(nn$Pix2PixGenerate(d$p[[1]],d$a)$image$values(),o$fake_b$values());expect_identical(nn$Pix2PixGenerate(d$p[[2]],d$a)$image$values(),o$identity_a$values());expect_identical(lapply(d$p,function(p)p$flatten()),before)
})

test_that("Joint G captures precede both explicit Adam writes, after separately stepped critics",{
 d<-cg_inputs();cg_forward(d);cfg<-nn$OptimizerConfig();cfg$kind<-nn$OptimizerKind$Adam;cfg$beta1<-.5;opt<-lapply(d$p,function(p)nn$Optimizer(p,.0002,cfg))
 for(k in 3:4){o<-cg_forward(d);loss<-if(k==3)o$d_a_loss else o$d_b_loss;before<-lapply(d$p,function(p)p$flatten());opt[[k]]$step(nn$tensor_vector(nn$grad(loss,d$p[[k]]$parameters())));expect_identical(before[-k],lapply(d$p[-k],function(p)p$flatten()))}
 o<-cg_forward(d);keep<-nn$GradOptions();keep$retain_graph<-TRUE;captured<-lapply(d$p[1:2],function(p)nn$grad(o$g_loss,p$parameters(),keep));before_ba<-d$p[[2]]$flatten();critics<-lapply(d$p[3:4],function(p)p$flatten());opt[[1]]$step(nn$tensor_vector(captured[[1]]));expect_identical(d$p[[2]]$flatten(),before_ba);wrong<-nn$grad(cg_forward(d)$g_loss,d$p[[2]]$parameters());change<-max(vapply(seq_along(wrong),function(i)nn$max(nn$abs(nn$sub(wrong[[i]],captured[[2]][[i]])))$item(),numeric(1)));expect_gt(change,1e-10);after_ab<-d$p[[1]]$flatten();opt[[2]]$step(nn$tensor_vector(captured[[2]]));expect_identical(d$p[[1]]$flatten(),after_ab);expect_identical(lapply(d$p[3:4],function(p)p$flatten()),critics);expect_equal(vapply(opt,function(o)o$iterations(),numeric(1)),rep(1,4))
})

test_that("Full G and half-scaled critics retain diagnostics at zero cycle/identity weights",{
 d<-cg_inputs();cg_forward(d);for(p in d$p[3:4])for(name in p$names())p$set(name,nn$zeros(p$at(name)$shape(),p$dtype()));o<-cg_forward(d);expect_equal(o$d_a_loss$item(),.5);expect_equal(o$d_b_loss$item(),.5);expect_equal(o$adversarial_a$item(),1);expect_equal(o$adversarial_b$item(),1);expect_equal(o$g_loss$item(),2+10*(o$cycle_loss_a$item()+o$cycle_loss_b$item())+5*(o$identity_loss_a$item()+o$identity_loss_b$item()));expect_equal(cg_forward(d,0,0)$g_loss$item(),2)
})

test_that("Owning cycles retain higher input paths and invalid calls fail before leaves",{
 d<-cg_inputs();a<-d$a$requires_grad_();o<-cg_forward(d);loss<-o$g_loss;cycle<-o$cycle_a;rm(o,d);gc();graph<-nn$GradOptions();graph$create_graph<-TRUE;first<-nn$grad(loss,list(a),graph)[[1]];second<-nn$grad(nn$sum(nn$square(first)),list(a))[[1]];expect_equal(cycle$shape(),a$shape());expect_gt(nn$max(nn$abs(second))$item(),0)
 d<-cg_inputs();expect_error(cg_forward(d,-1));expect_error(cg_forward(d,10,Inf));d$p[[4]]<-d$p[[1]];expect_error(cg_forward(d));expect_equal(sum(vapply(d$p,function(p)p$size(),numeric(1))),0)
})
