library(testthat)
nn<-datamungerNN::nn
asr_inputs<-function(){t<-nn$DType$Float64;list(p=lapply(c(7001L,7013L,0L),function(s)nn$ParameterStore(s,t)),lr=nn$full(c(2L,16L,16L,3L),.2,t),hr=nn$full(c(2L,32L,32L,3L),.4,t),bic=nn$full(c(2L,32L,32L,3L),.2,t),w=nn$full(c(3L,3L,3L,64L),.001,t),b=nn$zeros(64L,t))}
asr_forward<-function(q,enhanced=FALSE,phase=TRUE)do.call(nn$AdversarialSR,c(q$p,list(q$lr,q$hr,q$bic,q$w,q$b,enhanced,phase)))
asr_diff<-function(a,b)nn$max(nn$abs(nn$sub(a,b)))$item()
asr_round<-function(value)nn$full(1L,value,nn$DType$Float32)$item()
asr_names<-c('residual0/bn1','residual0/bn2','residual1/bn1','residual1/bn2','body_bn')

test_that("Relativistic opposite-pool means stay live with unequal rows and common shifts",{
 r<-nn$from_vector_double(c(-2,0,3),c(3L,1L))$requires_grad_();f<-nn$from_vector_double(c(-1,2),c(2L,1L))$requires_grad_();o<-nn$RelativisticAverageGan(r,f);graph<-nn$GradOptions();graph$create_graph<-TRUE;g<-nn$grad(o$generator,list(r,f),graph);expect_equal(nn$sum(g[[1]])$item(),-nn$sum(g[[2]])$item());expect_gt(nn$max(nn$abs(nn$grad(nn$sum(nn$square(g[[1]])),list(f))[[1]]))$item(),0);expect_equal(nn$RelativisticAverageGan(nn$add(r$detach(),100),nn$add(f$detach(),100))$generator$item(),o$generator$item());expect_error(nn$RelativisticAverageGan(nn$ones(c(2L,2L)),nn$ones(c(2L,1L))))
})

test_that("Frozen features use rounded BGR means and stop kernel and bias paths",{
 t<-nn$DType$Float64;v<-rep(0,3*3*3*64);for(c in 0:2)v[((1*3+1)*3+c)*64+c+1]<-1;w<-nn$from_vector_double(v,c(3L,3L,3L,64L))$requires_grad_();b<-nn$zeros(64L,t)$requires_grad_();x<-nn$from_vector_double(c(1,0,.5),c(1L,1L,1L,3L))$requires_grad_();o<-nn$AdversarialSRFeatures(x,w,b);expect_equal(o$values()[1:3],c(127.5-103.93900299072266,-116.77899932861328,255-123.68000030517578));unused<-nn$GradOptions();unused$allow_unused<-TRUE;g<-nn$grad(nn$sum(o),list(x,w,b),unused);expect_equal(g[[1]]$values(),rep(255,3));expect_true(!g[[2]]$defined() || nn$sum(nn$abs(g[[2]]))$item()==0);expect_true(!g[[3]]$defined() || nn$sum(nn$abs(g[[3]]))$item()==0)
})

test_that("Both source censuses retain target-free G paths and source objective scales",{
 for(enhanced in c(FALSE,TRUE)){q<-asr_inputs();o<-asr_forward(q,enhanced);expect_equal(vapply(q$p,function(p)p$count(),numeric(1)),if(enhanced)c(3511,1672,0) else c(1591,1673,40));expect_equal(vapply(q$p,function(p)p$size(),numeric(1)),if(enhanced)c(40,7,0) else c(25,8,10));before<-lapply(q$p,function(p)p$flatten());q$hr<-nn$zeros(q$hr$shape(),q$hr$dtype());changed<-asr_forward(q,enhanced);expect_identical(o$prediction$values(),changed$prediction$values());expect_identical(o$prediction_training$values(),changed$prediction_training$values());expect_false(o$target_feature$requires_grad());warm<-asr_forward(q,enhanced,FALSE);expect_identical(warm$g_loss$values(),warm$pixel_loss$values());expect_equal(o$g_loss$item(),o$content_loss$item()+asr_round(if(enhanced).005 else .001)*o$adversarial_loss$item()+(if(enhanced)asr_round(.01)*o$pixel_loss$item() else 0));expect_identical(lapply(q$p,function(p)p$flatten()),before)}
})

test_that("Detached BN captures do not write stats and only saved inference reads them",{
 q<-asr_inputs();o<-asr_forward(q);before<-q$p[[3]]$flatten();capture<-nn$AdversarialSRGenerate(q$p[[1]],q$p[[3]],q$lr,q$bic,FALSE,TRUE);expect_equal(capture$moments_mean$shape(),c(5,4));expect_false(capture$moments_mean$requires_grad());expect_identical(q$p[[3]]$flatten(),before);for(name in asr_names){q$p[[3]]$set(paste0(name,'/moving_mean'),nn$ones(4L,q$lr$dtype()));q$p[[3]]$set(paste0(name,'/moving_var'),nn$full(4L,2,q$lr$dtype()))};batch<-nn$AdversarialSRGenerate(q$p[[1]],q$p[[3]],q$lr,q$bic,FALSE,TRUE);saved<-nn$AdversarialSRGenerate(q$p[[1]],q$p[[3]],q$lr,q$bic);expect_identical(batch$image$values(),capture$image$values());expect_gt(asr_diff(saved$image,o$prediction),1e-7);expect_false(saved$moments_mean$defined())
})

test_that("G writes precede explicit BN commits from the same capture",{
 q<-asr_inputs();o<-asr_forward(q,FALSE,FALSE);means<-o$moments_mean;vars<-o$moments_unbiased_variance;before<-q$p[[3]]$flatten();cfg<-nn$OptimizerConfig();cfg$kind<-nn$OptimizerKind$Adam;cfg$beta1<-.9;optimizer<-nn$Optimizer(q$p[[1]],.0001,cfg);optimizer$step(nn$tensor_vector(nn$grad(o$g_loss,q$p[[1]]$parameters())));expect_identical(q$p[[3]]$flatten(),before);wrong<-nn$AdversarialSRGenerate(q$p[[1]],q$p[[3]],q$lr,q$bic,FALSE,TRUE);expect_gt(asr_diff(wrong$moments_mean,means),1e-8);for(i in seq_along(asr_names)){name<-asr_names[[i]];next_values<-nn$batch_norm_moving_values(q$p[[3]]$at(paste0(name,'/moving_mean')),q$p[[3]]$at(paste0(name,'/moving_var')),nn$select(means,0,i-1L),nn$select(vars,0,i-1L));q$p[[3]]$set(paste0(name,'/moving_mean'),next_values$mean);q$p[[3]]$set(paste0(name,'/moving_var'),next_values$variance)};expect_false(identical(q$p[[3]]$flatten(),before))
})

test_that("Cold optimizer reset and fresh G preserve update owners",{
 q<-asr_inputs();asr_forward(q,TRUE);cfg<-nn$OptimizerConfig();cfg$kind<-nn$OptimizerKind$Adam;cfg$beta1<-.9;go<-nn$Optimizer(q$p[[1]],.0001,cfg);do_<-nn$Optimizer(q$p[[2]],.0001,cfg);db<-q$p[[2]]$flatten();go$step(nn$tensor_vector(nn$grad(asr_forward(q,TRUE,FALSE)$g_loss,q$p[[1]]$parameters())));expect_identical(q$p[[2]]$flatten(),db);expect_equal(go$iterations(),1);expect_equal(do_$iterations(),0);before<-lapply(q$p,function(p)p$flatten());go<-nn$Optimizer(q$p[[1]],.0001,cfg);do_<-nn$Optimizer(q$p[[2]],.0001,cfg);expect_equal(go$iterations(),0);expect_identical(lapply(q$p,function(p)p$flatten()),before);o<-asr_forward(q,TRUE);keep<-nn$GradOptions();keep$retain_graph<-TRUE;stale<-nn$grad(o$g_loss,q$p[[1]]$parameters(),keep);dd<-nn$grad(o$d_loss,q$p[[2]]$parameters());gb<-q$p[[1]]$flatten();do_$step(nn$tensor_vector(dd));expect_identical(q$p[[1]]$flatten(),gb);after<-q$p[[2]]$flatten();gg<-nn$grad(asr_forward(q,TRUE)$g_loss,q$p[[1]]$parameters());expect_gt(max(vapply(seq_along(gg),function(i)asr_diff(stale[[i]],gg[[i]]),numeric(1))),1e-13);go$step(nn$tensor_vector(gg));expect_identical(q$p[[2]]$flatten(),after);expect_equal(q$p[[3]]$size(),0)
})

test_that("Owned SR fields keep higher LR paths and invalid calls create no leaves",{
 for(enhanced in c(FALSE,TRUE)){q<-asr_inputs();lr<-q$lr$requires_grad_();o<-asr_forward(q,enhanced);image<-o$prediction;loss<-nn$mean(nn$square(o$prediction_training));rm(o,q);gc();graph<-nn$GradOptions();graph$create_graph<-TRUE;first<-nn$grad(loss,list(lr),graph)[[1]];expect_gt(nn$max(nn$abs(nn$grad(nn$sum(nn$square(first)),list(lr))[[1]]))$item(),0);expect_equal(image$shape(),c(2,32,32,3))};q<-asr_inputs();expect_error(nn$AdversarialSR(q$p[[1]],q$p[[1]],q$p[[3]],q$lr,q$hr,q$bic,q$w,q$b));q$w<-nn$zeros(c(3L,3L,3L,63L),q$lr$dtype());expect_error(asr_forward(q));expect_equal(sum(vapply(q$p,function(p)p$size(),numeric(1))),0)
})
