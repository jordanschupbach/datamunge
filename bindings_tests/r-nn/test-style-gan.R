library(testthat)
nn<-datamungerNN::nn
sg_inputs<-function(){t<-nn$DType$Float64;list(g=nn$ParameterStore(6607L,t),d=nn$ParameterStore(6619L,t),x=nn$full(c(2L,8L,8L,1L),.3,t),z=nn$cast(nn$GaussianNoise(8L,2,6659L),t),a=nn$cast(nn$GaussianNoise(c(4L,4L,1L),2,6653L),t),b=nn$cast(nn$GaussianNoise(c(8L,8L,1L),2,6661L),t))}
sg_forward<-function(q,v2=FALSE)nn$StyleGan(q$g,q$d,q$x,q$z,q$a,q$b,v2)
sg_diff<-function(a,b)nn$max(nn$abs(nn$sub(a,b)))$item()

test_that("Population spatial normalization and kernel normalization keep live derivatives",{
 x<-nn$from_vector_double(c(1,3,5,7),c(1L,2L,2L,1L))$requires_grad_();y<-nn$spatial_instance_normalize(x);expect_equal(nn$mean(y)$item(),0);expect_equal(nn$mean(nn$square(y))$item(),5/(5+1e-8));expect_error(nn$spatial_instance_normalize(x,0))
 t<-nn$DType$Float64;x<-nn$from_vector_double(c(1,0),c(1L,1L,1L,2L));w<-nn$from_vector_double(c(3,4),c(1L,1L,2L,1L))$requires_grad_();s<-nn$ones(c(1L,2L),t)$requires_grad_();o<-nn$modulated_conv2d(x,w,s);image<-o$image;factor<-o$demodulator;rm(o);gc();expect_equal(image$item(),.6);expect_equal(factor$item(),.2);graph<-nn$GradOptions();graph$create_graph<-TRUE;grads<-nn$grad(nn$sum(image),list(w,s),graph);expect_equal(grads[[1]]$values(),c(.128,-.096));expect_equal(grads[[2]]$values(),c(.384,-.384));expect_gt(nn$max(nn$abs(nn$grad(nn$sum(nn$square(grads[[1]])),list(w))[[1]]))$item(),0);expect_equal(nn$modulated_conv2d(x,w,s,FALSE)$image$item(),3);expect_equal(nn$modulated_conv2d(x,w,nn$zeros(s$shape(),t))$image$item(),0);expect_error(nn$modulated_conv2d(x,w,s,TRUE,0))
})

test_that("Both source variants map mix and synthesize without hidden state",{
 for(v2 in c(FALSE,TRUE)){q<-sg_inputs();o<-sg_forward(q,v2);expect_equal(q$g$count(),if(v2)625 else 661);expect_equal(q$g$size(),if(v2)19 else 17);expect_equal(q$d$count(),493);expect_equal(q$d$size(),6);before<-list(q$g$flatten(),q$d$flatten());w<-nn$StyleGanMap(q$g,q$z);p<-nn$StyleGanSynthesize(q$g,w,w,q$a,q$b,v2);expect_identical(p$image$values(),o$fake$values());expect_identical(w$values(),o$mapped$values());expect_identical(p$image$values(),nn$StyleGanSynthesize(q$g,w,w,nn$mul(q$a,2),nn$mul(q$b,3),v2)$image$values());other<-nn$StyleGanMap(q$g,nn$mul(q$z,-1));mix<-nn$StyleGanSynthesize(q$g,w,other,q$a,q$b,v2);expect_identical(p$first$values(),mix$first$values());expect_gt(sg_diff(p$image,mix$image),1e-8)
 if(v2){scaled<-nn$StyleGanSynthesize(q$g,w,w,q$a,q$b,TRUE,nn$mul(p$first_style,2));expect_lt(sg_diff(p$image,scaled$image),3e-6);expect_lt(nn$max(nn$abs(nn$sub(nn$div(nn$mul(scaled$first_demodulator,2),p$first_demodulator),1)))$item(),2e-4)}
 expect_identical(list(q$g$flatten(),q$d$flatten()),before)}
})

test_that("Zero initial noise remains learnable and zero truncation requires shared noise",{
 for(v2 in c(FALSE,TRUE)){q<-sg_inputs();q$a$requires_grad_();o<-sg_forward(q,v2);keep<-nn$GradOptions();keep$retain_graph<-TRUE;expect_equal(nn$sum(nn$abs(nn$grad(o$g_loss,list(q$a),keep)[[1]]))$item(),0);partial<-nn$grad(o$g_loss,list(q$g$at('layer4/noise_strength'),q$g$at('layer8/noise_strength')));expect_gt(sum(vapply(partial,function(p)nn$sum(nn$abs(p))$item(),numeric(1))),0);for(name in c('layer4/noise_strength','layer8/noise_strength'))q$g$set(name,nn$full(4L,.2,q$g$dtype()));w<-nn$StyleGanMap(q$g,q$z);center<-nn$tile(nn$mean(w,0L,TRUE),c(2L,1L));varied<-nn$StyleGanSynthesize(q$g,center,center,q$a,q$b,v2)$image;common<-nn$StyleGanSynthesize(q$g,center,center,nn$take_rows(q$a,c(0L,0L)),nn$take_rows(q$b,c(0L,0L)),v2)$image;expect_gt(sg_diff(nn$select(varied,0,0),nn$select(varied,0,1)),1e-8);expect_lt(sg_diff(nn$select(common,0,0),nn$select(common,0,1)),1e-13)}
})

test_that("D then fresh G captures change the gradient and preserve update ownership",{
 q<-sg_inputs();o<-sg_forward(q,TRUE);keep<-nn$GradOptions();keep$retain_graph<-TRUE;stale<-nn$grad(o$g_loss,q$g$parameters(),keep);dg<-nn$grad(o$d_loss,q$d$parameters());cfg<-nn$OptimizerConfig();cfg$kind<-nn$OptimizerKind$Adam;cfg$beta1<-.5;go<-nn$Optimizer(q$g,.0002,cfg);do_<-nn$Optimizer(q$d,.0002,cfg);before<-q$g$flatten();do_$step(nn$tensor_vector(dg));expect_identical(q$g$flatten(),before);dafter<-q$d$flatten();gg<-nn$grad(sg_forward(q,TRUE)$g_loss,q$g$parameters());expect_gt(max(vapply(seq_along(gg),function(i)sg_diff(stale[[i]],gg[[i]]),numeric(1))),1e-12);go$step(nn$tensor_vector(gg));expect_identical(q$d$flatten(),dafter);expect_equal(go$iterations(),1);expect_equal(do_$iterations(),1)
 for(name in q$d$names())q$d$set(name,nn$zeros(q$d$at(name)$shape(),q$d$dtype()));o<-sg_forward(q,TRUE);expect_equal(o$d_loss$item(),.5);expect_equal(o$g_loss$item(),1)
})

test_that("Owned style fields survive parents and invalid calls initialize no leaves",{
 q<-sg_inputs();z<-q$z$requires_grad_();o<-sg_forward(q,TRUE);loss<-o$g_loss;image<-o$fake;rm(o,q);gc();graph<-nn$GradOptions();graph$create_graph<-TRUE;first<-nn$grad(loss,list(z),graph)[[1]];expect_gt(nn$max(nn$abs(nn$grad(nn$sum(nn$square(first)),list(z))[[1]]))$item(),0);expect_equal(image$shape(),c(2,8,8,1));q<-sg_inputs();expect_error(nn$StyleGan(q$g,q$g,q$x,q$z,q$a,q$b,TRUE));expect_error(nn$StyleGan(q$g,q$d,q$x,q$z,nn$ones(c(1L,4L,4L,1L),q$g$dtype()),q$b,TRUE));expect_error(nn$StyleGanSynthesize(q$g,q$z,q$z,q$a,q$b,FALSE,nn$ones(c(2L,4L),q$g$dtype())));expect_equal(q$g$size(),0);expect_equal(q$d$size(),0)
})
