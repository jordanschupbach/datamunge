# Native R NN example. Derivatives, Jacobians and gradient updates run here.
.nn_common<-file.path(Sys.getenv('DATAMUNGE_EXAMPLES_ROOT',getwd()),'examples/r/nn/common.R')
if(!file.exists(.nn_common)).nn_common<-system.file('examples/nn/common.R',package='datamungerNN',mustWork=TRUE)
source(.nn_common,local=TRUE);rm(.nn_common)
matrix_loss<-function(x,weights,bias,target)nn$mean(nn$square(nn$sub(nn$add(nn$matmul(x,weights),bias),target)),c(0L,1L))
main<-function(arguments=commandArgs(TRUE)){
  args<-nn_options(arguments,list(epochs=5L,full=FALSE));stopifnot(args$epochs>0)
  for(value in c(.5,1,2)){
    x<-nn$scalar(value)$requires_grad_();y<-nn$add(nn$pow(x,3),nn$mul(x,2L))
    create<-nn$GradOptions();create$create_graph<-TRUE
    dy<-nn$grad(y,list(x),create)[[1L]];d2y<-nn$grad(dy,list(x))[[1L]]
    stopifnot(abs(dy$item()-(3*value^2+2))<1e-5,abs(d2y$item()-6*value)<1e-5)
    cat(sprintf('x=%.1f: y=%.9g,dy=%.9g,d2y=%.9g\n',value,y$item(),dy$item(),d2y$item()))
  }
  for(double in c(FALSE,TRUE)){
    dtype<-if(double)nn$DType$Float64 else nn$DType$Float32
    epsilon<-if(double)1e-5 else .01;tolerance<-if(double)1e-7 else 1e-3
    x<-nn$cast(nn$from_vector(c(1,2,3,0,-1,1,2,2,2,-1,0,1),c(4L,3L)),dtype)
    target<-nn$cast(nn$from_vector(c(1,0,0,1,1,1,0,0),c(4L,2L)),dtype)
    weights<-nn$cast(nn$from_vector(c(.1,-.2,.3,.4,-.5,.6),c(3L,2L)),dtype)$requires_grad_()
    bias<-nn$cast(nn$from_vector(c(.05,-.05)),dtype)$requires_grad_()
    loss<-matrix_loss(x,weights,bias,target);gradients<-nn$grad(loss,list(weights,bias))
    error<-nn$no_grad(function(){
      maximum<-0
      for(which in 1L:2L){
        parameter<-list(weights,bias)[[which]]
        for(index in seq_len(parameter$numel())){
          values<-parameter$values();samples<-numeric()
          for(sign in c(-1,1)){
            changed<-values;changed[index]<-changed[index]+sign*epsilon
            perturbed<-nn$from_vector_double(changed,parameter$shape())$to(dtype)
            samples<-c(samples,matrix_loss(x,if(which==1L)perturbed else weights,if(which==2L)perturbed else bias,target)$item())
          }
          maximum<-max(maximum,abs((samples[2L]-samples[1L])/(2*epsilon)-gradients[[which]]$values()[index]))
        }
      }
      maximum
    })
    stopifnot(error<tolerance);cat(sprintf('Matrix gradients %s: max finite-difference error=%.9g\n',nn$dtype_name(dtype),error))
  }
  x<-nn$from_vector(c(2,3))$requires_grad_();x0<-nn$select(x,0L,0L);x1<-nn$select(x,0L,1L)
  y<-nn$stack(nn$tensor_vector(list(nn$mul(x0,x1),nn$add(x0,x1),nn$sin(x0))))
  retain<-nn$GradOptions();retain$retain_graph<-TRUE;expected<-list(c(3,2),c(1,1),c(cos(2),0))
  for(row in 0L:2L){
    seed<-nn$one_hot(nn$scalar(row,nn$DType$Int64),3L)
    vjp<-nn$grad(list(y),list(x),retain,list(seed))[[1L]]
    stopifnot(max(abs(vjp$values()-expected[[row+1L]]))<1e-6);cat('Jacobian row',row,':',vjp$values(),'\n')
  }
  x<-nn$scalar(3)$requires_grad_()
  stopped<-nn$grad(nn$mul(x,nn$stop_gradient(x)),list(x))[[1L]]$item()
  complete<-nn$grad(nn$mul(x,x),list(x))[[1L]]$item();stopifnot(stopped==3,complete==6)
  cat('Stop-gradient derivative:',stopped,'full derivative:',complete,'\n')
  x<-nn$from_vector(c(-2,-1,0,1,2,3));target<-nn$from_vector(c(-7,-4,-1,2,5,8))
  weight<-nn$scalar(0)$requires_grad_();intercept<-nn$scalar(0)$requires_grad_();initial<-NULL
  for(epoch in seq_len(args$epochs)){
    loss<-nn$mean(nn$square(nn$sub(nn$add(nn$mul(x,weight),intercept),target)),0L)
    if(is.null(initial))initial<-loss$item()
    gradients<-nn$grad(loss,list(weight,intercept))
    nn$no_grad(function(){weight$sub_(gradients[[1L]],.05);intercept$sub_(gradients[[2L]],.05)})
    cat(sprintf('epoch%d: pre-update loss=%.9g,w=%.7g,c=%.7g\n',epoch,loss$item(),weight$item(),intercept$item()))
  }
  final<-nn$mean(nn$square(nn$sub(nn$add(nn$mul(x,weight),intercept),target)))$item()
  stopifnot(is.finite(final),final<initial);cat('All autodiff checks passed.\n');0L
}
if(sys.nframe()==0L)quit(status=main())
