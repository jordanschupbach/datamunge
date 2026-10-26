# Native R NN example. Naming, eager ordering, placement and errors live here.
.nn_common<-file.path(Sys.getenv('DATAMUNGE_EXAMPLES_ROOT',getwd()),'examples/r/nn/common.R')
if(!file.exists(.nn_common)).nn_common<-system.file('examples/nn/common.R',package='datamungerNN',mustWork=TRUE)
source(.nn_common,local=TRUE);rm(.nn_common)
main<-function(arguments=commandArgs(TRUE)){
  store<-nn$ParameterStore(1L)
  a<-store$get('a',integer(),nn$init$constant(1))
  w<-nn$from_vector(c(1,0,0,1),c(2L,2L))$requires_grad_(TRUE)
  store$set('encoder/w',w)
  h1<-nn$add(a,nn$scalar(2));h2<-nn$add(h1,nn$scalar(3))
  p<-store$get('block/k',integer(),nn$init$constant(5))
  q<-store$get('block/k',integer(),nn$init$constant(7))
  stopifnot(p$is_same(q),q$item()==5,store$size()==3,h2$item()==6)
  cat('parameter names:',store$names(),'\n')
  cat('recorded operations:',h1$grad_fn_name(),h2$grad_fn_name(),'\n')
  y<-nn$sum(nn$matmul(nn$mul(nn$reshape(nn$mul(h2,p),c(1L,1L)),nn$ones(c(1L,2L))),w))
  gradients<-nn$grad(y,list(a,w))
  stopifnot(gradients[[1]]$item()==10,identical(gradients[[2]]$values(),rep(30,4)))
  cat('dy/da:',gradients[[1]]$item(),'dy/dw:',gradients[[2]]$values(),'\n')
  counter<-nn$scalar(0,nn$DType$Int32)
  reads<-replicate(3,counter$clone()$item())
  for(i in seq_len(3)){
    counter$add_(nn$scalar(1,nn$DType$Int32));reads<-c(reads,counter$clone()$item())
  }
  reads<-c(reads,counter$clone()$item())
  stopifnot(identical(as.numeric(reads),c(0,0,0,1,2,3,3)))
  cat('ordered counter reads:',reads,'\n')
  matrix<-nn$from_vector(c(1,2,3,4),c(2L,2L))
  placed<-nn$add(nn$matmul(matrix,matrix),nn$matmul(matrix,matrix))
  stopifnot(identical(placed$values(),c(14,20,30,44)))
  cat('CPU placement result:',placed$values(),'\n')
  bad<-nn$from_vector(1:6,c(2L,3L))
  rejected<-tryCatch({nn$matmul(bad,bad);FALSE},error=function(error){
    cat('Expected construction error:',conditionMessage(error),'\n');TRUE
  })
  stopifnot(rejected)
  later<-nn$neg(bad);stopifnot(later$defined(),tail(later$values(),1)==-6)
  cat('All graph construction checks passed.\n');0L
}
if(sys.nframe()==0L)quit(status=main())
