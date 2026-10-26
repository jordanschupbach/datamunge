# Native R NN example. Training, dependency pruning and frozen export live here.
.nn_common<-file.path(Sys.getenv('DATAMUNGE_EXAMPLES_ROOT',getwd()),'examples/r/nn/common.R')
if(!file.exists(.nn_common)).nn_common<-system.file('examples/nn/common.R',package='datamungerNN',mustWork=TRUE)
source(.nn_common,local=TRUE);rm(.nn_common)
Affine<-function(initial){
  store<-nn$ParameterStore()
  w<-store$get('model/w',integer(),nn$init$constant(initial))
  b<-store$get('model/b',integer(),nn$init$constant(initial))
  list(store=store,w=w,b=b,predict=function(x)nn$add(nn$mul(x,w),b))
}
freeze<-function(store,endpoints,outputs){
  used<-rep(FALSE,store$size());options<-nn$GradOptions();options$allow_unused<-TRUE
  probe<-nn$from_vector(1)
  for(name in outputs){
    if(!name %in% names(endpoints))stop('unknown export output: ',name)
    gradients<-nn$grad(nn$sum(endpoints[[name]](probe,probe)),nn$tensor_list(store$parameters()),options)
    used<-used|vapply(gradients,function(value)value$defined(),logical(1))
  }
  result<-list()
  for(name in store$names()[used])result[[name]]<-store$at(name)$detach()$clone()
  result
}
main<-function(arguments=commandArgs(TRUE)){
  options<-nn_options(arguments,list(epochs=if('--quick' %in% arguments)5L else 500L,full=FALSE))
  epochs<-options$epochs;if(epochs<=0L)stop('positive training budget required')
  model<-Affine(0);cfg<-nn$OptimizerConfig();cfg$kind<-nn$OptimizerKind$Adam
  optimizer<-nn$Optimizer(model$store,.05,cfg)
  endpoints<-list(predictions=function(x,y)model$predict(x),
                  loss=function(x,y)nn$mean(nn$square(nn$sub(model$predict(x),y))))
  x<-nn$from_vector(c(-2,-1,0,1,2));y<-nn$from_vector(c(-4.5,-2.5,-.5,1.5,3.5))
  initial_loss<-endpoints$loss(x,y)$item()
  for(i in seq_len(epochs))optimizer$minimize(endpoints$loss(x,y))
  stopifnot(optimizer$iterations()==epochs,endpoints$loss(x,y)$item()<initial_loss)
  query<-nn$from_vector(c(-3,.5,4));expected<-model$predict(query)$values()
  new_batch<-model$predict(nn$from_vector(5))$item()
  stopifnot(all(is.finite(expected)))
  if(epochs>=500L)stopifnot(max(abs(expected-c(-6.5,.5,7.5)))<1e-5,abs(new_batch-9.5)<1e-5)
  frozen<-freeze(model$store,endpoints,'predictions')
  slots<-optimizer$state();training_count<-model$store$size()+slots$size()
  stopifnot(length(frozen)==2L,length(frozen)<training_count,all(startsWith(names(frozen),'model/')),
            all(vapply(frozen,function(value)!value$requires_grad(),logical(1))))
  directory<-file.path(nn_output_root(options$full),'models/frozen_affine')
  dir.create(directory,recursive=TRUE,showWarnings=FALSE);path<-file.path(directory,'weights.npz')
  nn$save_npz(path,frozen)
  writeLines(c('input=features float32 [batch]','output=predictions float32 [batch]',
               'predictions = features * model/w + model/b'),file.path(directory,'signature.txt'))
  nn$no_grad(function(){model$w$fill_(-100);model$b$fill_(100)})
  serving<-Affine(NaN);serving$store$load(path,TRUE)
  stopifnot(identical(serving$predict(query)$values(),expected),serving$predict(nn$from_vector(5))$item()==new_batch)
  rejected<-tryCatch({freeze(model$store,endpoints,'does_not_exist');FALSE},error=function(e)TRUE)
  stopifnot(rejected)
  cat('training state arrays=',training_count,'frozen arrays=',length(frozen),'\n')
  cat('predictions=',expected,'exact after reload; new batch=',new_batch,'\n')
  cat('All frozen-graph checks passed.\n');0L
}
if(sys.nframe()==0L)quit(status=main())
