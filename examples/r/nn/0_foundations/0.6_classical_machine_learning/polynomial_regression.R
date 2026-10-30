# Native R NN example. Legendre features, QR fits and diagnostics run here.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.6_classical_machine_learning/polynomial_regression.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if(!file.exists(.nn_common)).nn_common<-system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE);rm(.nn_common)
polynomial_features<-function(x,degree){
  if(degree<0L)stop('polynomial degree must be nonnegative')
  columns<-list(nn$ones_like(x))
  if(degree>0L)columns[[2L]]<-x
  if(degree>=2L)for(n in 2L:degree){
    columns[[n+1L]]<-nn$div(nn$sub(nn$mul(nn$mul(x,columns[[n]]),2*n-1),nn$mul(columns[[n-1L]],n-1)),n)
  }
  nn$concat(nn$tensor_vector(columns),1L)
}
main<-function(arguments=commandArgs(TRUE)){
  args<-nn_options(arguments,list(epochs=5L,full=FALSE))
  n<-20L;replicas<-400L;grid<-201L;sigma<-.3;rng<-nn$Mt19937(113L)
  xs<--1+2*(0L:(n-1L))/(n-1L);qs<--1+2*(0L:(grid-1L))/(grid-1L)
  ys<-numeric(n*replicas)
  for(i in seq_len(n))for(r in seq_len(replicas))ys[(i-1L)*replicas+r]<-sin(pi*xs[i])+rng$normal64(0,sigma)
  x<-nn$from_vector_double(xs,c(n,1L));y<-nn$from_vector_double(ys,c(n,replicas));query<-nn$from_vector_double(qs,c(grid,1L))
  directory<-file.path(nn_output_root(args$full),'metrics/polynomial_regression');dir.create(directory,recursive=TRUE,showWarnings=FALSE)
  curves<-file(file.path(directory,'curves.tsv'),'w');summary<-file(file.path(directory,'summary.tsv'),'w')
  on.exit({close(curves);close(summary)},add=TRUE)
  writeLines('degree\tx\ttruth\tmean\tstddev',curves)
  writeLines('degree\ttrain_mse\tbias_squared\tvariance\tnoise_variance\texpected_test_mse',summary)
  errors<-biases<-variances<-risks<-numeric()
  for(degree in c(1L,3L,15L)){
    design<-polynomial_features(x,degree);query_design<-polynomial_features(query,degree)
    factor<-nn$qr(design);weights<-nn$triangular_solve(factor$r,nn$matmul(factor$q,y,TRUE,FALSE),FALSE)
    prediction<-nn$matmul(query_design,weights)$values();fit<-nn$matmul(design,weights)$values()
    train_error<-mean((fit-ys)^2);bias<-variance<-direct_mse<-0
    for(i in seq_len(grid)){
      truth<-sin(pi*qs[i]);row<-prediction[(i-1L)*replicas+seq_len(replicas)]
      average<-sum(row/replicas);var<-sum((row-average)^2/replicas)
      direct_mse<-direct_mse+sum((row-truth)^2/(replicas*grid))
      bias<-bias+(average-truth)^2/grid;variance<-variance+var/grid
      writeLines(sprintf('%d\t%.17g\t%.17g\t%.17g\t%.17g',degree,qs[i],truth,average,sqrt(var)),curves)
    }
    stopifnot(abs(direct_mse-bias-variance)<1e-11);risk<-direct_mse+sigma^2
    writeLines(sprintf('%d\t%.17g\t%.17g\t%.17g\t%.17g\t%.17g',degree,train_error,bias,variance,sigma^2,risk),summary)
    errors<-c(errors,train_error);biases<-c(biases,bias);variances<-c(variances,variance);risks<-c(risks,risk)
    cat(sprintf('degree%d: train MSE=%.9g; bias^2=%.9g; variance=%.9g; noisy-test risk=%.9g\n',degree,train_error,bias,variance,risk))
  }
  stopifnot(errors[1L]>errors[2L],errors[2L]>errors[3L],biases[1L]>biases[2L]*10,variances[3L]>variances[2L]*5,risks[2L]<risks[1L],risks[2L]<risks[3L])
  cat('All polynomial regression checks passed.\n');0L
}
if(sys.nframe()==0L)quit(status=main())
