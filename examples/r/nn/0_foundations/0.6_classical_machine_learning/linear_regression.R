# Native R NN example. Direct solvers and optimization are implemented here.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.6_classical_machine_learning/linear_regression.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE);rm(.nn_common)

least_squares_cholesky <- function(x,y) {
  lower <- nn$cholesky(nn$matmul(x,x,TRUE,FALSE))
  rhs <- nn$matmul(x,y,TRUE,FALSE)
  z <- nn$triangular_solve(lower,rhs)
  nn$triangular_solve(lower,z,TRUE,TRUE)
}
least_squares_qr <- function(x,y) {
  factor <- nn$qr(x)
  nn$triangular_solve(factor$r,nn$matmul(factor$q,y,TRUE,FALSE),FALSE)
}
main <- function(arguments=commandArgs(TRUE)) {
  args <- nn_options(arguments,list(epochs=5L,full=FALSE));stopifnot(args$epochs>0)
  n <- 64L;truth <- c(.7,-1.2,2.5);design <- target <- numeric()
  for(i in 0L:(n-1L)) {
    t <- 2*pi*i/n;row <- c(1,sqrt(2)*cos(t),sqrt(2)*sin(t));design <- c(design,row)
    value <- .1*cos(2*t)
    for(j in 1L:3L) value <- value+row[j]*truth[j]
    target <- c(target,value)
  }
  x <- nn$from_vector_double(design,c(n,3L));y <- nn$from_vector_double(target,c(n,1L))
  chol <- least_squares_cholesky(x,y);qr <- least_squares_qr(x,y)
  normal <- nn$matmul(x,nn$sub(nn$matmul(x,qr),y),TRUE,FALSE)
  stopifnot(max(abs(chol$values()-truth))<1e-12,max(abs(qr$values()-truth))<1e-12,max(abs(normal$values()))<1e-11)
  cat('Cholesky / QR coefficients:',chol$values(),'/',qr$values(),'\n')
  xf <- nn$cast(x,nn$DType$Float32);yf <- nn$cast(y,nn$DType$Float32)
  w <- nn$zeros(c(3L,1L));w$requires_grad_()
  for(epoch in seq_len(args$epochs)) {
    loss <- nn$mul(nn$mean(nn$square(nn$sub(nn$matmul(xf,w),yf)),c(0L,1L)),.5)
    gradient <- nn$grad(loss,list(w))[[1L]]
    nn$no_grad(function()w$sub_(gradient,.1))
    cat(sprintf('epoch%d: pre-update half-MSE=%.9g\n',epoch,loss$item()))
  }
  mse <- nn$mean(nn$square(nn$sub(nn$matmul(xf,w),yf)))$item();values <- w$values()
  expected <- truth*(1-.9^args$epochs)
  stopifnot(max(abs(values-expected))<2e-6,abs(mse-(.005+sum(truth^2)*.9^(2*args$epochs)))<3e-6)
  test_mse <- 0
  for(i in 0L:(n-1L)) {
    t <- 2*pi*(i+.5)/n;row <- c(1,sqrt(2)*cos(t),sqrt(2)*sin(t));error <- -.1*cos(2*t)
    for(j in 1L:3L) error <- error+row[j]*(values[j]-truth[j])
    test_mse <- test_mse+error^2/n
  }
  stopifnot(abs(test_mse-(.005+sum((values-truth)^2)))<1e-12)
  cat(sprintf('%d SGD steps: train MSE=%.9g; test MSE=%.9g\n',args$epochs,mse,test_mse))
  collinear <- near_target <- numeric()
  for(i in 0L:(n-1L)) {
    a <- design[3L*i+2L];b <- design[3L*i+3L];last <- a+1e-5*b
    collinear <- c(collinear,1,a,last);near_target <- c(near_target,1+2*a-2*last)
  }
  cx <- nn$from_vector_double(collinear,c(n,3L));cy <- nn$from_vector_double(near_target,c(n,1L))
  ch <- least_squares_cholesky(cx,cy);q <- least_squares_qr(cx,cy)
  error_chol <- max(abs(ch$values()-c(1,2,-2)));error_qr <- max(abs(q$values()-c(1,2,-2)))
  stopifnot(error_qr<1e-8,error_chol<.01)
  cat(sprintf('Collinear max coefficient error: Cholesky=%.9g, QR=%.9g\n',error_chol,error_qr))
  cat('All linear regression checks passed.\n');0L
}
if(sys.nframe()==0L)quit(status=main())
