# Native R NN example. Eager feeds, fetches, side effects and timings.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT", getwd()), "examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R", package="datamungerNN", mustWork=TRUE)
source(.nn_common, local=TRUE)
rm(.nn_common)

batch <- function(rows) nn$from_vector(as.numeric(t(outer(0:(rows-1L), 0:2, "+"))), c(rows, 3L))

main <- function(arguments=commandArgs(TRUE)) {
    weights <- nn$from_vector(c(1,0,0,1,1,1), c(3L,2L))
    bias <- nn$from_vector(c(.5,-.5))
    step <- nn$scalar(0, nn$DType$Int64)
    xw <- function(x) {
        if (!x$defined()) stop("x: no value supplied")
        if (x$dim()!=2L || x$size(1L)!=3L) stop("x: expected shape [?, 3]")
        nn$matmul(x, weights)
    }
    y_from_xw <- function(value) nn$add(value, bias)
    y <- function(x) y_from_xw(xw(x))
    increment <- function() step$add_(nn$scalar(1, nn$DType$Int64))
    previous_threads <- nn$get_num_threads()
    on.exit(nn$set_num_threads(previous_threads))
    nn$set_num_threads(2L)
    stopifnot(nn$get_num_threads()==2L)
    cat("Eager CPU execution; threads:",nn$get_num_threads(),"GEMM:",nn$gemm_backend(),"\n")
    result <- y(batch(2L))
    cat("y (batch 2):",result$str(),"sum:",nn$sum(result)$item(),"\n")
    stopifnot(identical(result$values(),c(2.5,2.5,4.5,4.5)), nn$sum(result)$item()==14)
    stopifnot(identical(as.numeric(y(batch(5L))$shape()),c(5,2)))
    fed <- y_from_xw(nn$from_vector(c(100,200),c(1L,2L)))
    stopifnot(identical(fed$values(),c(100.5,199.5)))
    cat("Fed intermediate:",fed$str(),"\n")
    for (invalid in list(nn$Tensor(), nn$zeros(c(2L,2L)))) {
        rejected <- tryCatch({y(invalid); FALSE}, error=function(e) {
            cat("Expected feed error:",conditionMessage(e),"\n"); TRUE
        })
        stopifnot(rejected)
    }
    step$fill_(0)
    for (i in 1:3) increment()
    stopifnot(step$item()==3)
    cat("Target-only counter:",step$item(),"\n")
    cat("Combined fetch:",y(batch(1L))$str(),"\n")
    increment()
    x <- batch(64L)
    start <- proc.time()[[3L]]
    value <- xw(x); mm <- proc.time()[[3L]]
    value <- y_from_xw(value); add <- proc.time()[[3L]]
    total <- nn$sum(value); end <- proc.time()[[3L]]
    stopifnot(total$item()==8384)
    cat("Stage timings (microseconds):",c(mm-start,add-mm,end-add)*1e6,"\n")
    callable_step <- function(x) {
        total <- nn$sum(y(x)); increment(); total
    }
    stopifnot(callable_step(batch(2L))$item()==14)
    iterations <- 2000L; x <- batch(8L)
    start <- proc.time()[[3L]]
    for (i in seq_len(iterations)) {nn$sum(y(x)); increment()}
    direct <- proc.time()[[3L]]
    for (i in seq_len(iterations)) callable_step(x)
    end <- proc.time()[[3L]]
    cat("Direct / callable microseconds per step:",c(direct-start,end-direct)*1e6/iterations,"\n")
    stopifnot(step$item()==4005)
    cat("Final counter:",step$item(),"All session checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
