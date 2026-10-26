# Native R NN example. Variable lifecycle and initializer algorithms.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

lifecycle <- function() {
    w <- nn$Tensor()
    stopifnot(!w$defined())
    rejected <- tryCatch({w$values(); FALSE},error=function(e) {
        cat("Read before assignment:",conditionMessage(e),"\n"); TRUE
    })
    stopifnot(rejected)
    w <- nn$from_vector(c(1,2,3,4),c(2L,2L)); alias <- w
    w$add_(nn$full(c(2L,2L),1))
    w$copy_(nn$mul(w,10))
    w$copy_(nn$concat(nn$tensor_vector(list(nn$slice_axis(w,0L,0L,1L),nn$from_vector(c(-1,-2),c(1L,2L)))),0L))
    stopifnot(identical(alias$values(),c(20,30,-1,-2)))
    store <- nn$ParameterStore(1L); store$set("w",w)
    other <- store$clone()
    nn$no_grad(function() other$at("w")$add_(nn$full(c(2L,2L),100)))
    stopifnot(identical(store$at("w")$values(),c(20,30,-1,-2)),identical(other$at("w")$values(),c(120,130,99,98)))
    cat("Variable and cloned store:",store$at("w")$str(),other$at("w")$str(),"\n")
}

initializers <- function() {
    fan_in <- 256L; fan_out <- 128L; shape <- c(fan_in,fan_out)
    uniform <- function(limit,seed) nn$uniform(shape,-limit,limit,nn$Generator(seed))
    normal <- function(stddev,seed) nn$normal(shape,0,stddev,nn$Generator(seed))
    truncated <- function(stddev,seed) nn$truncated_normal(shape,0,stddev/.87962566103423978,nn$Generator(seed))
    orthogonal <- function(rows,cols,seed) {
        flip <- rows<cols
        qr <- nn$qr(nn$normal(c(max(rows,cols),min(rows,cols)),0,1,nn$Generator(seed)))
        q <- nn$mul(qr$q,nn$sign(nn$matrix_diag_part(qr$r)))
        if (flip) nn$transpose(q,c(1L,0L)) else q
    }
    glorot <- sqrt(2/(fan_in+fan_out)); he <- sqrt(2/fan_in)
    entries <- list(
        list("zeros",nn$zeros(shape),0),list("constant .1",nn$full(shape,.1),0),
        list("uniform",uniform(.05,1L),.05/sqrt(3)),list("normal",normal(.02,2L),.02),
        list("truncated normal",truncated(.02,3L),.02),
        list("glorot uniform",uniform(sqrt(6/(fan_in+fan_out)),4L),glorot),
        list("glorot normal",truncated(glorot,5L),glorot),
        list("he uniform",uniform(sqrt(6/fan_in),6L),he),list("he normal",truncated(he,7L),he),
        list("orthogonal",orthogonal(fan_in,fan_out,8L),1/sqrt(fan_in)),
        list("orthogonal wide",orthogonal(16L,64L,9L),1/sqrt(64)))
    for (entry in entries) {
        label <- entry[[1L]]; tensor <- entry[[2L]]; expected <- entry[[3L]]
        values <- tensor$values(); average <- mean(values)
        stddev <- sqrt(max(0,mean(values*values)-average*average))
        cat(label,":",tensor$shape(),"mean:",average,"std:",stddev,"expected:",expected,"range:",range(values),"\n")
        if (expected>0) {
            stopifnot(abs(stddev/expected-1)<.03,abs(average)<4*expected/sqrt(length(values))+1e-7)
        } else stopifnot(max(values)==min(values))
    }
    stopifnot(max(entries[[5L]][[2L]]$values())<=2*.02/.87962566103423978+1e-6)
    for (entry in entries[10:11]) {
        q <- entry[[2L]]; tall <- q$size(0L)>=q$size(1L)
        gram <- nn$matmul(q,q,tall,!tall)
        error <- max(abs(nn$sub(gram,nn$eye(gram$size(0L)))$values()))
        stopifnot(error<1e-5)
        cat(entry[[1L]],"orthogonality error:",error,"\n")
    }
}

main <- function(arguments=commandArgs(TRUE)) {
    lifecycle(); initializers()
    seeded <- list(); unseeded <- list()
    for (i in 1:2) {
        seeded[[i]] <- nn$normal(3L,0,1,nn$Generator(123L))$values()
        unseeded[[i]] <- nn$normal(3L)$values()
        cat("Seeded / unseeded:",seeded[[i]],unseeded[[i]],"\n")
    }
    stopifnot(identical(seeded[[1L]],seeded[[2L]]),!identical(unseeded[[1L]],unseeded[[2L]]))
    variable <- nn$from_vector(c(1,2,3))$requires_grad_()
    nn$no_grad(function() variable$add_(nn$from_vector(c(10,10,10))))
    gradient <- nn$grad(nn$sum(nn$square(variable)),list(variable))[[1L]]
    stopifnot(identical(variable$values(),c(11,12,13)),identical(gradient$values(),c(22,24,26)))
    cat("Resource variable:",variable$str(),"gradient:",gradient$str(),"\n")
    cat("All variable and initializer checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
