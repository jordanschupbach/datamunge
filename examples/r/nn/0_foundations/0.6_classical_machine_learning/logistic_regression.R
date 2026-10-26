# Native R NN example. Binary and multinomial classification, including confusion matrices.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

binary_cross_entropy <- function(logits,labels) {
    positive <- nn$greater_equal(logits,0)
    magnitude <- nn$where(positive,nn$neg(logits),logits)
    linear <- nn$where(positive,nn$mul(logits,nn$sub(1,labels)),nn$neg(nn$mul(logits,labels)))
    nn$add(linear,nn$log1p(nn$exp(magnitude)))
}

Adam <- function(parameters,rate=.04) {
    m <- lapply(parameters,nn$zeros_like); v <- lapply(parameters,nn$zeros_like); t <- 0L
    list(step=function(gradients) {
        t <<- t+1L; c1 <- 1-.9^t; c2 <- 1-.999^t
        nn$no_grad(function() {
            for (i in seq_along(parameters)) {
                m[[i]] <<- nn$add(nn$mul(m[[i]],.9),nn$mul(gradients[[i]],.1))
                v[[i]] <<- nn$add(nn$mul(v[[i]],.999),nn$mul(nn$square(gradients[[i]]),.001))
                update <- nn$div(nn$div(m[[i]],c1),nn$add(nn$sqrt(nn$div(v[[i]],c2)),1e-8))
                parameters[[i]]$sub_(update,rate)
            }
        })
    })
}

print_confusion <- function(labels,prediction,classes) {
    confusion <- nn$ConfusionMatrix(classes)
    confusion$add(labels,prediction)
    cells <- matrix(confusion$to_tensor()$values(),nrow=classes,byrow=TRUE)
    stopifnot(sum(cells)==labels$numel())
    dimnames(cells) <- list(truth=0:(classes-1L),prediction=0:(classes-1L))
    cat("Confusion matrix (rows=truth, columns=prediction):\n")
    print(cells)
    invisible(confusion$accuracy())
}

binary <- function(steps) {
    extreme <- nn$from_vector(c(-1000,-1000,1000,1000,0,0))$requires_grad_()
    labels <- nn$from_vector(c(0,1,0,1,0,1))
    stable <- binary_cross_entropy(extreme,labels)
    derivative <- nn$grad(nn$sum(stable),list(extreme))[[1L]]
    stopifnot(max(abs(stable$values()-c(0,1000,1000,0,log(2),log(2))))<1e-6,
              identical(derivative$values(),c(0,-1,1,0,.5,-.5)))
    rng <- nn$Mt19937(31L)
    dataset <- function(n) {
        features <- numeric(n*3L); targets <- numeric(n)
        for (i in seq_len(n)) {
            a <- rng$normal32(); b <- rng$normal32()
            chance <- 1/(1+exp(-(-.4+1.5*a-2*b)))
            features[(i-1L)*3L+1:3] <- c(1,a,b)
            targets[i] <- as.numeric(rng$uniform32()<chance)
        }
        list(x=nn$from_vector(features,c(n,3L)),y=nn$from_vector(targets,c(n,1L)))
    }
    train <- dataset(1200L); test <- dataset(600L)
    w <- nn$zeros(c(3L,1L))$requires_grad_()
    mask <- nn$from_vector(c(0,1,1),c(3L,1L))
    objective <- function(x,y) {
        loss <- nn$mean(binary_cross_entropy(nn$matmul(x,w),y),c(0L,1L))
        list(loss=loss,regularized=nn$add(loss,nn$mul(nn$sum(nn$mul(mask,nn$square(w)),c(0L,1L)),.005)))
    }
    opt <- Adam(list(w))
    initial <- objective(train$x,train$y)$regularized$item()
    for (epoch in seq_len(steps)) {
        loss <- objective(train$x,train$y)$regularized
        opt$step(nn$grad(loss,list(w)))
        cat(sprintf("binary epoch %d/%d: objective=%.9g\n",epoch,steps,loss$item()))
    }
    stopifnot(objective(train$x,train$y)$regularized$item()<initial)
    stationarity <- nn$grad(objective(train$x,train$y)$regularized,list(w))[[1L]]
    if (steps>=400L) stopifnot(max(abs(stationarity$values()))<1e-5)
    nn$no_grad(function() {
        probability <- nn$sigmoid(nn$matmul(test$x,w))
        tx <- matrix(test$x$values(),ncol=3L,byrow=TRUE)
        calibration <- mean(abs(probability$values()-1/(1+exp(-(-.4+1.5*tx[,2L]-2*tx[,3L])))))
        loss <- objective(test$x,test$y)$loss$item()
        prediction <- nn$reshape(nn$cast(nn$greater_equal(probability,.5),nn$DType$Int64),-1L)
        truth <- nn$reshape(nn$cast(test$y,nn$DType$Int64),-1L)
        accuracy <- nn$accuracy(prediction,truth)
        cat(sprintf("Binary NLL=%.9g; accuracy=%.9g; calibration error=%.9g; weights=%s\n",loss,accuracy,calibration,w$str()))
        if (steps>=400L) stopifnot(loss<.5,accuracy>.75,calibration<.06)
        print_confusion(truth,prediction,2L)
    })
}

multinomial <- function(steps) {
    rng <- nn$Mt19937(47L)
    dataset <- function(n) {
        features <- numeric(n*2L); labels <- numeric(n)
        for (i in 0:(n-1L)) {
            c <- i%%3L; angle <- 2*pi*c/3
            features[i*2L+1:2] <- c(2*cos(angle)+rng$normal32(0,.55),2*sin(angle)+rng$normal32(0,.55))
            labels[i+1L] <- c
        }
        list(x=nn$from_vector(features,c(n,2L)),y=nn$from_vector_int64(labels))
    }
    train <- dataset(450L); test <- dataset(300L)
    limit <- sqrt(6/5)
    w <- nn$uniform(c(2L,3L),-limit,limit,nn$Generator(19L))$requires_grad_()
    b <- nn$zeros(3L)$requires_grad_()
    nll <- function(x,y) {
        log_probs <- nn$log_softmax(nn$add(nn$matmul(x,w),b),1L)
        nn$neg(nn$mean(nn$sum(nn$mul(nn$one_hot(y,3L),log_probs),1L),0L))
    }
    opt <- Adam(list(w,b)); initial <- nll(train$x,train$y)$item()
    for (epoch in seq_len(steps)) {
        loss <- nn$add(nll(train$x,train$y),nn$mul(nn$sum(nn$square(w)),.0005))
        opt$step(nn$grad(loss,list(w,b)))
        cat(sprintf("multinomial epoch %d/%d: objective=%.9g\n",epoch,steps,loss$item()))
    }
    stopifnot(nll(train$x,train$y)$item()<initial)
    nn$no_grad(function() {
        logits <- nn$add(nn$matmul(test$x,w),b)
        probabilities <- nn$softmax(logits,1L)
        stopifnot(max(abs(nn$sum(probabilities,1L)$values()-1))<1e-6)
        loss <- nll(test$x,test$y)$item(); accuracy <- nn$accuracy(logits,test$y)
        if (steps>=300L) stopifnot(accuracy>.97,loss<.1)
        cat(sprintf("Multinomial NLL=%.9g; accuracy=%.9g\n",loss,accuracy))
        print_confusion(test$y,nn$argmax(logits,1L),3L)
    })
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(epochs=5L,full=FALSE))
    if (is.na(args$epochs)||args$epochs<1L) stop("epochs must be positive")
    binary(args$epochs); multinomial(args$epochs)
    cat("All logistic regression checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
