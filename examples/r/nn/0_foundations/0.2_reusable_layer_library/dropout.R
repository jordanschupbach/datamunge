# Native R NN example. Inverted dropout, regularization and Monte Carlo uncertainty.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

statistics <- function() {
    generator <- nn$Generator(7L);n <- 100000L;x <- nn$ones(c(1L,n))
    for (p in c(0,.1,.25,.5,.8)) {
        p <- nn$scalar(p)$item();keep <- nn$scalar(1-p)$item()
        values <- nn$dropout(x,p,TRUE,generator)$values()
        zero <- mean(values==0);average <- mean(values);variance <- mean(values*values)-average*average
        stopifnot(abs(zero-p)<.006,abs(average-1)<.02,abs(variance-p/keep)<.03*(1+p/keep),abs(max(values)-1/keep)<1e-5)
        cat(sprintf("p=%.6g: zero fraction=%.9g, mean=%.9g, variance=%.9g, expected variance=%.9g, survivor=%.9g\n",p,zero,average,variance,p/keep,max(values)))
    }
    rng <- nn$Mt19937_64(0L)
    random <- nn$from_vector(vapply(seq_len(n),function(i) rng$normal32(),numeric(1)),c(1L,n))
    stopifnot(identical(nn$dropout(random,.5,FALSE,generator)$values(),random$values()))
    cat("Inference dropout is exactly the identity for all",n,"entries.\n")
}

train_mlp <- function(p,seed,x_train,y_train,x_test,y_test,steps,mc=FALSE) {
    parameters <- nn$ParameterStore(seed);drop1 <- nn$Generator(seed*10L+1L);drop2 <- nn$Generator(seed*10L+2L)
    hidden <- nn$DenseOptions();hidden$activation <- nn$Activation$Relu;hidden$init <- nn$init$he_uniform()
    forward <- function(x,training) {
        h <- nn$dense(parameters,"h1",x,256L,hidden)
        h <- nn$dropout(h,p,training,drop1)
        h <- nn$dense(parameters,"h2",h,256L,hidden)
        h <- nn$dropout(h,p,training,drop2)
        nn$dense(parameters,"out",h,1L)
    }
    x <- nn$from_vector(x_train,c(length(x_train),1L));y <- nn$from_vector(y_train,c(length(y_train),1L))
    xt <- nn$from_vector(x_test,c(length(x_test),1L));yt <- nn$from_vector(y_test,c(length(y_test),1L))
    nn$no_grad(function() forward(x,FALSE))
    cfg <- nn$OptimizerConfig();cfg$kind <- nn$OptimizerKind$Adam
    adam <- nn$Optimizer(parameters,1e-3,cfg)
    for (i in seq_len(steps)) adam$minimize(nn$mse(forward(x,TRUE),y))
    stopifnot(adam$iterations()==steps)
    nn$no_grad(function() {
        train <- nn$mse(forward(x,FALSE),y)$item();test <- nn$mse(forward(xt,FALSE),yt)$item();std <- numeric()
        stopifnot(is.finite(train),is.finite(test))
        if (mc) {
            samples <- 100L;total <- numeric(length(x_test));square <- numeric(length(x_test))
            for (i in seq_len(samples)) {
                values <- forward(xt,TRUE)$values();total <- total+values;square <- square+values*values
            }
            std <- sqrt(pmax(0,square/samples-(total/samples)^2))
        }
        list(train=train,test=test,std=std)
    })
}

regularization <- function(steps) {
    rng <- nn$Mt19937_64(42L);x_train <- numeric(40L);y_train <- numeric(40L)
    for (i in 1:40) {
        x <- rng$uniform32(-1,1)
        value <- nn$sin(nn$mul(nn$scalar(x),3))$item()
        y <- nn$scalar(value+rng$normal32(0,.3))$item()
        x_train[i] <- x;y_train[i] <- y
    }
    x_test <- numeric(201L);y_test <- numeric(201L)
    for (i in 0:200) {
        x <- nn$add(nn$div(nn$mul(nn$scalar(i),3),200),-1.5)$item()
        x_test[i+1L] <- x;y_test[i+1L] <- nn$sin(nn$mul(nn$scalar(x),3))$item()
    }
    inside <- abs(x_test)<=1;xin <- x_test[inside];yin <- y_test[inside];means <- list()
    cat(sprintf("MLP 1-256-256-1, 40 noisy sin(3x) points; %d Adam updates per fit.\n",steps))
    for (p in c(0,.5)) {
        train_sum <- 0;test_sum <- 0
        for (seed in 1:3) {
            fit <- train_mlp(p,seed,x_train,y_train,xin,yin,steps)
            train_sum <- train_sum+fit$train/3;test_sum <- test_sum+fit$test/3
            cat(sprintf("p=%.6g, seed=%d: train MSE=%.9g, in-range test MSE=%.9g\n",p,seed,fit$train,fit$test))
        }
        means[[length(means)+1L]] <- c(train_sum,test_sum)
        cat("p=",p,"three-seed mean train/test:",train_sum,test_sum,"\n")
    }
    std <- train_mlp(.5,1L,x_train,y_train,x_test,y_test,steps,TRUE)$std
    stopifnot(all(is.finite(std)))
    cat("Monte Carlo dropout, 100 stochastic forward passes:\n")
    for (i in seq.int(1L,201L,20L)) cat(sprintf("x=%.6g, predictive std=%.9g\n",x_test[i],std[i]))
    if (steps>=3000L) stopifnot(means[[1L]][1L]<means[[2L]][1L],means[[2L]][2L]<means[[1L]][2L],means[[2L]][2L]<.1)
    stopifnot(std[1L]>std[101L],std[201L]>std[101L])
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(epochs=5L,full=FALSE))
    if (is.na(args$epochs)||args$epochs<1L) stop("epochs must be positive")
    nn$set_num_threads(2L)
    statistics();regularization(args$epochs)
    cat("All dropout checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
