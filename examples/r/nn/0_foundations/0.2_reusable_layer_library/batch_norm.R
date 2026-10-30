# Native R NN example. Batch statistics, moving state, deep stacks and gradients.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.2_reusable_layer_library/batch_norm.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

EPS <- 1e-3
random_matrix <- function(rows,cols,rng,means,stds) {
    rng$reset_normal()
    z <- nn$from_vector(vapply(seq_len(rows*cols),function(i) rng$normal32(),numeric(1)),c(rows,cols))
    nn$add(nn$mul(z,nn$from_vector(stds)),nn$from_vector(means))
}
configuration <- function(momentum) {option <- nn$BatchNormOptions();option$momentum <- momentum;option$eps <- EPS;option}

hand_check <- function() {
    p <- nn$ParameterStore();stats <- nn$ParameterStore();rng <- nn$Mt19937_64(1L)
    x <- random_matrix(8L,4L,rng,c(1,-2,0,5),c(1,3,.1,2))
    infer <- nn$batch_norm(p,stats,"bn1",x,FALSE,configuration(.9))
    train <- nn$batch_norm(p,stats,"bn1",x,TRUE,configuration(.9))
    fused <- nn$batch_norm_train(x,p$at("bn1/gamma"),p$at("bn1/beta"),EPS)
    ey <- em <- eb <- eu <- ei <- el <- update <- 0
    for (c in 0:3) {
        values <- vapply(0:7,function(i) x$at(c(i,c)),numeric(1));average <- mean(values);variance <- mean((values-average)^2)
        for (i in 0:7) {
            ey <- max(ey,abs((values[i+1L]-average)/sqrt(variance+EPS)-fused$y$at(c(i,c))))
            el <- max(el,abs(train$at(c(i,c))-fused$y$at(c(i,c))))
            ei <- max(ei,abs(values[i+1L]/sqrt(1+EPS)-infer$at(c(i,c))))
        }
        reported <- fused$variance$at(c);em <- max(em,abs(average-fused$mean$at(c)))
        eb <- max(eb,abs(variance-reported));eu <- max(eu,abs(variance*8/7-reported))
        update <- max(update,abs(stats$at("bn1/moving_mean")$at(c)-.1*fused$mean$at(c)),abs(stats$at("bn1/moving_var")$at(c)-(.9+.1*reported)))
        cat("Channel",c,"mean / biased variance / unbiased variance / reported:",average,variance,variance*8/7,reported,"\n")
    }
    stopifnot(ey<1e-5,em<1e-5,eu<1e-5,eb>.1,ei<1e-5,el==0,update<1e-6)
    cat("Hand errors: output, mean, biased/unbiased variance, inference, layer, moving update:",ey,em,eb,eu,ei,el,update,"\n")
}

moving_statistics <- function(steps) {
    p <- nn$ParameterStore();stats <- nn$ParameterStore();rng <- nn$Mt19937_64(2L)
    means <- c(1,2,3,4);stds <- c(1,2,.5,3)
    expected_mean <- rep(0,4);expected_var <- rep(1,4)
    for (step in seq_len(steps)) {
        batch <- random_matrix(32L,4L,rng,means,stds)
        nn$no_grad(function() nn$batch_norm(p,stats,"bn1",batch,TRUE,configuration(.9)))
        values <- matrix(batch$values(),nrow=32L,ncol=4L,byrow=TRUE)
        expected_mean <- .9*expected_mean+.1*colMeans(values)
        expected_var <- .9*expected_var+.1*apply(values,2L,var)
        stopifnot(max(abs(stats$at("bn1/moving_mean")$values()-expected_mean))<2e-5,
                  max(abs(stats$at("bn1/moving_var")$values()-expected_var))<2e-5)
        if (step %in% c(1L,10L,30L,100L,steps)) {
            mm <- stats$at("bn1/moving_mean")$values();mv <- stats$at("bn1/moving_var")$values()
            cat("Statistics step",step,"moving mean:",mm,"moving variance:",mv,"\n")
        }
    }
    fresh <- random_matrix(4096L,4L,rng,means,stds)
    output <- nn$batch_norm(p,stats,"bn1",fresh,FALSE,configuration(.9))
    for (c in 0:3) {
        values <- vapply(0:4095,function(i) output$at(c(i,c)),numeric(1));average <- mean(values);std <- sqrt(mean(values*values)-average*average)
        expected <- vapply(0:4095,function(i) (fresh$at(c(i,c))-mm[c+1L])/sqrt(mv[c+1L]+EPS),numeric(1))
        error <- max(abs(values-expected));stopifnot(error<1e-5)
        cat("Inference channel",c,"mean/std:",average,std,"moving-stat hand error:",error,"\n")
    }
}

deep_stack <- function() {
    depth <- 24L;width <- 256L;batch <- 128L
    for (use_bn in c(FALSE,TRUE)) {
        p <- nn$ParameterStore(3L);stats <- nn$ParameterStore();rng <- nn$Mt19937_64(4L)
        h <- random_matrix(batch,width,rng,rep(0,width),rep(1,width));acts <- list()
        for (layer in 0:(depth-1L)) {
            w <- p$get(paste0("l",layer,"/W"),c(width,width),nn$init$normal(.5/sqrt(width)))
            z <- nn$matmul(h,w)
            if (use_bn) z <- nn$batch_norm(p,stats,paste0("bn",layer),z,TRUE,configuration(.99))
            h <- nn$relu(z);acts[[layer+1L]] <- h
        }
        gradient <- nn$grad(nn$mean(nn$square(h)),list(p$at("l0/W")))[[1L]]
        norm <- nn$sqrt(nn$sum(nn$square(gradient)))$item()
        scales <- vapply(c(1L,4L,8L,12L,16L,20L,24L),function(i) sqrt(mean(acts[[i]]$values()^2)),numeric(1))
        if (use_bn) stopifnot(abs(scales[7L]-.70)<.05,norm>1,norm<100) else stopifnot(scales[7L]<1e-8,norm<1e-12)
        cat(if(use_bn) "With BN" else "Without BN","RMS at layers 1,4,8,12,16,20,24:",scales,"first-weight gradient norm:",norm,"\n")
    }
}

backward <- function() {
    p <- nn$ParameterStore();stats <- nn$ParameterStore();rng <- nn$Mt19937_64(5L);n <- 6L;channels <- 3L
    x <- random_matrix(n,channels,rng,c(0,1,-1),c(1,2,.5))$requires_grad_()
    r <- random_matrix(n,channels,rng,c(0,0,0),c(1,1,1))
    loss <- function(value) nn$sum(nn$mul(nn$batch_norm(p,stats,"bn1",value,TRUE,configuration(.9)),r))
    objective <- loss(x);grads <- nn$grad(objective,list(x,p$at("bn1/gamma"),p$at("bn1/beta")))
    gx <- grads[[1L]];gg <- grads[[2L]];gb <- grads[[3L]];ex <- eg <- eb <- 0
    for (col in 0:(channels-1L)) {
        values <- vapply(0:(n-1L),function(i) x$at(c(i,col)),numeric(1));dy <- vapply(0:(n-1L),function(i) r$at(c(i,col)),numeric(1))
        average <- mean(values);variance <- mean((values-average)^2);sigma <- sqrt(variance+EPS)
        xhat <- (values-average)/sigma;sdy <- sum(dy);sdyx <- sum(dy*xhat)
        expected <- (n*dy-sdy-xhat*sdyx)/(n*sigma)
        actual <- vapply(0:(n-1L),function(i) gx$at(c(i,col)),numeric(1))
        ex <- max(ex,max(abs(expected-actual)));eg <- max(eg,abs(sdyx-gg$at(col)));eb <- max(eb,abs(sdy-gb$at(col)))
    }
    values <- x$values();dx <- gx$values();eps <- nn$scalar(1e-2)$item()
    fd <- nn$no_grad(function() {
        worst <- 0
        for (i in seq_along(values)) {
            plus <- minus <- values;plus[i] <- plus[i]+eps;minus[i] <- minus[i]-eps
            numeric <- (loss(nn$from_vector(plus,c(n,channels)))$item()-loss(nn$from_vector(minus,c(n,channels)))$item())/(2*eps)
            worst <- max(worst,abs(numeric-dx[i]))
        }
        worst
    })
    gmax <- max(abs(dx));sums <- vapply(0:(channels-1L),function(col) sum(vapply(0:(n-1L),function(i) gx$at(c(i,col)),numeric(1))),numeric(1))
    stopifnot(max(ex,eg,eb)<1e-5,fd/gmax<1e-3,is.finite(gmax),gmax>0,max(abs(sums))<1e-5)
    cat("Backward errors dx/gamma/beta:",ex,eg,eb,"finite difference:",fd,"relative:",fd/gmax,"gradient max:",gmax,"column sums:",sums,"\n")
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(epochs=5L,full=FALSE))
    if (is.na(args$epochs)||args$epochs<1L) stop("epochs must be positive")
    nn$set_num_threads(2L)
    hand_check();moving_statistics(args$epochs);deep_stack();backward()
    cat("All batch normalization checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
