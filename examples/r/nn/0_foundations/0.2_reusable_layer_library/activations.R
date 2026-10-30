# Native R NN example. Activation values, derivatives, statistics and deep stacks.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.2_reusable_layer_library/activations.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

LAMBDA <- 1.0507009873554804934193349852946
ALPHA <- 1.6732632423543772848170429916717
LEAKY <- nn$scalar(.2)$item()
NAMES <- c("Relu","Relu6","LeakyRelu","Elu","Selu","Gelu","GeluTanh","Silu","Mish","Softplus","Softsign","Sigmoid","Tanh")
sigmoid <- function(x) 1/(1+exp(-x))
softplus <- function(x) ifelse(x>30,x,log1p(exp(x)))

value_and_derivative <- function(name,x) {
    switch(name,
        Relu=list(pmax(x,0),as.numeric(x>0)),
        Relu6=list(pmin(pmax(x,0),6),as.numeric(x>0&x<6)),
        LeakyRelu=list(ifelse(x>0,x,LEAKY*x),ifelse(x>0,1,LEAKY)),
        Elu=list(ifelse(x>0,x,expm1(x)),ifelse(x>0,1,exp(x))),
        Selu=list(LAMBDA*ifelse(x>0,x,ALPHA*expm1(x)),LAMBDA*ifelse(x>=0,1,ALPHA*exp(x))),
        Gelu=list(x*pnorm(x),pnorm(x)+x*dnorm(x)),
        GeluTanh={u <- sqrt(2/pi)*(x+.044715*x*x*x);du <- sqrt(2/pi)*(1+3*.044715*x*x);t <- tanh(u)
            list(.5*x*(1+t),.5*(1+t)+.5*x*(1-t*t)*du)},
        Silu={s <- sigmoid(x);list(x*s,s*(1+x*(1-s)))},
        Mish={t <- tanh(softplus(x));list(x*t,t+x*(1-t*t)*sigmoid(x))},
        Softplus=list(softplus(x),sigmoid(x)),
        Softsign=list(x/(1+abs(x)),1/(1+abs(x))^2),
        Sigmoid={s <- sigmoid(x);list(s,s*(1-s))},
        Tanh={t <- tanh(x);list(t,1-t*t)},stop("unknown activation ",name))
}

table <- function() {
    grid <- c(-4,-2,-1,-.5,0,.5,1,2,4)
    cat("Activation grid:",grid,"\n")
    for (name in NAMES) {
        x <- nn$from_vector(grid)$requires_grad_()
        y <- nn$activate(x,nn$Activation[[name]],LEAKY)
        dy <- nn$grad(nn$sum(y),list(x))[[1L]]
        expected <- value_and_derivative(name,grid)
        value_error <- max(abs(y$values()-expected[[1L]]));gradient_error <- max(abs(dy$values()-expected[[2L]]))
        stopifnot(value_error<2e-6,gradient_error<2e-6)
        cat(name,"values:",y$values(),"derivatives:",dy$values(),"max errors:",value_error,gradient_error,"\n")
    }
}

properties <- function() {
    rng <- nn$Mt19937_64(1L)
    samples <- vapply(seq_len(200000L),function(i) rng$normal64(),numeric(1))
    cat("Properties on 200000 standard-normal samples: min, max, mean, std, dead fraction, max derivative\n")
    for (name in NAMES) {
        result <- value_and_derivative(name,samples);values <- result[[1L]];derivatives <- result[[2L]]
        average <- mean(values);std <- sqrt(mean(values*values)-average*average);fraction <- mean(derivatives==0)
        if (name=="Selu") stopifnot(abs(average)<.02,abs(std-1)<.02)
        if (name=="Relu") stopifnot(abs(fraction-.5)<.01)
        cat(name,min(values),max(values),average,std,fraction,max(derivatives),"\n")
    }
}

selu_depth <- function() {
    depth <- 30L;width <- 256L;batch <- 256L
    rng <- nn$Mt19937_64(2L)
    inputs <- nn$from_vector(vapply(seq_len(batch*width),function(i) rng$normal32(),numeric(1)),c(batch,width))
    nn$no_grad(function() {
        for (name in c("Selu","Elu","Tanh","Relu")) {
            parameters <- nn$ParameterStore(5L);h <- inputs;means <- numeric(depth);stds <- numeric(depth)
            for (layer in 0:(depth-1L)) {
                option <- nn$DenseOptions();option$init <- nn$init$lecun_normal();option$bias <- FALSE;option$activation <- nn$Activation[[name]]
                h <- nn$dense(parameters,paste0("l",layer),h,width,option)
                double <- nn$cast(h,nn$DType$Float64);average <- nn$mean(double)$item()
                means[layer+1L] <- average;stds[layer+1L] <- sqrt(nn$mean(nn$square(double))$item()-average*average)
            }
            if (name=="Selu") stopifnot(all(abs(means)<.1),all(stds>.85&stds<1.15)) else stopifnot(stds[depth]<.25)
            cat(name,"layer, mean, std:\n")
            for (i in c(1L,2L,5L,10L,20L,30L)) cat(i,means[i],stds[i],"\n")
        }
    })
}

main <- function(arguments=commandArgs(TRUE)) {
    nn$set_num_threads(2L)
    table();properties();selu_depth()
    cat("All activation checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
