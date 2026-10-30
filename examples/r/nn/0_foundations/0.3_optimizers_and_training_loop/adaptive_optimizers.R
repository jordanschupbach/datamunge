# Native R NN example. Adaptive optimizer formulas, state invariants and fitting.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.3_optimizers_and_training_loop/adaptive_optimizers.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

KINDS <- c("Adagrad","RMSProp","Adam","AdamW","Nadam","AMSGrad","Adamax")
near <- function(actual,expected,tolerance=2e-6) stopifnot(abs(actual-expected)<=tolerance)
config <- function(name) {cfg <- nn$OptimizerConfig(); cfg$kind <- nn$OptimizerKind[[name]]; cfg}
state_arrays <- function(opt) {
    state <- opt$state(); result <- list()
    for (i in seq_len(state$size())-1L) {
        pair <- state$`__getitem__`(i)
        result[[pair$first]] <- pair$second$clone()
    }
    result
}

fused_kernel <- function(name) {
    w <- nn$from_vector(c(1,-2,.5))$requires_grad_()
    cfg <- config(name); cfg$epsilon <- 1e-4
    opt <- nn$Optimizer(nn$tensor_vector(list(w)),.03,cfg)
    reference <- c(1,-2,.5); first <- numeric(3)
    second <- rep(if(name=="Adagrad") .1 else 0,3)
    eps <- cfg$epsilon; rho <- cfg$rho; b1 <- cfg$beta1; b2 <- cfg$beta2
    worst <- 0
    for (t in 1:40) {
        gradient <- nn$from_vector(c(if(t%%3==0) 2 else -.2,if(t%%4==0) 0 else .005,0))
        rate <- nn$scalar(if(t<20) .03 else .01)$item()
        opt$set_learning_rate(rate); opt$step(nn$tensor_vector(list(gradient)))
        b1p <- b1^t; b2p <- b2^t
        for (i in 1:3) {
            g <- gradient$values()[i]
            if (name=="Adagrad") {
                second[i] <- second[i]+g*g
                reference[i] <- reference[i]-rate*g/(sqrt(second[i])+eps)
            } else if (name=="RMSProp") {
                second[i] <- second[i]+(g*g-second[i])*(1-rho)
                first[i] <- rate*g/sqrt(second[i]+eps)
                reference[i] <- reference[i]-first[i]
            } else {
                lr_t <- rate*sqrt(1-b2p)/(1-b1p); eps_hat <- eps*sqrt(1-b2p)
                first[i] <- first[i]+(g-first[i])*(1-b1)
                second[i] <- second[i]+(g*g-second[i])*(1-b2)
                reference[i] <- reference[i]-lr_t*first[i]/(sqrt(second[i])+eps_hat)
            }
        }
        stopifnot(max(abs(w$values()-reference))<3e-6)
        worst <- max(worst,max(abs(w$values()-reference)))
    }
    cat(sprintf("%s: max fused-kernel error over 40 diagnostic updates=%.9g\n",name,worst))
}

invariants <- function(name) {
    w <- nn$from_vector(c(1,-2))$requires_grad_(); twin <- nn$from_vector(c(1,-2))$requires_grad_()
    cfg <- config(name)
    if (name=="AdamW") cfg$weight_decay <- .1
    opt <- nn$Optimizer(nn$tensor_vector(list(w,twin)),.1,cfg)
    fresh <- opt$state()
    feed <- function(values) {
        gradient <- nn$from_vector(values)
        opt$step(nn$tensor_vector(list(gradient,gradient)))
    }
    feed(c(3,-4))
    for (i in 1:2) {
        g <- c(3,-4)[i]; initial <- c(1,-2)[i]
        if (name=="Adagrad") {
            direction <- g/(sqrt(.1+g*g)+cfg$epsilon)
        } else if (name=="RMSProp") {
            direction <- g/sqrt((1-cfg$rho)*g*g+cfg$epsilon)
        } else {
            scale <- 1
            if (name=="Nadam") {
                mu <- cfg$beta1*(1-.5*.96); mu_next <- cfg$beta1*(1-.5*.96*.96)
                scale <- scale+mu_next*(1-cfg$beta1)/(1-mu*mu_next)
            }
            direction <- scale*g/(abs(g)+cfg$epsilon)+cfg$weight_decay*initial
        }
        near(w$values()[i],initial-.1*direction); near(w$values()[i],twin$values()[i],1e-7)
    }
    if (name %in% c("AMSGrad","Adamax")) {
        slot <- if(name=="AMSGrad") "vmax/0" else "u/0"
        previous <- -1
        for (t in 0:7) {
            value <- state_arrays(opt)[[slot]]$values()[1L]
            if (name=="AMSGrad") stopifnot(value>=previous) else near(value,3*cfg$beta2^t)
            previous <- value; feed(c(0,0))
        }
    }
    nn$no_grad(function() {w$copy_(nn$from_vector(c(1,-2))); twin$copy_(nn$from_vector(c(1,-2)))})
    opt$load_state(fresh)
    for (t in 1:5) feed(c(0,0))
    near(w$values()[1L],(1-.1*cfg$weight_decay)^5)
    stopifnot(opt$iterations()==5L)
    cat(name,"first update, independent slots, reset and zero-gradient checks passed.\n")
}

fit <- function(name,steps) {
    w <- nn$from_vector(c(4,-4))$requires_grad_()
    target <- nn$from_vector(c(1.5,-.75)); weight <- nn$from_vector(c(1,30))
    loss <- function() nn$sum(nn$mul(weight,nn$square(nn$sub(w,target))))
    cfg <- config(name)
    if (name=="AdamW") cfg$weight_decay <- .01
    opt <- nn$Optimizer(nn$tensor_vector(list(w)),.1,cfg)
    initial <- loss()$item()
    for (t in 0:(steps-1L)) {
        rate <- nn$scalar(if(t<3200L) .1 else if(t<3600L) .01 else .001)$item()
        opt$set_learning_rate(rate); opt$minimize(loss())
    }
    stopifnot(loss()$item()<initial,opt$iterations()==steps)
    values <- w$values()
    if (steps>=4000L) {near(values[1L],1.5,.04); near(values[2L],-.75,.04)}
    arrays <- state_arrays(opt)
    slots <- sum(vapply(setdiff(names(arrays),c("step",if(name!="Nadam") "nadam_product")),function(key) arrays[[key]]$numel(),numeric(1)))
    cat(sprintf("%s: %d updates, w=(%.9g,%.9g), loss=%.9g, slots=%d floats + int64 step\n",name,steps,values[1L],values[2L],loss()$item(),slots))
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(epochs=5L,full=FALSE))
    if (is.na(args$epochs)||args$epochs<1L) stop("epochs must be positive")
    for (name in KINDS[1:3]) fused_kernel(name)
    for (name in KINDS) invariants(name)
    for (name in KINDS) fit(name,args$epochs)
    cat("All adaptive optimizer checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
