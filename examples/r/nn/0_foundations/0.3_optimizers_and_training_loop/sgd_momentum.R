# Native R NN example. SGD, momentum and Nesterov equations and minibatches.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

KINDS <- c("Sgd","Momentum","Nesterov")
configuration <- function(name) {config <- nn$OptimizerConfig(); config$kind <- nn$OptimizerKind[[name]]; config}
near <- function(actual,expected,tolerance=2e-6) stopifnot(abs(actual-expected)<=tolerance)
f <- function(value) nn$scalar(value)$item()
slot_floats <- function(opt) {
    state <- opt$state(); count <- 0
    for (i in seq_len(state$size())-1L) {
        pair <- state$`__getitem__`(i)
        if (!pair$first %in% c("step","nadam_product")) count <- count+pair$second$numel()
    }
    count
}

kernel_agreement <- function(name) {
    initial <- c(1,-2,.5); w <- nn$from_vector(initial)$requires_grad_()
    config <- configuration(name)
    opt <- nn$Optimizer(nn$tensor_vector(list(w)),.1,config); fresh <- opt$state()
    reference <- initial; velocity <- numeric(3); mu <- f(config$momentum); worst <- 0
    for (t in 0:19) {
        gradient <- nn$from_vector(c(if(t%%3L==0L) -.7 else .3,f(f(.1)*(t-8)),0))
        rate <- f(if(t<10L) .1 else .03)
        opt$set_learning_rate(rate); opt$step(nn$tensor_vector(list(gradient)))
        for (j in 1:3) {
            g <- gradient$values()[j]
            if (name=="Sgd") reference[j] <- f(reference[j]-f(rate*g)) else {
                velocity[j] <- f(f(velocity[j]*mu)+g)
                update <- if(name=="Nesterov") f(f(g*rate)+f(f(velocity[j]*mu)*rate)) else f(rate*velocity[j])
                reference[j] <- f(reference[j]-update)
            }
        }
        stopifnot(max(abs(w$values()-reference))<=2e-6,opt$iterations()==t+1L)
        worst <- max(worst,max(abs(w$values()-reference)))
    }
    cat(sprintf("%s: 20 changing-gradient diagnostic updates, max kernel error=%.9g, slots=%d floats\n",name,worst,slot_floats(opt)))
    nn$no_grad(function() w$copy_(nn$from_vector(initial)))
    opt$load_state(fresh)
    stopifnot(identical(w$values(),initial),opt$iterations()==0,slot_floats(opt)==0)
}

coupled_gradients <- function() {
    x <- nn$scalar(1)$requires_grad_(); y <- nn$scalar(2)$requires_grad_()
    opt <- nn$Optimizer(nn$tensor_vector(list(x,y)),.1)
    expected_x <- 1; expected_y <- 2
    for (t in 1:10) {
        next_x <- expected_x-.1*expected_y
        expected_y <- expected_y-.1*expected_x; expected_x <- next_x
        opt$minimize(nn$mul(x,y)); near(x$item(),expected_x); near(y$item(),expected_y)
    }
    cat("Coupled gradients use the same old parameter state.\n")
}

quadratic <- function(name,steps) {
    w <- nn$from_vector(c(4,4))$requires_grad_(); scale <- nn$from_vector(c(.5,50))
    opt <- nn$Optimizer(nn$tensor_vector(list(w)),.01,configuration(name))
    for (t in 0:steps) {
        loss <- nn$sum(nn$mul(scale,nn$square(w)))
        if (t %in% c(0L,10L,50L,steps)) {
            value <- loss$item(); stopifnot(is.finite(value))
            if (t==steps&&steps>=200L) stopifnot(value<.15)
            cat(name,"quadratic step",t,"loss",value,"\n")
        }
        if (t<steps) opt$minimize(loss)
    }
}

minibatches <- function(name,epochs) {
    w <- nn$scalar(0)$requires_grad_(); b <- nn$scalar(0)$requires_grad_()
    opt <- nn$Optimizer(nn$tensor_vector(list(w,b)),.03,configuration(name))
    ordering <- 0L:63L; rng <- nn$Mt19937(7L)
    for (epoch in seq_len(epochs)) {
        ordering <- rng$shuffle(ordering)
        stopifnot(identical(as.integer(sort(ordering)),0L:63L))
        for (start in seq.int(1L,64L,8L)) {
            x <- nn$sub(nn$div(nn$mul(nn$from_vector(ordering[start:(start+7L)]),2),63),1)
            y <- nn$sub(nn$mul(x,3),.5)
            prediction <- nn$add(nn$mul(x,w),b)
            opt$minimize(nn$mean(nn$square(nn$sub(prediction,y))))
        }
    }
    stopifnot(is.finite(w$item()),is.finite(b$item()),opt$iterations()==8L*epochs)
    if (epochs>=40L) {near(w$item(),3,.005);near(b$item(),-.5,.005)}
    cat(sprintf("%s: %d minibatch epochs, slope=%.9g, intercept=%.9g, updates=%d\n",name,epochs,w$item(),b$item(),opt$iterations()))
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(epochs=5L,full=FALSE))
    if (is.na(args$epochs)||args$epochs<1L) stop("epochs must be positive")
    for (name in KINDS) kernel_agreement(name)
    coupled_gradients()
    for (name in KINDS) quadratic(name,args$epochs)
    for (name in KINDS) minibatches(name,args$epochs)
    cat("All SGD/momentum/Nesterov checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
