# Native R NN example. LAMB/LARS trust ratios, Lion signs and Adafactor moments.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.3_optimizers_and_training_loop/advanced_optimizers.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

KINDS <- c("Lamb","Lars","Lion","Adafactor")
near <- function(actual,expected,tolerance=1e-6) stopifnot(abs(actual-expected)<=tolerance)
config <- function(name) {cfg <- nn$OptimizerConfig(); cfg$kind <- nn$OptimizerKind[[name]]; cfg}
state_arrays <- function(opt) {
    state <- opt$state(); result <- list()
    for (i in seq_len(state$size())-1L) {
        pair <- state$`__getitem__`(i); result[[pair$first]] <- pair$second$clone()
    }
    result
}
slot_floats <- function(opt) {
    arrays <- state_arrays(opt)
    sum(vapply(setdiff(names(arrays),c("step","nadam_product")),function(key) arrays[[key]]$numel(),numeric(1)))
}

trust_ratio <- function(name) {
    w <- nn$from_vector(c(3,4))$requires_grad_()
    cfg <- config(name); cfg$momentum <- 0; cfg$trust_coefficient <- .1
    opt <- nn$Optimizer(nn$tensor_vector(list(w)),.1,cfg)
    opt$step(nn$tensor_vector(list(nn$from_vector(c(1,-2)))))
    relative <- sqrt(sum((w$values()-c(3,4))^2))/5
    near(relative,if(name=="Lamb") .1 else .01,1e-7)
    cat(name,"relative update norm:",relative,"\n")
}

lion_sign <- function() {
    w <- nn$scalar(1)$requires_grad_()
    cfg <- config("Lion"); cfg$beta2 <- .99
    opt <- nn$Optimizer(nn$tensor_vector(list(w)),.1,cfg)
    for (gradient in c(10,-.85)) opt$step(nn$tensor_vector(list(nn$scalar(gradient))))
    near(w$item(),.8); near(state_arrays(opt)[['m/0']]$item(),.0905)
    cat("Lion uses old momentum for signs; gradients 10 then -.85 produce w=.8.\n")
}

factorization <- function() {
    w <- nn$ones(c(2L,3L))$requires_grad_()
    cfg <- config("Adafactor"); cfg$scale_parameter <- FALSE; cfg$relative_step <- FALSE; cfg$update_clip <- .5
    opt <- nn$Optimizer(nn$tensor_vector(list(w)),.1,cfg)
    opt$step(nn$tensor_vector(list(nn$from_vector(c(1,2,3,2,4,6),c(2L,3L)))))
    state <- state_arrays(opt)
    stopifnot(max(abs(w$values()-.95))<1e-6)
    row <- state[['row/0']]$values(); col <- state[['col/0']]$values()
    near(row[1L],14/3); near(row[2L],56/3,2e-6)
    stopifnot(max(abs(col-2.5*(1:3)^2))<1e-6,slot_floats(opt)==5L)
    cat("Adafactor rank-one preconditioner, clipped RMS update .5, five slots for 2x3 matrix passed.\n")
}

zeros_and_memory <- function() {
    for (name in KINDS) {
        for (shape in list(integer(),3L,c(2L,3L,4L))) {
            w <- nn$zeros(shape)$requires_grad_()
            opt <- nn$Optimizer(nn$tensor_vector(list(w)),.01,config(name))
            for (i in 1:3) opt$step(nn$tensor_vector(list(nn$zeros_like(w))))
            stopifnot(all(w$values()==0))
        }
    }
    w <- nn$zeros(c(256L,128L))$requires_grad_()
    opt <- nn$Optimizer(nn$tensor_vector(list(w)),.01,config("Adafactor"))
    opt$step(nn$tensor_vector(list(nn$ones_like(w))))
    stopifnot(slot_floats(opt)==384L)
    cat("Zero-gradient scalar/vector/tensor checks passed; 256x128 Adafactor uses 384 moment floats.\n")
}

fit <- function(name,steps) {
    w <- nn$ones(c(3L,4L))$requires_grad_()
    target <- nn$reshape(nn$add(nn$mul(nn$sin(nn$arange(0,12)),.5),.3),c(3L,4L))
    loss <- function() nn$mean(nn$square(nn$sub(w,target)))
    cfg <- config(name); cfg$trust_coefficient <- .1
    if (name=="Lion") cfg$beta2 <- .99
    initial_lr <- nn$scalar(if(name=="Lion") .01 else if(name=="Lars") .1 else .03)$item()
    opt <- nn$Optimizer(nn$tensor_vector(list(w)),initial_lr,cfg); initial <- loss()$item()
    for (t in 0:(steps-1L)) {
        rate <- nn$mul(nn$sub(1,nn$div(nn$scalar(t),1500)),initial_lr)$item()
        opt$set_learning_rate(rate); opt$minimize(loss())
    }
    final <- loss()$item()
    stopifnot(is.finite(final),final<initial,opt$iterations()==steps)
    if (steps>=1500L) stopifnot(final<1e-5)
    cat(sprintf("%s: %d updates, initial lr=%.6g, final MSE=%.9g, persistent slots=%d floats\n",name,steps,initial_lr,final,slot_floats(opt)))
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(epochs=5L,full=FALSE))
    if (is.na(args$epochs)||args$epochs<1L||args$epochs>1500L) stop("epochs must be in [1,1500]")
    nn$set_num_threads(2L)
    trust_ratio("Lamb"); trust_ratio("Lars"); lion_sign(); factorization(); zeros_and_memory()
    for (name in KINDS) fit(name,args$epochs)
    cat("All advanced optimizer checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
