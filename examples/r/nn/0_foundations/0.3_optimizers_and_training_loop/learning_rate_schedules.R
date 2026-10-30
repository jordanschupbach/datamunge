# Native R NN example. Schedule boundaries and explicit scheduled SGD.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.3_optimizers_and_training_loop/learning_rate_schedules.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

boundaries <- function() {
    s <- nn$schedule
    actual <- c(s$step_decay(1,.5,10L,9L),s$step_decay(1,.5,10L,10L),
        s$exponential(1,.25,10L,5L),s$cosine(1,.1,100L,0L),
        s$cosine(1,.1,100L,50L),s$cosine(1,.1,100L,200L),
        s$warmup(1,10L,0L),s$warmup(1,0L,0L),s$warmup_cosine(1,.1,10L,100L,10L),
        s$one_cycle(.1,1,.01,30L,100L,30L),s$one_cycle(.1,1,.01,30L,100L,200L),
        s$cyclical(.1,1,10L,10L),s$cyclical(.1,1,10L,20L),s$cyclical(.1,1,10L,30L,.5))
    stopifnot(max(abs(actual-c(1,.5,.5,1,.55,.1,0,1,1,1,.01,1,.1,.55)))<1e-12)
    invalid <- list(function() s$step_decay(1,.5,0L,1L),function() s$cosine(1,0,0L,1L),
        function() s$warmup_cosine(1,0,10L,10L,1L),function() s$one_cycle(1,.1,0,5L,10L,0L),
        function() s$cyclical(.1,1,0L,0L),function() s$warmup(1,10L,-1L))
    for (call in invalid) stopifnot(tryCatch({call();FALSE},error=function(e) TRUE))
    cat("Endpoint, transition, clamping, cycle and invalid-input checks passed.\n")
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(epochs=5L,full=FALSE))
    if (is.na(args$epochs)||args$epochs<1L) stop("epochs must be positive")
    boundaries()
    s <- nn$schedule
    schedules <- list(step=function(t) s$step_decay(.1,.5,25L,t),
        exponential=function(t) s$exponential(.1,.5,25L,t),
        cosine=function(t) s$cosine(.1,.001,100L,t),
        'warmup+cosine'=function(t) s$warmup_cosine(.1,.001,10L,100L,t),
        'one-cycle'=function(t) s$one_cycle(.01,.1,.0001,30L,100L,t),
        cyclical=function(t) s$cyclical(.01,.1,20L,t),
        triangular2=function(t) s$cyclical(.01,.1,20L,t,.5))
    ticks <- c(0L,10L,20L,30L,50L,80L,100L,120L)
    cat("Learning rates by completed update:",ticks,"\n")
    for (name in names(schedules)) {
        schedule <- schedules[[name]]
        cat(name,vapply(ticks,schedule,numeric(1)),"\n")
        w <- nn$scalar(2)$requires_grad_()
        optimizer <- nn$Optimizer(nn$tensor_vector(list(w)),0)
        expected <- 2
        for (step in 0:(args$epochs-1L)) {
            stopifnot(optimizer$iterations()==step)
            rate <- nn$scalar(schedule(step))$item()
            expected <- expected*(1-rate)
            optimizer$set_learning_rate(rate)
            optimizer$minimize(nn$mul(nn$square(w),.5))
        }
        stopifnot(optimizer$iterations()==args$epochs,abs(w$item()-expected)<2e-6,abs(expected)<=2)
        if (args$epochs>=100L) stopifnot(abs(expected)<.5)
        cat(sprintf("%s: %d SGD updates, w=%.9g, closed form=%.9g\n",name,args$epochs,w$item(),expected))
    }
    cat("All learning-rate schedule checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
