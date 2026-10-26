# Native R NN example. Gradient clipping and unequal microbatch accumulation.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

near <- function(actual,expected,tolerance=1e-6) stopifnot(abs(actual-expected)<=tolerance)

clipping <- function() {
    gradient <- nn$from_vector(c(3,4))
    result <- nn$clip_by_global_norm(nn$tensor_vector(list(gradient)),2)
    clipped <- nn$tensor_list(result$first); norm <- result$second
    near(norm,5,0); near(2/max(norm,2),.4)
    stopifnot(max(abs(clipped[[1L]]$values()-c(1.2,1.6)))<1e-6)
    values <- nn$tensor_list(nn$clip_by_value(nn$tensor_vector(list(gradient)),-2,2))
    stopifnot(identical(values[[1L]]$values(),c(2,2)))
    split <- nn$clip_by_global_norm(nn$tensor_vector(list(nn$scalar(3),nn$scalar(4))),2)
    tensors <- nn$tensor_list(split$first)
    near(tensors[[1L]]$item(),1.2); near(tensors[[2L]]$item(),1.6)
    zero <- nn$clip_by_global_norm(nn$tensor_vector(list(nn$zeros(2L))),2)
    tensors <- nn$tensor_list(zero$first)
    stopifnot(identical(tensors[[1L]]$values(),c(0,0)),zero$second==0,2/max(zero$second,2)==1)
    cat("(3,4), limit 2: global norm -> (1.2,1.6); by value -> (2,2).\n")
    cat("Split tensors share one norm; zero gradients remain finite.\n")
}

slots <- function(optimizer) {
    state <- optimizer$state(); result <- list()
    for (i in seq_len(state$size())-1L) {
        pair <- state$`__getitem__`(i)
        if (!pair$first %in% c("step","nadam_product")) result[[pair$first]] <- pair$second$clone()
    }
    result
}

accumulation <- function(steps) {
    w <- nn$scalar(3)$requires_grad_(); b <- nn$scalar(-1)$requires_grad_()
    wr <- nn$scalar(3)$requires_grad_(); br <- nn$scalar(-1)$requires_grad_()
    loss <- function(weight,bias,x,y) nn$mean(nn$square(nn$sub(nn$add(nn$mul(x,weight),bias),y)))
    config <- nn$OptimizerConfig(); config$kind <- nn$OptimizerKind$Adam
    micro <- nn$Optimizer(nn$tensor_vector(list(w,b)),.01,config)
    batch <- nn$Optimizer(nn$tensor_vector(list(wr,br)),.01,config)
    accumulator <- nn$GradientAccumulator()
    apply <- function() {
        mean <- accumulator$mean()
        clipped <- nn$clip_by_global_norm(mean,1)
        micro$step(clipped$first)
        accumulator$reset()
    }
    stopifnot(tryCatch({apply();FALSE},error=function(e) TRUE),micro$iterations()==0,w$item()==3)
    x <- nn$from_vector(c(-3,-2,-1,0,1,2,3)); y <- nn$from_vector(c(-5,-3,-1,1,3,5,7))
    rejected <- tryCatch({
        accumulator$add(nn$tensor_vector(nn$grad(loss(w,b,x,y),list(w,b))),0); FALSE
    },error=function(e) TRUE)
    stopifnot(rejected,accumulator$total_weight()==0)
    worst <- 0
    for (update in 0:(steps-1L)) {
        start <- 0L
        for (size in c(2L,3L,2L)) {
            bx <- nn$slice_axis(x,0L,start,start+size); by <- nn$slice_axis(y,0L,start,start+size)
            accumulator$add(nn$tensor_vector(nn$grad(loss(w,b,bx,by),list(w,b))),size)
            start <- start+size
        }
        if (update==0L) {
            mean <- nn$tensor_list(accumulator$mean())
            near(mean[[1L]]$item(),8); near(mean[[2L]]$item(),-4)
        }
        apply()
        clipped <- nn$clip_by_global_norm(nn$tensor_vector(nn$grad(loss(wr,br,x,y),list(wr,br))),1)
        batch$step(clipped$first)
        actual <- slots(micro); reference <- slots(batch)
        stopifnot(identical(names(actual),names(reference)),length(actual)==4L)
        got <- c(w$item(),b$item(),vapply(actual,function(t) t$item(),numeric(1)))
        want <- c(wr$item(),br$item(),vapply(reference,function(t) t$item(),numeric(1)))
        stopifnot(max(abs(got-want))<2e-6)
        worst <- max(worst,max(abs(got-want)))
        stopifnot(micro$iterations()==update+1L,batch$iterations()==update+1L,accumulator$total_weight()==0)
    }
    cat(sprintf("Microbatches 2+3+2 vs batch 7: max weight/moment error %.9g over %d Adam updates.\n",worst,steps))
    cat("Empty updates and zero weights rejected without changing optimizer state.\n")
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(epochs=5L,full=FALSE))
    if (is.na(args$epochs)||args$epochs<1L) stop("epochs must be positive")
    clipping(); accumulation(args$epochs)
    cat("All clipping and accumulation checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
