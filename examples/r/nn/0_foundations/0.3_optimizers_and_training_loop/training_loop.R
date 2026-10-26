# Native R NN example. Resumable batching, weighted logging and explicit training.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

batching_checks <- function() {
    original <- nn$BatchOrder(11L,4L,7L); restored <- nn$BatchOrder(11L,4L,999L)
    original$begin_epoch(); first <- original$`_next`()
    restored$restore_state(original$save_state()); seen <- first
    repeat {
        batch <- original$`_next`(); stopifnot(identical(batch,restored$`_next`()))
        if (!length(batch)) break
        stopifnot(!length(intersect(seen,batch))); seen <- c(seen,batch)
    }
    stopifnot(identical(as.integer(sort(seen)),0L:10L))
    for (epoch in 1:3) {
        original$begin_epoch(); restored$begin_epoch()
        repeat {
            batch <- original$`_next`(); stopifnot(identical(batch,restored$`_next`()))
            if (!length(batch)) break
        }
    }
    state <- restored$save_state()
    rejected <- tryCatch({restored$restore_state(paste0(state," bad"));FALSE},error=function(e) TRUE)
    stopifnot(rejected,identical(restored$save_state(),state))
    drop <- nn$BatchOrder(10L,4L,0L,FALSE,TRUE); drop$begin_epoch()
    stopifnot(identical(as.integer(drop$`_next`()),0L:3L),identical(as.integer(drop$`_next`()),4L:7L),!length(drop$`_next`()))
    singleton <- nn$BatchOrder(1L,16L,0L); singleton$begin_epoch()
    stopifnot(length(singleton$`_next`())==1L,!length(singleton$`_next`()))
    source <- nn$reshape(nn$arange(0,6,1,nn$DType$Int32),c(3L,2L))
    stopifnot(identical(as.numeric(nn$take_rows(source,c(2L,0L,2L))$values()),c(4,5,0,1,4,5)))
    stopifnot(tryCatch({nn$take_rows(source,3L);FALSE},error=function(e) TRUE))
    mean <- nn$WeightedMean(); mean$add(2,4L); mean$add(10,1L)
    stopifnot(abs(mean$value()-3.6)<1e-12,mean$count()==5L)
    mean$reset(); stopifnot(mean$count()==0)
    cat("Exact mid-epoch resume and three future shuffles passed.\n")
    cat("Coverage, drop_last, singleton, row gathering, invalid state and weighted means passed.\n")
}

train <- function(epochs) {
    index <- nn$arange(0,103)
    a <- nn$sin(nn$mul(index,.7)); b <- nn$cos(nn$mul(index,.4))
    inputs <- nn$stack(nn$tensor_vector(list(a,b)),1L)
    labels <- nn$reshape(nn$add(nn$sub(nn$mul(a,1.5),nn$mul(b,.7)),.2),c(103L,1L))
    w <- nn$zeros(c(2L,1L))$requires_grad_(); bias <- nn$zeros(1L)$requires_grad_()
    loss_of <- function(x,y) nn$mean(nn$square(nn$sub(nn$bias_add(nn$matmul(x,w),bias),y)))
    cfg <- nn$OptimizerConfig(); cfg$kind <- nn$OptimizerKind$Adam
    optimizer <- nn$Optimizer(nn$tensor_vector(list(w,bias)),.03,cfg)
    sampler <- nn$BatchOrder(103L,16L,42L); initial <- loss_of(inputs,labels)$item()
    for (epoch in seq_len(epochs)) {
        sampler$begin_epoch(); epoch_loss <- nn$WeightedMean(); visited <- numeric(); batches <- 0L
        repeat {
            rows <- sampler$`_next`()
            if (!length(rows)) break
            stopifnot(!length(intersect(visited,rows))); visited <- c(visited,rows)
            loss <- loss_of(nn$take_rows(inputs,rows),nn$take_rows(labels,rows))
            optimizer$minimize(loss); epoch_loss$add(loss$item(),length(rows)); batches <- batches+1L
        }
        stopifnot(identical(as.integer(sort(visited)),0L:102L),epoch_loss$count()==103L,batches==7L)
        cat(sprintf("epoch %d: batches=%d, samples=%d, online mean loss=%.9g\n",epoch,batches,epoch_loss$count(),epoch_loss$value()))
    }
    final <- loss_of(inputs,labels)$item(); weights <- w$values()
    stopifnot(is.finite(final),final<initial,optimizer$iterations()==7L*epochs)
    if (epochs>=50L) stopifnot(final<1e-9,max(abs(weights-c(1.5,-.7)))<1e-4,abs(bias$item()-.2)<1e-4)
    cat(sprintf("Final fixed-weight MSE=%.9g, weights=(%.9g,%.9g), bias=%.9g, updates=%d\n",final,weights[1L],weights[2L],bias$item(),optimizer$iterations()))
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(epochs=5L,full=FALSE))
    if (is.na(args$epochs)||args$epochs<1L) stop("epochs must be positive")
    nn$set_num_threads(2L)
    batching_checks(); train(args$epochs)
    cat("All training-loop checks passed.\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
