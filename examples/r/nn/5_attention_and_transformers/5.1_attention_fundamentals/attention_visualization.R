# Native R NN example. Attention maps and measured key-removal interventions.
# CMD: nix develop .#r --command Rscript examples/r/nn/5_attention_and_transformers/5.1_attention_fundamentals/attention_visualization.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

exercise_case <- function(args,equal_values) {
    rng <- nn$Mt19937(45901L)
    random <- function(shape) nn$from_vector(rng$normal32_values(prod(shape)),shape)
    q <- random(c(2L,4L,4L));k <- random(c(2L,5L,4L));v <- random(c(2L,5L,3L))
    if (equal_values) v <- nn$from_vector(rep(c(-1,0,1),10L),c(2L,5L,3L))
    keep <- numeric(40L);i <- 1L
    for(h in 0:1) for(t in 0:3) for(j in 0:4) {
        keep[i] <- as.numeric((t<3L&&j<=t+1L)||(h==1L&&t==3L&&j==0L));i <- i+1L
    }
    mask <- function(flags) nn$cast(nn$from_vector(flags,c(2L,4L,5L)),nn$DType$Bool)
    forward <- function(flags) {
        config <- nn$AttentionOptions();config$mask <- mask(flags);config$scale <- .5
        config$need_weights <- TRUE;config$impl <- nn$AttentionImpl$Plain
        nn$scaled_dot_product_attention(q,k,v,config)
    }
    result <- forward(keep)
    entropy <- nn$neg(nn$sum(nn$mul(result$weights,nn$log(nn$maximum(result$weights,nn$scalar(1e-30)))),-1L))
    support <- nn$mul(nn$exp(entropy),nn$cast(nn$greater(nn$sum(result$weights,-1L),0),nn$DType$Float32))
    maxw <- nn$max(result$weights,-1L)
    arrays <- list(q=q,k=k,v=v,allowed=mask(keep),weights=result$weights,output=result$output,entropy=entropy,effective_support=support,max_weight=maxw)
    effects <- numeric(40L)
    for(r in 0:4) {
        flags <- as.numeric(keep!=0 & (0:39)%%5L!=r)
        without <- forward(flags)$output;arrays[[paste0("without_",r)]] <- without
        difference <- nn$sub(result$output,without)$values()
        for(h in 0:1) for(t in 0:3) {
            start <- (h*4L+t)*3L
            effects[(h*4L+t)*5L+r+1L] <- sqrt(sum(difference[start+1:3]^2))
        }
    }
    arrays$removal_effect <- nn$from_vector(effects,c(2L,4L,5L))
    name <- if(equal_values) "equal_values" else "distinct_values"
    dir.create(args$out,recursive=TRUE,showWarnings=FALSE);nn$save_npz(file.path(args$out,paste0(name,".npz")),arrays)
    cat(name,":\n")
    for(h in 0:1) {
        cat(" head",h,"weights | entropy support max weight | key removal effects\n")
        for(t in 0:3) {
            weights <- vapply(0:4,function(j) result$weights$at(c(h,t,j)),numeric(1))
            stats <- c(entropy$at(c(h,t)),support$at(c(h,t)),maxw$at(c(h,t)))
            removal <- effects[(h*4L+t)*5L+1:5]
            cat("  q",t,":",sprintf("%.3f",weights),"|",sprintf("%.3f",stats),"|",sprintf("%.3f",removal),"\n")
        }
    }
    stopifnot(support$at(c(0L,3L))==0,abs(support$at(c(1L,3L))-1)<1e-6)
    if(equal_values) for(h in 0:1) for(t in 0:2) stopifnot(all(effects[(h*4L+t)*5L+1:5]<2e-6))
    cat(name,": maps, entropy, effective support and five key-removal interventions PASS\n")
    fixture <- file.path(args$tf,paste0(name,".npz"))
    if (file.exists(fixture)) {
        reference <- nn$load_npz(fixture);inputs <- c("q","k","v")
        difference <- function(a,b) {if(!identical(a$shape(),b$shape())) return(Inf);max(abs(a$values()-b$values()),0)}
        input_error <- max(vapply(intersect(inputs,names(reference)),function(key) difference(arrays[[key]],reference[[key]]),numeric(1)),0)
        if (input_error==0) {
            worst <- 0
            for (key in setdiff(intersect(names(arrays),names(reference)),c(inputs,"native_fd","metadata"))) {
                delta <- difference(arrays[[key]],reference[[key]]);worst <- max(worst,delta)
                if (args$verbose!=0L) cat(key,"max difference vs TF:",delta,"\n")
            }
            stopifnot(worst<=1e-5);cat("TF comparison with identical inputs, max error:",worst,"\n")
        } else cat("TF comparison skipped: input random streams differ by",input_error,"\n")
    }
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(out=file.path(nn_output_root(),"attention_visualization"),tf=file.path(path.expand("~"),"projects/tflow_examples/output/models/attention_visualization"),verbose=0L,epochs=5L,full=FALSE))
    nn$set_num_threads(2L)
    exercise_case(args,FALSE);exercise_case(args,TRUE)
    0L
}
if(sys.nframe()==0L) quit(status=main())
