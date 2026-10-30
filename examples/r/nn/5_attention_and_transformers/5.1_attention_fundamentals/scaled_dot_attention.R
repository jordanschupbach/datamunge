# Native R NN example. Scaled attention outputs and complete Q/K/V derivative fixtures.
# CMD: nix develop .#r --command Rscript examples/r/nn/5_attention_and_transformers/5.1_attention_fundamentals/scaled_dot_attention.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

exercise_case <- function(args,name,heads,strict,one_key,seed) {
    keys <- if(one_key) 1L else 5L
    qs <- if(heads) c(2L,2L,3L,4L) else c(2L,3L,4L)
    ks <- if(heads) c(2L,2L,keys,4L) else c(2L,keys,4L)
    vs <- if(heads) c(2L,2L,keys,3L) else c(2L,keys,3L)
    os <- if(heads) c(2L,2L,3L,3L) else c(2L,3L,3L)
    rng <- nn$Mt19937(seed)
    random <- function(shape,scale=1) nn$mul(nn$from_vector(rng$normal32_values(prod(shape)),shape),scale)
    q <- random(qs);k <- random(ks);v <- random(vs);probe <- random(os,.2)
    q$requires_grad_();k$requires_grad_();v$requires_grad_()
    mask <- nn$Tensor();bias <- nn$Tensor();allowed <- nn$Tensor()
    kl <- nn$cast(nn$from_vector(c(4,0)),nn$DType$Int32);ql <- nn$cast(nn$from_vector(c(2,3)),nn$DType$Int32)
    if (name=="biased") {bias <- nn$reshape(nn$mul(nn$sin(nn$mul(nn$arange(0,3*keys),.7)),.3),c(3L,keys));mask <- bias}
    if (strict) {
        flags <- numeric(2L*3L*keys);index <- 1L
        for (n in 0:1) for (t in 0:2) for (j in 0:(keys-1L)) {
            flags[index] <- as.numeric(t<ql$at(n)&&j<kl$at(n)&&j<=t);index <- index+1L
        }
        allowed <- nn$cast(nn$from_vector(flags,c(2L,3L,keys)),nn$DType$Bool)
        if (heads) allowed <- nn$expand_dims(allowed,1L)
        mask <- allowed
    }
    config <- nn$AttentionOptions();config$mask <- mask;config$scale <- .5;config$need_weights <- TRUE;config$impl <- nn$AttentionImpl$Plain
    forward <- function() nn$scaled_dot_product_attention(q,k,v,config)
    result <- forward();objective <- nn$sum(nn$mul(result$output,probe));gradients <- nn$grad(objective,list(q,k,v))
    arrays <- list(q=q$detach(),k=k$detach(),v=v$detach(),probe=probe,output=result$output$detach(),weights=result$weights$detach(),objective=objective$detach(),q_gradient=gradients[[1L]],k_gradient=gradients[[2L]],v_gradient=gradients[[3L]])
    if (strict) {arrays$allowed <- allowed;arrays$key_lengths <- kl;arrays$query_lengths <- ql}
    if (bias$defined()) arrays$bias <- bias
    epsilon <- nn$scalar(.002)$item()
    diagnostic <- nn$no_grad(function() {
        error <- 0;count <- 0L
        leaves <- list(q,k,v)
        for (field in 1:3) {
            value <- leaves[[field]];gradient <- gradients[[field]]
            flat <- nn$reshape(value,-1L);expected <- gradient$values()
            for (i in 0:5) {
                coordinate <- floor(i*(value$numel()-1)/5);cell <- nn$select(flat,0L,coordinate);old <- cell$item()
                cell$fill_(old+epsilon);plus <- nn$sum(nn$mul(forward()$output,probe))$item()
                cell$fill_(old-epsilon);minus <- nn$sum(nn$mul(forward()$output,probe))$item();cell$fill_(old)
                numeric <- nn$scalar(plus-minus)$item()/.004
                error <- max(error,abs(numeric-expected[coordinate+1L]));count <- count+1L
            }
        }
        list(error=error,count=count)
    })
    error <- diagnostic$error;count <- diagnostic$count;stopifnot(error<5e-4,count==18L)
    if (strict) for (tensor in list(result$weights,result$output,gradients[[2L]],gradients[[3L]])) {
        values <- tensor$values();stopifnot(all(tail(values,length(values)%/%2L)==0))
    }
    if (one_key) stopifnot(all(abs(result$weights$values()-1)<1e-7))
    arrays$native_fd <- nn$from_vector_double(c(count,error),2L)
    dir.create(args$out,recursive=TRUE,showWarnings=FALSE);nn$save_npz(file.path(args$out,paste0(name,".npz")),arrays)
    cat(name,": Q",qs,"K",ks,"V",vs,"native FD=",count,"max error=",error,"PASS\n")
    fixture <- file.path(args$tf,paste0(name,".npz"))
    if (file.exists(fixture)) {
        reference <- nn$load_npz(fixture);inputs <- c("q","k","v","probe")
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
    args <- nn_options(arguments,list(out=file.path(nn_output_root(),"scaled_dot_attention"),tf=file.path(path.expand("~"),"projects/tflow_examples/output/models/scaled_dot_attention"),verbose=0L,epochs=5L,full=FALSE))
    nn$set_num_threads(2L)
    for (entry in list(list("cross",FALSE,FALSE,FALSE,45501L),list("masked_heads",TRUE,TRUE,FALSE,45503L),list("one_key",FALSE,FALSE,TRUE,45505L),list("biased",FALSE,FALSE,FALSE,45507L))) do.call(exercise_case,c(list(args),entry))
    rejected <- tryCatch({invalid <- nn$ones(c(1L,1L,0L));nn$scaled_dot_product_attention(invalid,invalid,nn$ones(c(1L,1L,1L)),nn$Tensor());FALSE},error=function(e) TRUE)
    stopifnot(rejected);cat("Scaled dot-product: all native derivative checks and dimension guard PASS\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
