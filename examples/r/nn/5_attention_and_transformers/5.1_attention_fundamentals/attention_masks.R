# Native R NN example. Padding/causal masks, cache equivalence and gradient isolation.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

allowed_mask <- function(kl,ql,queries,keys,causal=FALSE,offset=0L) {
    if (queries<1L||keys<1L||offset<0L) stop("positive attention sizes and nonnegative offset required")
    if (kl$dim()!=1L||ql$dim()!=1L||kl$size(0L)!=ql$size(0L)) stop("lengths must be vectors of the batch size")
    batch <- kl$size(0L);flags <- numeric(batch*queries*keys);index <- 1L
    for (n in 0:(batch-1L)) for (t in 0:(queries-1L)) for (j in 0:(keys-1L)) {
        flags[index] <- as.numeric(t<ql$at(n)&&j<kl$at(n)&&(!causal||j<=t+offset));index <- index+1L
    }
    nn$cast(nn$from_vector(flags,c(batch,queries,keys)),nn$DType$Bool)
}

exercise_case <- function(args,name,queries,causal,offset,probe_index,seed) {
    keys <- 5L
    if (name=="padding_cross") {klv <- c(4,2,0);qlv <- c(2,3,1)}
    else if (name=="causal_square") {klv <- c(5,3,0);qlv <- c(5,3,0)}
    else if (queries==1L) {klv <- c(5,3,0);qlv <- c(1,1,1)}
    else {klv <- c(5,4,0);qlv <- c(2,1,2)}
    kl <- nn$cast(nn$from_vector(klv),nn$DType$Int32);ql <- nn$cast(nn$from_vector(qlv),nn$DType$Int32)
    rng <- nn$Mt19937(seed)
    random <- function(shape,scale=1) nn$mul(nn$from_vector(rng$normal32_values(prod(shape)),shape),scale)
    q <- random(c(3L,queries,4L));k <- random(c(3L,keys,4L));v <- random(c(3L,keys,3L));probe <- random(c(3L,queries,3L),.2)
    if (probe_index>=0L) {
        keep <- nn$expand_dims(nn$equal(nn$arange(0,queries),probe_index),1L)
        probe <- nn$where(keep,probe,nn$zeros_like(probe))
    }
    q$requires_grad_();k$requires_grad_();v$requires_grad_()
    allowed <- allowed_mask(kl,ql,queries,keys,causal,offset)
    config <- nn$AttentionOptions();config$mask <- allowed;config$scale <- .5;config$need_weights <- TRUE;config$impl <- nn$AttentionImpl$Plain
    forward <- function() nn$scaled_dot_product_attention(q,k,v,config)
    result <- forward();objective <- nn$sum(nn$mul(result$output,probe));gradients <- nn$grad(objective,list(q,k,v))
    arrays <- list(q=q$detach(),k=k$detach(),v=v$detach(),probe=probe,key_lengths=kl,query_lengths=ql,
        output=result$output$detach(),weights=result$weights$detach(),allowed=allowed,objective=objective$detach(),
        q_gradient=gradients[[1L]],k_gradient=gradients[[2L]],v_gradient=gradients[[3L]])
    for (n in 0:2) for (j in 0:(keys-1L)) {
        seen <- any(vapply(0:(queries-1L),function(t) allowed$at(c(n,t,j))!=0&&probe$at(c(n,t,0L))!=0,logical(1)))
        if (!seen) {
            stopifnot(all(vapply(0:3,function(c) gradients[[2L]]$at(c(n,j,c))==0,logical(1))))
            stopifnot(all(vapply(0:2,function(c) gradients[[3L]]$at(c(n,j,c))==0,logical(1))))
        }
    }
    if (offset>0L) {
        values <- numeric(3L*keys*4L)
        for (n in 0:2) for (t in 0:(queries-1L)) for (j in 0:3) values[(n*keys+offset+t)*4L+j+1L] <- q$at(c(n,t,j))
        full_q <- nn$from_vector(values,c(3L,keys,4L));full_ql <- nn$cast(nn$from_vector(offset+qlv),nn$DType$Int32)
        full_config <- nn$AttentionOptions();full_config$mask <- allowed_mask(kl,full_ql,keys,keys,TRUE)
        full_config$scale <- .5;full_config$need_weights <- TRUE;full_config$impl <- nn$AttentionImpl$Plain
        full <- nn$scaled_dot_product_attention(full_q,k,v,full_config)$output
        arrays$full_q <- full_q;arrays$full_query_lengths <- full_ql;arrays$full_output <- full$detach()
        for (n in 0:2) for (t in 0:(queries-1L)) for (j in 0:2) stopifnot(abs(result$output$at(c(n,t,j))-full$at(c(n,offset+t,j)))<2e-6)
    }
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
    arrays$metadata <- nn$from_vector_double(c(as.numeric(causal),offset,probe_index,count,error),5L)
    dir.create(args$out,recursive=TRUE,showWarnings=FALSE);nn$save_npz(file.path(args$out,paste0(name,".npz")),arrays)
    cat(name,"queries=",queries,"keys=5 offset=",offset,"native FD=",count,"max error=",error,"PASS\n")
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
    args <- nn_options(arguments,list(out=file.path(nn_output_root(),"attention_masks"),tf=file.path(path.expand("~"),"projects/tflow_examples/output/models/attention_masks"),verbose=0L,epochs=5L,full=FALSE))
    nn$set_num_threads(2L)
    for (entry in list(list("padding_cross",3L,FALSE,0L,-1L,45701L),list("causal_square",5L,TRUE,0L,2L,45703L),list("cached_one",1L,TRUE,4L,-1L,45705L),list("cached_chunk",2L,TRUE,3L,0L,45707L))) do.call(exercise_case,c(list(args),entry))
    lengths <- nn$cast(nn$from_vector(1),nn$DType$Int32)
    for (entry in list(c(0L,5L,0L),c(3L,0L,0L),c(3L,5L,-1L))) {
        rejected <- tryCatch({allowed_mask(lengths,lengths,entry[1L],entry[2L],TRUE,entry[3L]);FALSE},error=function(e) TRUE)
        stopifnot(rejected)
    }
    cat("Padding/causal/query masks and full/cache equivalence native PASS\n")
    0L
}
if (sys.nframe()==0L) quit(status=main())
