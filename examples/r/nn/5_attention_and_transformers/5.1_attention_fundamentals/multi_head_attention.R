# Native R NN example. Multi-head projections and shared-input derivatives.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

exercise_case <- function(args,name,heads,self_attention,strict,identity,seed) {
    tq <- if(self_attention) 4L else 3L;tk <- if(self_attention) 4L else 5L
    store <- nn$ParameterStore(seed);rng <- nn$Mt19937(seed)
    random <- function(shape,scale=1) nn$mul(nn$from_vector(rng$normal32_values(prod(shape)),shape),scale)
    xq <- random(c(2L,tq,8L));xkv <- if(self_attention) xq else random(c(2L,tk,8L))
    probe <- random(c(2L,tq,8L),.2);xq$requires_grad_();if(!self_attention) xkv$requires_grad_()
    projections <- c("Wq","Wk","Wv","Wo");source <- "nn";reference <- NULL
    fixture <- file.path(args$tf,paste0(name,".npz"))
    if(!identity&&args$init!="nn") {
        if(file.exists(fixture)) {reference <- nn$load_npz(fixture);source <- "TF"}
        else if(args$init=="tf") stop("Required TF fixture missing: ",fixture)
    }
    if(identity||!is.null(reference)) {
        for(key in projections) {
            value <- if(identity) nn$eye(8L) else reference[[paste0("mha/",key)]]
            stopifnot(identical(value$shape(),c(8L,8L)))
            parameter <- store$get(paste0("mha/",key),c(8L,8L),nn$init$constant(0))
            nn$no_grad(function() parameter$copy_(value))
        }
        if(identity) source <- "identity"
    }
    klv <- if(self_attention) c(4,3) else c(4,0);qlv <- if(self_attention) c(4,3) else c(2,3)
    kl <- nn$cast(nn$from_vector(klv),nn$DType$Int32);ql <- nn$cast(nn$from_vector(qlv),nn$DType$Int32)
    mask <- nn$Tensor()
    if(strict) {
        flags <- numeric(2L*tq*tk);i <- 1L
        for(n in 1:2) for(t in 0:(tq-1L)) for(j in 0:(tk-1L)) {
            flags[i] <- as.numeric(t<qlv[n]&&j<klv[n]&&j<=t);i <- i+1L
        }
        mask <- nn$cast(nn$from_vector(flags,c(2L,1L,tq,tk)),nn$DType$Bool)
    }
    config <- nn$MultiHeadOptions();config$mask <- mask;config$need_weights <- TRUE;config$impl <- nn$AttentionImpl$Plain
    forward <- function() nn$multi_head_attention(store,"mha",xq,xkv,heads,config)
    result <- forward();objective <- nn$sum(nn$mul(result$output,probe))
    leaves <- c(lapply(projections,function(key) store$at(paste0("mha/",key))),list(xq),if(!self_attention) list(xkv) else list())
    gradients <- nn$grad(objective,leaves)
    arrays <- list(xq=xq$detach(),xkv=xkv$detach(),probe=probe,output=result$output$detach(),weights=result$weights$detach(),objective=objective$detach())
    for(i in 1:4) arrays[[paste0("gradient_mha/",projections[i])]] <- gradients[[i]]
    arrays$xq_gradient <- gradients[[5L]];if(!self_attention) arrays$xkv_gradient <- gradients[[6L]]
    if(strict) {arrays$allowed <- mask;arrays$key_lengths <- kl;arrays$query_lengths <- ql}
    if(identity) {
        primitive_config <- nn$AttentionOptions();primitive_config$scale <- 1/sqrt(8);primitive_config$impl <- nn$AttentionImpl$Plain
        primitive <- nn$scaled_dot_product_attention(xq,xkv,xkv,primitive_config)$output$detach()
        arrays$primitive <- primitive;stopifnot(max(abs(result$output$values()-primitive$values()))<2e-6)
    }
    for(i in 1:4) arrays[[paste0("mha/",projections[i])]] <- leaves[[i]]$detach()$clone()
    diagnostic <- nn$no_grad(function() {
        error <- 0;count <- 0L;epsilon <- nn$scalar(.002)$item()
        for(leaf in seq_along(leaves)) {
            value <- leaves[[leaf]];expected <- gradients[[leaf]]$values();samples <- if(leaf<=4L) 3L else 6L
            flat <- nn$reshape(value,-1L)
            for(i in 0:(samples-1L)) {
                coordinate <- floor(i*(value$numel()-1)/(samples-1L));cell <- nn$select(flat,0L,coordinate);old <- cell$item()
                cell$fill_(old+epsilon);plus <- nn$sum(nn$mul(forward()$output,probe))$item()
                cell$fill_(old-epsilon);minus <- nn$sum(nn$mul(forward()$output,probe))$item();cell$fill_(old)
                numeric <- nn$scalar(plus-minus)$item()/.004
                error <- max(error,abs(numeric-expected[coordinate+1L]));count <- count+1L
            }
        }
        list(error=error,count=count)
    })
    error <- diagnostic$error;count <- diagnostic$count
    stopifnot(error<2e-3,store$count()==256L,store$size()==4L)
    arrays$metadata <- nn$from_vector_double(c(heads,as.integer(self_attention),count,error),4L)
    dir.create(args$out,recursive=TRUE,showWarnings=FALSE);nn$save_npz(file.path(args$out,paste0(name,".npz")),arrays)
    cat(name,": heads=",heads,"shared input=",self_attention,"weights=",source,"native FD=",count,"max error=",error,"PASS\n")
    fixture <- file.path(args$tf,paste0(name,".npz"))
    if (file.exists(fixture)) {
        reference <- nn$load_npz(fixture);inputs <- c("xq","xkv","probe",paste0("mha/",projections))
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
    args <- nn_options(arguments,list(out=file.path(nn_output_root(),"multi_head_attention"),tf=file.path(path.expand("~"),"projects/tflow_examples/output/models/multi_head_attention"),init="auto",verbose=0L,epochs=5L,full=FALSE))
    stopifnot(args$init %in% c("auto","tf","nn"));nn$set_num_threads(2L)
    for(entry in list(list("cross_two",2L,FALSE,FALSE,FALSE,45601L),list("masked_two",2L,FALSE,TRUE,FALSE,45603L),list("self_four",4L,TRUE,TRUE,FALSE,45605L),list("identity_one",1L,FALSE,FALSE,TRUE,45607L))) do.call(exercise_case,c(list(args),entry))
    for(pair in list(c(8L,0L),c(8L,3L),c(0L,1L))) {
        rejected <- tryCatch({x <- nn$ones(c(1L,1L,pair[1L]));nn$multi_head_attention(nn$ParameterStore(),"bad",x,x,pair[2L]);FALSE},error=function(e) TRUE)
        stopifnot(rejected)
    }
    cat("Multi-head projection/input checks and dimension guards PASS\n")
    0L
}
if(sys.nframe()==0L) quit(status=main())
