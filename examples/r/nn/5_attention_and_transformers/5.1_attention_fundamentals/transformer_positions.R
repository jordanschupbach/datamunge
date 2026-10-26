# Native R NN example. Absolute, rotary, ALiBi and T5 positions with full derivatives.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

sinusoidal <- function(length,width) {
    if(length<1L||width<2L||width%%2L!=0L) stop("positive length and even width required")
    nn$sinusoidal_encoding(length,width)
}
rotary <- function(x,offset=0L) {
    if(x$dim()<2L||x$size(-2L)<1L||x$size(-1L)<2L||x$size(-1L)%%2L!=0L||offset<0L) stop("invalid rotary dimensions or offset")
    nn$apply_rotary(x,offset)
}
slopes <- function(heads) {
    if(heads<1L) stop("positive head count required")
    power <- 1L;while(power<=heads%/%2L) power <- power*2L
    result <- 2^(-8*seq_len(power)/power)
    if(heads>power) result <- c(result,2^(-4*seq(1L,2L*(heads-power)-1L,by=2L)/power))
    nn$from_vector(result)$values()
}
bucket <- function(relative,bidirectional=TRUE,buckets=32L,max_distance=128L) {
    if(buckets<4L||(bidirectional&&buckets%%2L!=0L)) stop("invalid bucket count")
    bins <- if(bidirectional) buckets%/%2L else buckets;exact <- bins%/%2L
    if(max_distance<=exact) stop("maximum distance must exceed exact range")
    distance <- -relative;side <- if(bidirectional&&distance<0L) bins else 0L
    distance <- if(bidirectional) abs(distance) else max(distance,0L)
    value <- if(distance<exact) distance else min(bins-1L,exact+as.integer(log(distance/exact)/log(max_distance/exact)*(bins-exact)))
    side+value
}
buckets_grid <- function(queries,keys,bidirectional=TRUE,offset=0L) {
    if(queries<1L||keys<1L||offset<0L) stop("invalid lengths or offset")
    values <- unlist(lapply(0:(queries-1L),function(q) vapply(0:(keys-1L),function(k) as.numeric(bucket(k-q-offset,bidirectional,16L,32L)),numeric(1))))
    nn$cast(nn$from_vector(values,c(queries,keys)),nn$DType$Int64)
}
exercise <- function(args) {
    rng <- nn$Mt19937(45801L);store <- nn$ParameterStore(45803L)
    random <- function(shape) nn$from_vector(rng$normal32_values(prod(shape)),shape)
    x <- random(c(2L,5L,8L));probe <- random(c(2L,5L,8L));source <- "nn"
    fixture <- file.path(args$tf,"positions.npz")
    if(args$init!="nn"&&file.exists(fixture)) {
        reference <- nn$load_npz(fixture)
        for(entry in list(list("absolute/pos",c(9L,8L),"learned_table"),list("relative",c(16L,3L),"relative_table"))) {
            parameter <- store$get(entry[[1L]],entry[[2L]],nn$init$constant(0));value <- reference[[entry[[3L]]]]
            stopifnot(identical(value$shape(),entry[[2L]]));nn$no_grad(function() parameter$copy_(value))
        }
        source <- "TF"
    } else if(args$init=="tf") stop("Required TF fixture missing: ",fixture)
    sin <- sinusoidal(5L,8L);table <- nn$learned_encoding(store,"absolute",9L,8L,.02);reltable <- store$get("relative",c(16L,3L),nn$init$normal(.2))
    indices <- nn$cast(nn$from_vector(c(0,2,2,5,7)),nn$DType$Int32)
    q <- random(c(3L,5L,8L));k <- random(c(3L,5L,8L));v <- random(c(3L,5L,4L));ap <- random(c(3L,5L,4L));x$requires_grad_()
    relative <- function(bidirectional) nn$transpose(nn$gather(reltable,buckets_grid(5L,5L,bidirectional,3L),0L),c(2L,0L,1L))
    forward <- function() {
        absolute <- nn$add(x,sin);learned <- nn$add(x,nn$gather(table,indices,0L));rotated <- rotary(x,3L)
        bias <- relative(TRUE);causal <- relative(FALSE)
        config <- nn$AttentionOptions();config$mask <- bias;config$scale <- 1/sqrt(8);config$need_weights <- TRUE;config$impl <- nn$AttentionImpl$Plain
        attention <- nn$scaled_dot_product_attention(q,k,v,config)
        objective <- nn$add(nn$add(nn$sum(nn$mul(absolute,probe)),nn$sum(nn$mul(learned,probe))),nn$sum(nn$mul(rotated,probe)))
        objective <- nn$add(objective,nn$sum(nn$mul(attention$output,ap)))
        list(absolute=absolute,learned=learned,rotary=rotated,relative=bias,relative_causal=causal,attention=attention,objective=objective)
    }
    o <- forward();bias <- numeric(75L);i <- 1L
    for(s in slopes(3L)) for(m in 0:4) for(n in 0:4) {bias[i] <- if(n>m) -1e9 else -s*abs(m-n)+0;i <- i+1L}
    alibi <- nn$from_vector(bias,c(3L,5L,5L));gradients <- nn$grad(o$objective,list(x,table,reltable))
    cached <- rotary(nn$slice(x$detach(),c(0L,2L,0L),c(2L,3L,8L)),5L)
    dots <- function(offset) nn$matmul(rotary(q,offset),rotary(k,offset),FALSE,TRUE)
    dot0 <- dots(0L);dot7 <- dots(7L)
    arrays <- list(probe=probe,indices=indices,q=q,k=k,v=v,attention_probe=ap,x=x$detach()$clone(),learned_table=table$detach()$clone(),relative_table=reltable$detach()$clone(),sinusoidal=sin,absolute=o$absolute$detach(),learned=o$learned$detach(),rotary=o$rotary$detach(),alibi=alibi,relative=o$relative$detach(),relative_causal=o$relative_causal$detach(),weights=o$attention$weights$detach(),output=o$attention$output$detach(),objective=o$objective$detach(),x_gradient=gradients[[1L]],learned_gradient=gradients[[2L]],relative_gradient=gradients[[3L]],rotary_cached=cached,rotary_dot0=dot0,rotary_dot7=dot7)
    grid <- unlist(lapply(c(FALSE,TRUE),function(b) vapply(-256:256,function(r) as.numeric(bucket(r,b,16L,32L)),numeric(1))))
    arrays$bucket_grid <- nn$cast(nn$from_vector(grid,c(2L,513L)),nn$DType$Int32)
    for(heads in c(1L,2L,3L,4L,6L,8L)) {
        values <- slopes(heads);arrays[[paste0("slopes_",heads)]] <- nn$from_vector(values,heads)
        if(bitwAnd(heads,heads-1L)==0L) stopifnot(max(abs(nn$alibi_slopes(heads)-values))<1e-7)
    }
    difference <- function(a,b) max(abs(a$values()-b$values()))
    stopifnot(difference(cached,nn$slice(o$rotary$detach(),c(0L,2L,0L),c(2L,3L,8L)))<1e-6,difference(dot0,dot7)<1e-5)
    for(b in c(TRUE,FALSE)) stopifnot(difference(nn$relative_position_buckets(5L,5L,b,16L,32L),buckets_grid(5L,5L,b))==0)
    diagnostic <- nn$no_grad(function() {
        count <- 0L;error <- 0;epsilon <- nn$scalar(.002)$item();leaves <- list(x,table,reltable)
        for(leaf in 1:3) {
            value <- leaves[[leaf]];flat <- nn$reshape(value,-1L);expected <- gradients[[leaf]]$values()
            for(i in 0:(value$numel()-1L)) {
                cell <- nn$select(flat,0L,i);old <- cell$item()
                cell$fill_(old+epsilon);plus <- forward()$objective$item()
                cell$fill_(old-epsilon);minus <- forward()$objective$item();cell$fill_(old)
                error <- max(error,abs(nn$scalar(plus-minus)$item()/.004-expected[i+1L]));count <- count+1L
            }
        }
        list(error=error,count=count)
    })
    count <- diagnostic$count;error <- diagnostic$error;stopifnot(count==200L,error<.006)
    arrays$native_fd <- nn$from_vector_double(c(count,error),2L)
    dir.create(args$out,recursive=TRUE,showWarnings=FALSE);nn$save_npz(file.path(args$out,"positions.npz"),arrays)
    guards <- list(function() sinusoidal(5L,0L),function() sinusoidal(5L,3L),function() rotary(nn$slice(x$detach(),c(0L,0L,0L),c(2L,5L,7L))),function() rotary(x$detach(),-1L),function() slopes(0L),function() bucket(1L,TRUE,5L),function() bucket(1L,TRUE,16L,4L))
    for(check in guards) stopifnot(tryCatch({check();FALSE},error=function(e) TRUE))
    cat("Five position mechanisms (tables from",source,"):",count,"native FD coordinates, max error",error,"; seven invalid inputs rejected PASS\n")
    fixture <- file.path(args$tf,"positions.npz")
    if (file.exists(fixture)) {
        reference <- nn$load_npz(fixture);inputs <- c("x","probe","q","k","v","attention_probe","learned_table","relative_table")
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
    args <- nn_options(arguments,list(out=file.path(nn_output_root(),"transformer_positions"),tf=file.path(path.expand("~"),"projects/tflow_examples/output/models/transformer_positions"),init="auto",verbose=0L,epochs=5L,full=FALSE))
    stopifnot(args$init %in% c("auto","tf","nn"));nn$set_num_threads(2L);exercise(args)
    0L
}
if(sys.nframe()==0L) quit(status=main())
