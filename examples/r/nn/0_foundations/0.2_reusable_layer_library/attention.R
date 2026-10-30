# Native R NN example. Attention references, scaling, gradients and lookup training.
# CMD: nix develop .#r --command Rscript examples/r/nn/0_foundations/0.2_reusable_layer_library/attention.R
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

random_tensor <- function(shape,rng) {rng$reset_normal();nn$from_vector(rng$normal32_values(prod(shape)),shape)}
rows <- function(tensor,example=0L) {
    count <- tensor$size(-2L);width <- tensor$size(-1L)
    values <- tensor$values();matrix(values[example*count*width+seq_len(count*width)],nrow=count,byrow=TRUE)
}
softmax <- function(values) {exponent <- exp(values-max(values));exponent/sum(exponent)}
attend_reference <- function(q,k,v,causal=FALSE) {
    weights <- matrix(0,nrow=nrow(q),ncol=nrow(k));output <- matrix(0,nrow=nrow(q),ncol=ncol(v))
    for (i in seq_len(nrow(q))) {
        scores <- vapply(seq_len(nrow(k)),function(j) sum(q[i,]*k[j,])/sqrt(ncol(q)),numeric(1))
        if (causal) scores[seq_along(scores)>i] <- -1e9
        weight <- softmax(scores);weights[i,] <- weight
        for (a in seq_len(ncol(v))) output[i,a] <- sum(weight*v[,a])
    }
    list(weights=weights,output=output)
}

dot_product <- function() {
    rng <- nn$Mt19937_64(1L)
    q <- random_tensor(c(1L,4L,8L),rng);k <- random_tensor(c(1L,4L,8L),rng);v <- random_tensor(c(1L,4L,3L),rng)
    plain <- nn$scaled_dot_product_attention(q,k,v,nn$Tensor())
    causal <- nn$scaled_dot_product_attention(q,k,v,nn$causal_mask(4L))
    for (is_causal in c(FALSE,TRUE)) {
        result <- if(is_causal) causal else plain
        expected <- attend_reference(rows(q),rows(k),rows(v),is_causal)
        dw <- max(abs(as.numeric(t(expected$weights))-result$weights$values()))
        dout <- max(abs(as.numeric(t(expected$output))-result$output$values()))
        stopifnot(dw<1e-6,dout<1e-6)
        cat(if(is_causal) "Causal" else "Plain","attention hand errors:",dw,dout,"\n")
        for (i in 0:3) {
            row <- vapply(0:3,function(j) result$weights$at(c(0L,i,j)),numeric(1))
            stopifnot(abs(sum(row)-1)<1e-5);cat("Query",i,"weights:",row,"\n")
        }
    }
    config <- nn$AttentionOptions();config$impl <- nn$AttentionImpl$Fused;config$causal <- TRUE
    fused <- nn$scaled_dot_product_attention(q,k,v,config)
    error <- max(abs(fused$output$values()-causal$output$values()))
    stopifnot(error<1e-6);cat("Fused causal output difference:",error,"\n")
}

scaling <- function() {
    rng <- nn$Mt19937_64(2L);trials <- 200L
    for (width in c(4L,16L,64L,256L,1024L)) {
        raw <- scaled <- entropy_raw <- entropy_scaled <- 0
        for (trial in seq_len(trials)) {
            data <- nn$from_vector_double(rng$normal64_values(65L*width),c(65L,width))
            query <- nn$slice_axis(data,0L,0L,1L);keys <- nn$slice_axis(data,0L,1L,65L)
            scores <- nn$matmul(keys,query,FALSE,TRUE)$values()
            raw <- raw+sum(scores*scores)/(64*trials);scaled <- scaled+sum(scores*scores)/width/(64*trials)
            for (scale in c(1,1/sqrt(width))) {
                probabilities <- softmax(scores*scale)
                entropy <- -sum(probabilities[probabilities>0]*log(probabilities[probabilities>0]))/trials
                if (scale==1) entropy_raw <- entropy_raw+entropy else entropy_scaled <- entropy_scaled+entropy
            }
        }
        stopifnot(abs(scaled-1)<.1)
        cat("Scaling d=",width,"raw/scaled variance, raw/scaled entropy:",raw,scaled,entropy_raw,entropy_scaled,"\n")
    }
}

additive <- function() {
    p <- nn$ParameterStore(3L);rng <- nn$Mt19937_64(3L)
    q <- random_tensor(c(2L,5L),rng);k <- random_tensor(c(2L,6L,3L),rng);v <- random_tensor(c(2L,6L,2L),rng)
    result <- nn$additive_attention(p,"add",q,k,v,4L)
    wq <- p$at("add/Wq");wk <- p$at("add/Wk");vector <- p$at("add/v");dw <- dout <- 0
    for (n in 0:1) {
        scores <- numeric(6L)
        for (t in 0:5) {
            score <- 0
            for (u in 0:3) {
                h <- sum(vapply(0:4,function(i) q$at(c(n,i))*wq$at(c(i,u)),numeric(1)))
                h <- h+sum(vapply(0:2,function(i) k$at(c(n,t,i))*wk$at(c(i,u)),numeric(1)))
                score <- score+tanh(h)*vector$at(c(u,0L))
            }
            scores[t+1L] <- score
        }
        weights <- softmax(scores)
        dw <- max(dw,max(abs(weights-vapply(0:5,function(t) result$weights$at(c(n,t)),numeric(1)))))
        for (i in 0:1) dout <- max(dout,abs(sum(weights*vapply(0:5,function(t) v$at(c(n,t,i)),numeric(1)))-result$output$at(c(n,i))))
    }
    stopifnot(dw<1e-6,dout<1e-6,p$count()==36L,identical(as.numeric(result$weights$shape()),c(2,6)),identical(as.numeric(result$output$shape()),c(2,2)))
    cat("Additive attention hand errors:",dw,dout,"parameters:",p$count(),"\n")
}

multi_head <- function() {
    p <- nn$ParameterStore(4L);rng <- nn$Mt19937_64(4L);x <- random_tensor(c(2L,5L,16L),rng)
    config <- nn$MultiHeadOptions();config$need_weights <- TRUE
    result <- nn$multi_head_attention(p,"self",x,x,4L,config)
    stopifnot(identical(as.numeric(result$output$shape()),c(2,5,16)),identical(as.numeric(result$weights$shape()),c(2,4,5,5)),p$count()==1024L)
    matrices <- lapply(c("Wq","Wk","Wv","Wo"),function(name) matrix(p$at(paste0("self/",name))$values(),nrow=16L,byrow=TRUE))
    error <- 0
    for (n in 0:1) {
        concatenated <- matrix(0,nrow=5L,ncol=16L);input <- rows(x,n)
        for (h in 0:3) {
            columns <- 4L*h+1:4
            projected <- lapply(matrices[1:3],function(weight) input%*%weight[,columns,drop=FALSE])
            output <- attend_reference(projected[[1L]],projected[[2L]],projected[[3L]])$output
            concatenated[,columns] <- output
        }
        expected <- concatenated%*%matrices[[4L]]
        error <- max(error,max(abs(expected-rows(result$output,n))))
    }
    stopifnot(error<1e-5);cat("Multi-head direct per-head error:",error,"parameters:",p$count(),"\n")
    for (head in 0:3) cat("Head",head,"example-0/query-2 weights:",vapply(0:4,function(j) result$weights$at(c(0L,head,2L,j)),numeric(1)),"\n")
    scores <- random_tensor(5L,rng)$requires_grad_();r <- random_tensor(5L,rng);alpha <- nn$softmax(scores)
    gradient <- nn$grad(nn$sum(nn$mul(alpha,r)),list(scores))[[1L]]
    dot <- sum(alpha$values()*r$values());error <- max(abs(alpha$values()*(r$values()-dot)-gradient$values()))
    stopifnot(error<1e-6);cat("Softmax closed-form gradient error:",error,"\n")
}

lookup <- function(steps) {
    length <- 6L;ids <- 10L;width <- 8L;batch <- 64L;p <- nn$ParameterStore(5L)
    table <- p$get("emb",c(ids,width),nn$init$uniform(.5));cfg <- nn$OptimizerConfig();cfg$kind <- nn$OptimizerKind$Adam
    adam <- nn$Optimizer(p,.02,cfg);rng <- nn$Mt19937_64(5L)
    make <- function(n) {
        keys <- numeric(n*length);queries <- numeric(n);values <- numeric(n*length);targets <- numeric(n)
        for (i in seq_len(n)) {
            permutation <- rng$shuffle(0:(ids-1L));which <- rng$modulo(length)
            value <- rng$normal32_values(length);positions <- (i-1L)*length+seq_len(length)
            keys[positions] <- head(permutation,length);values[positions] <- value
            queries[i] <- permutation[which+1L];targets[i] <- value[which+1L]
        }
        list(keys=nn$from_vector_int64(keys,c(n,length)),query=nn$from_vector_int64(queries),values=nn$from_vector(values,c(n,length)),target=nn$from_vector(targets))
    }
    forward <- function(data) {
        k <- nn$gather(table,data$keys,0L);q <- nn$expand_dims(nn$gather(table,data$query,0L),1L)
        result <- nn$scaled_dot_product_attention(q,k,nn$expand_dims(data$values,-1L),nn$Tensor())
        prediction <- nn$reshape(result$output,-1L)
        list(loss=nn$mean(nn$square(nn$sub(prediction,data$target))),weights=result$weights)
    }
    test <- make(512L);final_weight <- final_mse <- 0
    for (step in seq_len(steps)) {
        data <- make(batch);adam$step(nn$tensor_vector(nn$grad(forward(data)$loss,list(table))))
        if (step==1L||step%%100L==0L||step==steps) {
            evaluated <- nn$no_grad(function() forward(test));on_target <- 0
            for (i in 0:511) for (t in 0:(length-1L)) {
                if (test$keys$at(c(i,t))==test$query$at(i)) on_target <- on_target+evaluated$weights$at(c(i,0L,t))/512
            }
            final_weight <- on_target;final_mse <- evaluated$loss$item()
            cat(sprintf("Lookup step %d: test MSE=%.9g, matching-key attention=%.9g\n",step,final_mse,on_target))
        }
    }
    stopifnot(is.finite(final_mse),final_mse>=0,final_weight>0,final_weight<=1,adam$iterations()==steps)
    if (steps>=600L) stopifnot(final_mse<.005,final_weight>.95)
    nn$no_grad(function() {
        evaluated <- forward(test);cm <- nn$ConfusionMatrix(length)
        for (i in 0:511) {
            keys <- vapply(0:(length-1L),function(t) test$keys$at(c(i,t)),numeric(1))
            truth <- which(keys==test$query$at(i))-1L
            weights <- vapply(0:(length-1L),function(t) evaluated$weights$at(c(i,0L,t)),numeric(1))
            prediction <- which.max(weights)-1L;cm$add(truth,prediction)
        }
        cells <- matrix(cm$to_tensor()$values(),nrow=length,byrow=TRUE);stopifnot(sum(cells)==512L)
        cat("Pointer accuracy:",cm$accuracy(),"\n")
        cat("Confusion matrix (rows=matching key position, columns=predicted position; held-out 512 queries):\n")
        for (row in seq_len(length)) cat(cells[row,],"\n")
    })
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(epochs=5L,full=FALSE))
    if (is.na(args$epochs)||args$epochs<1L) stop("epochs must be positive")
    nn$set_num_threads(2L)
    dot_product();scaling();additive();multi_head();lookup(args$epochs)
    0L
}
if (sys.nframe()==0L) quit(status=main())
