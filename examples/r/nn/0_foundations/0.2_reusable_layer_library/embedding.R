# Native R NN example. Embedding lookup, sparse gradients, masking and skip-gram.
.nn_common <- file.path(Sys.getenv("DATAMUNGE_EXAMPLES_ROOT",getwd()),"examples/r/nn/common.R")
if (!file.exists(.nn_common)) .nn_common <- system.file("examples/nn/common.R",package="datamungerNN",mustWork=TRUE)
source(.nn_common,local=TRUE)
rm(.nn_common)

lookup_and_gradient <- function() {
    p <- nn$ParameterStore(1L)
    ids <- nn$from_vector_int64(c(3,7,3,0,9,7),c(2L,3L))
    r <- nn$from_vector(.5*((0:23)%%5L)-1,c(2L,3L,4L))
    e <- nn$embedding(p,"words",ids,10L,4L,1)
    table <- p$at("words/table")
    onehot <- nn$reshape(nn$matmul(nn$one_hot(nn$reshape(ids,-1L),10L),table),c(2L,3L,4L))
    gradient <- nn$grad(nn$sum(nn$mul(e,r)),list(table))[[1L]]
    stopifnot(identical(as.numeric(e$shape()),c(2,3,4)),identical(e$values(),onehot$values()))
    worst <- 0
    for (word in 0:9) {
        expected <- numeric(4L)
        for (k in 0:3) for (i in 0:1) for (j in 0:2) {
            if (ids$at(c(i,j))==word) expected[k+1L] <- expected[k+1L]+r$at(c(i,j,k))
        }
        actual <- vapply(0:3,function(k) gradient$at(c(word,k)),numeric(1))
        worst <- max(worst,max(abs(actual-expected)))
        cat("Gradient row",word,actual,"expected:",expected,"\n")
    }
    stopifnot(worst<1e-6)
    cat("Lookup equals one-hot product; repeated ids accumulate; unused rows have zero gradient.\n")
}

masking <- function() {
    p <- nn$ParameterStore(2L)
    ids <- nn$from_vector_int64(c(4,2,0,0,1,5,9,3),c(2L,4L))
    e <- nn$embedding(p,"words",ids,10L,3L,1)
    pooled <- nn$masked_mean(e,nn$not_equal(ids,0));naive <- nn$mean(e,1L);worst <- 0
    for (i in 0:1) {
        positions <- (0:3)[vapply(0:3,function(j) ids$at(c(i,j))!=0,logical(1))]
        expected <- vapply(0:2,function(k) sum(vapply(positions,function(j) e$at(c(i,j,k)),numeric(1)))/length(positions),numeric(1))
        actual <- vapply(0:2,function(k) pooled$at(c(i,k)),numeric(1))
        worst <- max(worst,max(abs(actual-expected)))
        cat("Sequence",i,"real tokens:",length(positions),"masked:",actual,"hand:",expected,"unmasked:",vapply(0:2,function(k) naive$at(c(i,k)),numeric(1)),"\n")
    }
    stopifnot(worst<1e-6,max(abs(nn$sub(nn$select(pooled,0L,1L),nn$select(naive,0L,1L))$values()))<1e-6)
}

skip_gram <- function(steps) {
    vocab <- 12L;dim <- 8L;window <- 2L;rng <- nn$Mt19937_64(3L)
    pairs <- matrix(0L,nrow=10400L,ncol=2L);cursor <- 1L
    for (s in 1:400) {
        topic <- rng$modulo(3L)
        sentence <- vapply(1:8,function(i) 4*topic+rng$modulo(4L),numeric(1))
        for (i in 0:7) for (j in max(0L,i-window):min(7L,i+window)) {
            if (i!=j) {pairs[cursor,] <- c(sentence[i+1L],sentence[j+1L]);cursor <- cursor+1L}
        }
    }
    stopifnot(cursor==10401L)
    cat("Synthetic corpus:",nrow(pairs),"target/context pairs in three topics.\n")
    p <- nn$ParameterStore(4L)
    logits <- function(target) {
        e <- nn$embedding(p,"in",target,vocab,dim,.5)
        out <- p$get("out/table",c(vocab,dim),nn$init$uniform(.5))
        nn$matmul(e,out,FALSE,TRUE)
    }
    nn$no_grad(function() logits(nn$from_vector_int64(rep(0,256L))))
    cfg <- nn$OptimizerConfig();cfg$kind <- nn$OptimizerKind$Adam
    adam <- nn$Optimizer(p,.05,cfg)
    evaluate <- function() nn$no_grad(function() {
        total <- 0;cm <- nn$ConfusionMatrix(vocab)
        for (start in seq.int(1L,nrow(pairs),512L)) {
            batch <- pairs[start:min(start+511L,nrow(pairs)),,drop=FALSE]
            target <- nn$from_vector_int64(batch[,1L]);context <- nn$from_vector_int64(batch[,2L])
            scores <- logits(target)
            total <- total+nrow(batch)*nn$sparse_softmax_cross_entropy(scores,context)$item()
            cm$add(context,nn$argmax(scores,1L))
        }
        list(loss=total/nrow(pairs),confusion=cm)
    })
    initial <- evaluate()$loss;last <- Inf
    for (step in seq_len(steps)) {
        rows <- vapply(seq_len(256L),function(i) rng$modulo(nrow(pairs))+1,numeric(1))
        batch <- pairs[rows,,drop=FALSE]
        target <- nn$from_vector_int64(batch[,1L]);context <- nn$from_vector_int64(batch[,2L])
        loss <- nn$sparse_softmax_cross_entropy(logits(target),context)
        adam$minimize(loss);last <- loss$item()
        cat(sprintf("skip-gram step %d/%d: NLL=%.9g; floor log(4)=%.9g\n",step,steps,last,log(4)))
    }
    table <- p$at("in/table");within <- numeric();between <- numeric()
    cat("Cosine similarity (words 0-11, topics 0-3 / 4-7 / 8-11):\n")
    for (a in 0:(vocab-1L)) {
        row <- numeric(vocab)
        for (b in 0:(vocab-1L)) {
            av <- vapply(0:(dim-1L),function(k) table$at(c(a,k)),numeric(1))
            bv <- vapply(0:(dim-1L),function(k) table$at(c(b,k)),numeric(1))
            cosine <- sum(av*bv)/sqrt(sum(av*av)*sum(bv*bv));row[b+1L] <- cosine
            if (a!=b) {if(a%/%4L==b%/%4L) within <- c(within,cosine) else between <- c(between,cosine)}
        }
        cat(a,sprintf("%.6f",row),"\n")
    }
    mean_within <- mean(within);mean_between <- mean(between)
    cat("Mean cosine within topic:",mean_within,"between topics:",mean_between,"\n")
    if (steps>=300L) stopifnot(last<log(4)+.06,mean_within>.8,mean_between<-.2)
    final <- evaluate();cm <- final$confusion
    stopifnot(is.finite(final$loss),final$loss<initial,adam$iterations()==steps)
    cells <- matrix(cm$to_tensor()$values(),nrow=vocab,byrow=TRUE)
    stopifnot(sum(cells)==nrow(pairs))
    cat(sprintf("Context-word evaluation NLL=%.9g; accuracy=%.9g\n",final$loss,cm$accuracy()))
    cat("Confusion matrix (rows=true context, columns=predicted context; synthetic corpus):\n")
    for (row in seq_len(vocab)) cat(cells[row,],"\n")
}

main <- function(arguments=commandArgs(TRUE)) {
    args <- nn_options(arguments,list(epochs=5L,full=FALSE))
    if (is.na(args$epochs)||args$epochs<1L) stop("epochs must be positive")
    nn$set_num_threads(2L)
    lookup_and_gradient();masking();skip_gram(args$epochs)
    0L
}
if (sys.nframe()==0L) quit(status=main())
